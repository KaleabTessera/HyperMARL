"""
MAPPO with Full Parameter Sharing (FuPS) + Non-Jax Cpu Envs
"""

import multiprocessing as mp
from collections import defaultdict
import time
from typing import Any, Dict, List, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.struct import dataclass
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

import wandb
from baselines.MAPPO.mappo import get_update_fn
from baselines.utils.utils import (
    calculate_team_diversity,
    log_eval_metrics,
    log_train_metrics,
    log_videos,
)
from baselines.utils.eval import run_eval_jax
from baselines.utils.wrappers import make_env

# Set up multiprocessing with fork server for better performance
forkserver_available = "forkserver" in mp.get_all_start_methods()
start_method = "forkserver" if forkserver_available else "spawn"
mp.set_start_method(start_method, force=True)


@dataclass
class EpisodeStatistics:
    """
    Tracks statistics about episodes during training.
    
    Attributes:
        episode_returns: Cumulative rewards for ongoing episodes
        episode_lengths: Number of steps taken in ongoing episodes
        returned_episode_returns: Returns of completed episodes
        returned_episode_lengths: Lengths of completed episodes
    """
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class ActorCritic(nn.Module):
    """
    Actor-Critic network with parameter sharing across agents.
    
    Both actor and critic are implemented as separate networks within this module.
    The actor makes decisions based on local observations, while the critic 
    evaluates based on the global observation (all agents' observations).
    
    Attributes:
        action_dim: Dimension of action space
        num_agents: Number of agents in the environment
        activation: Activation function to use (tanh or relu)
        actor_layers: Sizes of hidden layers for actor network
        critic_layers: Sizes of hidden layers for critic network
    """
    action_dim: int
    num_agents: int
    activation: str = "tanh"
    actor_layers: List[int] = (64, 64)
    critic_layers: List[int] = (64, 64)

    @nn.compact
    def __call__(self, x, x_critic):
        """
        Forward pass through the actor-critic network.
        
        Args:
            x: Agent's local observation
            x_critic: Global observation (concatenated observations of all agents)
            
        Returns:
            actor_logits: Action logits for policy distribution
            critic_value: Value function estimate
        """
        # Select activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        def build_network(x, layer_sizes: List[int], activation):
            """Build a multi-layer perceptron with specified architecture."""
            for size in layer_sizes:
                x = nn.Dense(
                    size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
                x = activation(x)
            return x

        # Actor network (policy)
        actor_hidden = build_network(x, self.actor_layers, activation)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_hidden)

        # Critic network (value function)
        critic = build_network(x_critic, self.critic_layers, activation)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return actor_logits, jnp.squeeze(critic, axis=-1)


class TransitionInfo(NamedTuple):
    """
    Information about completed episodes during transitions.
    
    Attributes:
        returned_episode_returns: Returns of completed episodes
        returned_episode_lengths: Lengths of completed episodes
    """
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@dataclass
class Transition:
    """
    Stores a single step transition for training.
    
    Contains all necessary information for PPO updates.
    
    Attributes:
        done: Done flags for each agent
        action: Actions taken by each agent
        value: Value estimates
        reward: Rewards received
        log_prob: Log probabilities of actions
        obs: Observations (local)
        global_obs: Global observations (all agents)
        info: Additional episode information
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    info: TransitionInfo


def initialize_metrics_storage(config, metric_keys):
    """
    Initialize storage for training metrics.
    
    Args:
        config: Configuration dictionary
        metric_keys: Keys of metrics to track
        
    Returns:
        Dictionary mapping metric names to arrays
    """
    return {k: np.zeros(config["NUM_UPDATES"]) for k in metric_keys}


@jax.jit
def update_metrics(metrics: dict, new_values: dict, update_idx: int) -> dict:
    """
    Update metrics with new values at specified index.
    
    Args:
        metrics: Current metrics dictionary
        new_values: New values to insert
        update_idx: Index to update
        
    Returns:
        Updated metrics dictionary
    """
    def _update(arr, val):
        if isinstance(arr, jnp.ndarray):
            return arr.at[update_idx].set(val)
        elif isinstance(val, dict):
            return jax.tree_map(lambda a, v: _update(a, v), arr, val)
        return val

    return jax.tree_map(lambda x, y: _update(x, y), metrics, new_values)


def make_train(config):
    """
    Create the training function based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train: Function that performs training
    """
    # Initialize environment
    env, possible_agents, action_dim, num_actions, observation_size = make_env(
        config["ENV_NAME"], num_envs=config["NUM_ENVS"], **config["TRAIN_ENV_KWARGS"]
    )
    
    # Calculate derived parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        """Linear learning rate decay schedule."""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, wb_run=None):
        """
        Main training function.
        
        Args:
            rng: JAX random number generator key
            wb_run: Weights & Biases run object
            
        Returns:
            Dictionary of training results and environment
        """
        # Initialize network
        network = ActorCritic(
            action_dim,
            activation=config["ACTIVATION"],
            actor_layers=config.get("ACTOR_LAYERS"),
            critic_layers=config.get("CRITIC_LAYERS"),
            num_agents=env.num_agents,
        )
        
        # Initialize network parameters
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(observation_size)
        num_agents = len(env.agents)
        critic_obs_size = observation_size * env.num_agents
        init_x_critic = jnp.zeros((critic_obs_size))
        network_params = network.init(_rng, init_x, init_x_critic)

        # Configure optimizer with optional learning rate annealing
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # Create train state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # JIT-compile the network apply function
        network.apply = jax.jit(network.apply)

        # Load checkpoint if specified
        zero_shot_eval = False
        if config.get("CHECKPOINT_LOAD_DIR") is not None:
            print(f"Loading from {config.get('CHECKPOINT_LOAD_DIR')}")
            load_checkpointer = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename="checkpoints")
            )
            loaded_checkpoints = load_checkpointer.restore(
                config.get("CHECKPOINT_LOAD_DIR"), item=train_state.params
            )
            train_state = TrainState.create(
                apply_fn=train_state.apply_fn, params=loaded_checkpoints, tx=tx
            )
            zero_shot_eval = True

        # Initialize episode statistics
        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
            returned_episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            returned_episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
        )

        # Preallocate arrays for transitions (more efficient memory usage)
        transitions = Transition(
            done=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=bool
            ),
            action=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                dtype=jnp.float32,
            ),
            value=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            reward=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            log_prob=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            obs=np.zeros(
                (
                    config["NUM_STEPS"],
                    config["NUM_ENVS"] * num_agents,
                    observation_size,
                ),
                dtype=jnp.float32,
            ),
            global_obs=np.zeros(
                (
                    config["NUM_STEPS"],
                    config["NUM_ENVS"] * num_agents,
                    critic_obs_size,
                ),
                dtype=jnp.float32,
            ),
            info=TransitionInfo(
                returned_episode_returns=np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.float32,
                ),
                returned_episode_lengths=np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.int32,
                ),
            ),
        )

        @jax.jit
        def concat_local_obs_into_global_obs(obs):
            """
            Create global observations by concatenating all agents' observations.
            
            Args:
                obs: Local observations
                
            Returns:
                Global observations for each agent
            """
            reshaped_obs = obs.reshape((config["NUM_ENVS"], num_agents, -1))
            obs_dim = reshaped_obs.shape[-1]
            global_obs = jnp.zeros(
                (config["NUM_ENVS"] * num_agents, num_agents * obs_dim)
            )
            
            # For each environment, create the global observation
            for i in range(config["NUM_ENVS"]):
                flat_global_obs_per_env = reshaped_obs[i].flatten()
                env_begin = i * num_agents
                env_end = env_begin + num_agents
                global_obs = global_obs.at[env_begin:env_end].set(
                    jnp.tile(flat_global_obs_per_env, (num_agents,)).reshape(
                        num_agents, num_agents * obs_dim
                    )
                )
            return global_obs

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        int_seed = jax.random.randint(
            reset_rng, shape=(1,), minval=1, maxval=1000000
        ).item()
        obsv, infos = env.reset(seed=int_seed)
        global_obsv = concat_local_obs_into_global_obs(obsv)
        env_state = {}
        env_step = env.step

        def step_env_wrapped(action: Any) -> Any:
            """
            Step the environment and process the results.
            
            Args:
                action: Actions to take
                
            Returns:
                next_obs: Next observations
                reward: Rewards received
                termination: Terminal states
                truncs: Truncation flags
                info: Additional info
                global_obs: Global observations
            """
            next_obs, reward, termination, truncs, info = env_step(action.flatten())
            global_obs = concat_local_obs_into_global_obs(next_obs)
            return next_obs, reward, termination, truncs, info, global_obs

        @jax.jit
        def update_episode_stats(episode_stats, reward, done):
            """
            Update episode statistics after a step.
            
            Args:
                episode_stats: Current episode statistics
                reward: Rewards received
                done: Done flags
                
            Returns:
                Updated episode statistics
            """
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            returned_episode_returns = jnp.where(
                done, new_episode_return, episode_stats.returned_episode_returns
            )
            returned_episode_lengths = jnp.where(
                done, new_episode_length, episode_stats.returned_episode_lengths
            )

            return episode_stats.replace(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=returned_episode_returns,
                returned_episode_lengths=returned_episode_lengths,
            )

        @jax.jit
        def _select_action(params, obs, obs_critic, _rng):
            """
            Select actions for all agents.
            
            Args:
                params: Network parameters
                obs: Agent observations
                obs_critic: Global observations
                _rng: Random number generator key
                
            Returns:
                action: Selected actions
                log_prob: Log probabilities of actions
                value: Value function estimates
            """
            actor_logits, value = network.apply(params, obs, obs_critic)
            pi = distrax.Categorical(logits=actor_logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            return action, log_prob, value

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            """
            Select deterministic actions for evaluation.
            
            Args:
                params: Network parameters
                obs: Observations
                _rng: Random number generator key
                
            Returns:
                actions: Selected actions
            """
            # Create dummy global observations for critic
            dummy_obs_critic = jnp.zeros((obs.shape[0], critic_obs_size)) 
            actor_logits, _ = network.apply(params, obs, dummy_obs_critic)
            pi = distrax.Categorical(logits=actor_logits)
            
            # Use either stochastic or deterministic actions based on config
            if config.get("eval_stochastic"):
                action = pi.sample(seed=_rng)
            else:
                action = pi.mode()
                
            return (action,)

        # Get PPO update function
        calculate_advantage_and_update_ppo = get_update_fn(config, network)

        def _update_step(runner_state, unused):
            """
            Perform a single training update step.
            
            Args:
                runner_state: Current runner state
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated runner state and metrics
            """
            train_state, env_state, last_obs, last_global_obs, rng, episode_stats = runner_state

            # Collect transitions
            for t in range(config["NUM_STEPS"]):
                # Select actions
                rng, _rng = jax.random.split(rng)
                action, log_prob, value = _select_action(
                    train_state.params, last_obs, last_global_obs, _rng
                )

                # Step environment
                np_action = np.array(action)
                obsv, reward, termination, truncs, info, global_obs = step_env_wrapped(np_action)

                # Update episode statistics
                done = np.logical_or(termination, truncs)
                episode_stats = update_episode_stats(episode_stats, reward, done)

                # Store transitions
                transitions.done[t] = done
                transitions.action[t] = action
                transitions.value[t] = value
                transitions.reward[t] = reward
                transitions.log_prob[t] = log_prob
                transitions.obs[t] = last_obs
                transitions.global_obs[t] = last_global_obs
                transitions.info.returned_episode_returns[t] = (
                    episode_stats.returned_episode_returns
                )
                transitions.info.returned_episode_lengths[t] = (
                    episode_stats.returned_episode_lengths
                )

                # Update observations
                last_obs = obsv
                last_global_obs = global_obs

            # Update policy
            update_state, loss_info = calculate_advantage_and_update_ppo(
                transitions, last_obs, last_global_obs, train_state, rng
            )
            train_state = update_state[0]
            
            # Collect metrics
            metric = {
                "returned_episode_returns": episode_stats.returned_episode_returns,
                "returned_episode_lengths": episode_stats.returned_episode_lengths,
            }
            rng = update_state[-1]

            # Average metrics
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info}

            runner_state = (train_state, env_state, last_obs, last_global_obs, rng, episode_stats)
            return runner_state, metric

        # Initialize runner state
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, global_obsv, _rng, episode_stats)

        # Set up checkpointer
        checkpointers = checkpointer.Checkpointer(
            PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
        )

        # Perform zero-shot evaluation if loading from checkpoint
        if zero_shot_eval:
            print("Running zero shot eval")
            # Control video capture based on number of evaluation episodes
            if config.get("EVAL_EPISODES") <= 40:
                capture_video = True
            else:
                capture_video = False
                
            eval_data = run_eval_jax(
                cfg=config,
                agent_state=runner_state[0].params,
                writer=wb_run,
                acting_fns=_select_action_eval,
                eval_seed=42,
                global_step=0,
                capture_video=capture_video,
                recurrent=False,
                shared_weights=True,
                parallel=config.get("EVAL_PARALLEL", True),
            )

        # Initialize training metrics
        training_metrics = None

        # Calculate team diversity metrics if requested
        calculate_team_diversity_metrics = config.get(
            "CALCULATE_TEAM_DIVERSITY_METRICS", False
        )
        if calculate_team_diversity_metrics:
            _keys = jax.random.PRNGKey(config["SEED"])
            calculate_team_diversity(
                network, runner_state[0], _keys, num_agents, 
                centralised_critic=True, param_sharing=True
            )

        # Initialize evaluation metrics and training variables
        eval_metrics = []
        start_time = time.time()
        global_step = 0
        next_eval_step = config["EVAL_INTERVAL"]
        next_capture_video_step = config.get("CAPTURE_VIDEO_INTERVAL", None)
        next_checkpoint_step = config.get("CHECKPOINT_INTERVAL", None)
        
        # Early return if only evaluating
        eval_only = config.get("EVAL_ONLY", False)
        if eval_only:
            return {"metrics": {}, "eval_metrics": [(0, eval_data)]}, env

        # Main training loop
        for update in range(config["NUM_UPDATES"]):
            final_update = update == config["NUM_UPDATES"] - 1
            update_time_start = time.time()
            
            # Perform update
            runner_state, ret_metric = _update_step(runner_state, None)
            global_step += config["NUM_STEPS"] * config["NUM_ENVS"]

            # Log training speed periodically
            if update % 100 == 0:
                print(f"Update: {update}/{config['NUM_UPDATES']}")
                sps = int(global_step / (time.time() - start_time))
                sps_update = int(
                    config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                    / (time.time() - update_time_start)
                )
                print("SPS:", sps, sps_update)
                wb_run.log({"charts/SPS": sps}, global_step)
                wb_run.log({"charts/SPS_update": sps_update}, global_step)

            # Initialize or update metrics
            if training_metrics is None:
                training_metrics = initialize_metrics_storage(config, ret_metric.keys())
            training_metrics = update_metrics(training_metrics, ret_metric, update)

            # Run evaluation periodically
            record_final_episode = config.get("CAPTURE_VIDEO_INTERVAL") and final_update
            if (global_step >= next_eval_step) or record_final_episode:
                if (next_capture_video_step and global_step >= next_capture_video_step) or record_final_episode:
                    next_capture_video_step += config["CAPTURE_VIDEO_INTERVAL"]
                    capture_video = True
                else:
                    capture_video = False
                    
                # Run evaluation
                eval_data = run_eval_jax(
                    cfg=config,
                    agent_state=runner_state[0].params,
                    writer=wb_run,
                    acting_fns=_select_action_eval,
                    eval_seed=42,
                    global_step=global_step,
                    capture_video=capture_video,
                    recurrent=False,
                    shared_weights=True,
                    parallel=config.get("EVAL_PARALLEL", True),
                )
                eval_metrics.append((global_step, eval_data))
                next_eval_step += config["EVAL_INTERVAL"]

            # Save checkpoints periodically
            if (final_update) or (next_checkpoint_step and global_step >= next_checkpoint_step):
                agent_identity = f"agent_{config['SEED']}_seed"
                model_path = f"{config['CHP_DIR']}/{config['EXP_NAME']}_{global_step}_steps_{update}_updates.{agent_identity}"
                save_args = orbax_utils.save_args_from_target(runner_state[0].params)
                checkpointers.save(
                    model_path, runner_state[0].params, save_args=save_args
                )
                print(f"Model saved to {model_path} at step {global_step}")
                next_checkpoint_step += config["CHECKPOINT_INTERVAL"]

        return {
            "runner_state": runner_state,
            "metrics": training_metrics,
            "eval_metrics": eval_metrics,
        }, env

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_ff_shared_weights_vmas_dispersion",
)
def main(config):
    """
    Main entry point for training.
    
    Args:
        config: Hydra configuration
    """
    print("Starting training")
    config = OmegaConf.to_container(config, resolve=True)

    # Initialize wandb
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["EXP_TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=config["RUN_NAME"],
        save_code=True,
        reinit=True,
        group=config["GROUP"],
    )

    # Set up checkpoint directory
    config["CHP_DIR"] = f"{wandb.run.dir}/models/{config['RUN_NAME']}"  # type: ignore

    # Load checkpoint if specified
    if config.get("CHECKPOINT_NAME") is not None:
        artifact = run.use_artifact(config.get("CHECKPOINT_NAME"), type="checkpoint")
        artifact_dir = artifact.download()
        checkpoint_load_dir = f"{artifact_dir}/{config.get('CHECKPOINT_FOLDER')}"

        config["CHECKPOINT_LOAD_DIR"] = checkpoint_load_dir
        print(f"checkpoint_load_dir: {checkpoint_load_dir}")

    # Set up RNG
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Add environment kwargs to config
    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])

    # Run training
    train = make_train(config)
    out, env = train(rngs[0], run)

    # Log metrics
    log_train_metrics(config, out["metrics"], run)
    log_eval_metrics(config, out["eval_metrics"], run)
    
    # Clean up
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()