"""
MAPPO with Feedforward Networks for Non-Parameter Sharing (NPS) + Non-Jax Cpu Envs.
"""

import multiprocessing as mp
from collections import defaultdict
from typing import Any, Dict, NamedTuple, List
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.struct import dataclass
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import distrax
import wandb
from omegaconf import OmegaConf
import hydra
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

from baselines.MAPPO.mappo import get_update_fn
from baselines.utils.eval import run_eval_jax
from baselines.utils.wrappers import make_env
from baselines.utils.utils import (
    calculate_team_diversity,
    log_eval_metrics,
    log_train_metrics,
)

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
    Actor-Critic network with centralized critic for MAPPO.
    
    The actor makes decisions based on local observations, while the critic
    evaluates based on the global observation (observations of all agents).
    
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
            critic: Value function estimate
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
def update_metrics(
    metrics: dict, new_values: dict, update_idx: int
) -> dict:
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
    config["NUM_ACTORS"] = config["NUM_ENVS"]  # Each env counts as one actor now
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

    def make_update_all_agents_fn(networks: dict, num_agents: int, update_fns: dict):
        """
        Create a function to update all agents' networks.
        
        Args:
            networks: Dictionary of agent networks
            num_agents: Number of agents
            update_fns: Dictionary of update functions for each agent
            
        Returns:
            Function that updates all agents
        """
        @jax.jit
        def update_all_agents(
            traj_batches, train_state, last_obs_batch, last_global_obs_batch, rng, episode_stats
        ):
            """
            Update all agents' policies with their respective batches.
            
            Args:
                traj_batches: Dictionary of trajectory batches per agent
                train_state: Current training state
                last_obs_batch: Final observations
                last_global_obs_batch: Final global observations
                rng: Random number generator key
                episode_stats: Current episode statistics
                
            Returns:
                Updated train state, metrics, and RNG
            """
            metrics = {}
            for agent in range(num_agents):
                traj_batch = traj_batches[agent]
                update_state, loss_info = update_fns[agent](
                    traj_batch,
                    last_obs_batch[:, agent, :],
                    last_global_obs_batch[:, agent, :],
                    train_state[agent],
                    rng,
                )

                train_state[agent] = update_state[0]
                metric = {
                    "returned_episode_returns": traj_batch.info.returned_episode_returns,
                    "returned_episode_lengths": traj_batch.info.returned_episode_lengths,
                }
                rng = update_state[-1]

                loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
                metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
                metric = {**metric, **loss_info}
                metrics[agent] = metric
            return train_state, metrics, rng

        return update_all_agents

    def train(rng, wb_run=None):
        """
        Main training function.
        
        Args:
            rng: JAX random number generator key
            wb_run: Weights & Biases run object
            
        Returns:
            Dictionary of training results
        """
        # Initialize networks and states for all agents
        num_agents = len(env.agents)
        networks = {}
        train_states = {}
        update_fns = {}
        zero_shot_eval = False
        param_count = 0
        
        for agent in range(len(possible_agents)):
            # Create network
            network_x = ActorCritic(
                action_dim,
                activation=config["ACTIVATION"],
                actor_layers=config.get("ACTOR_LAYERS"),
                critic_layers=config.get("CRITIC_LAYERS"),
                num_agents=env.num_agents,
            )
            # Initialize network parameters
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(observation_size)
            init_x_critic = jnp.zeros((observation_size * num_agents))
            network_params = network_x.init(_rng, init_x, init_x_critic)
            param_count += sum(x.size for x in jax.tree_util.tree_leaves(network_params))
            
            # Set up optimizer
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

            # Create training state
            train_state_x = TrainState.create(
                apply_fn=network_x.apply,
                params=network_params,
                tx=tx,
            )

            # Load checkpoint if specified
            if config.get("CHECKPOINT_LOAD_DIR") is not None:
                full_checkpoint_path = f"{config.get('CHECKPOINT_LOAD_DIR')}.agent_{agent}_{config.get('SEED')}_seed"
                print(f"Loading from {full_checkpoint_path}")
                load_checkpointer = checkpointer.Checkpointer(
                    PyTreeCheckpointHandler(aggregate_filename="checkpoints")
                )
                loaded_checkpoints = load_checkpointer.restore(
                    full_checkpoint_path, item=train_state_x.params
                )
                train_state_x = TrainState.create(
                    apply_fn=train_state_x.apply_fn, params=loaded_checkpoints, tx=tx
                )
                zero_shot_eval = True

            networks[agent] = network_x
            train_states[agent] = train_state_x
            update_fns[agent] = get_update_fn(config, network_x)

        # Log parameter count
        wb_run.log({"num_params": param_count}, commit=False)

        # Initialize episode statistics
        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros((config["NUM_ENVS"] * num_agents), dtype=jnp.float32),
            episode_lengths=jnp.zeros((config["NUM_ENVS"] * num_agents), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((config["NUM_ENVS"] * num_agents), dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros((config["NUM_ENVS"] * num_agents), dtype=jnp.int32),
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
            global_obs = jnp.zeros((config["NUM_ENVS"] * num_agents, num_agents * obs_dim))
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
        int_seed = jax.random.randint(reset_rng, shape=(1,), minval=1, maxval=1000000).item()
        obsv, infos = env.reset(seed=int_seed)
        global_obsv = concat_local_obs_into_global_obs(obsv)
        env_state = {}
        env_step = env.step

        # Reshape observations for agent-wise processing
        obsv = obsv.reshape((config["NUM_ENVS"], num_agents, -1))
        global_obsv = global_obsv.reshape(((config["NUM_ENVS"], num_agents, -1)))

        def step_env_wrapped(episode_stats: EpisodeStatistics, action: Any) -> Any:
            """
            Step the environment and update episode statistics.
            
            Args:
                episode_stats: Current episode statistics
                action: Actions to take
                
            Returns:
                Updated episode statistics and transition information
            """
            next_obs, reward, termination, truncs, info = env_step(action.flatten())
            # Update episode statistics
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = jnp.add(episode_stats.episode_lengths, 1)
            returned_episode_returns = jnp.where(
                termination + truncs,
                new_episode_return,
                episode_stats.returned_episode_returns,
            )
            returned_episode_lengths = jnp.where(
                termination + truncs,
                new_episode_length,
                episode_stats.returned_episode_lengths,
            )

            # Reset episode stats for completed episodes
            episode_stats = episode_stats.replace(
                episode_returns=(new_episode_return) * (1 - termination) * (1 - truncs),
                episode_lengths=(new_episode_length) * (1 - termination) * (1 - truncs),
                returned_episode_returns=returned_episode_returns,
                returned_episode_lengths=returned_episode_lengths,
            )

            # Return transition info
            return_infos = TransitionInfo(
                returned_episode_returns=returned_episode_returns.reshape(
                    (config["NUM_ENVS"], num_agents, -1)
                ),
                returned_episode_lengths=returned_episode_lengths.reshape(
                    (config["NUM_ENVS"], num_agents, -1)
                ),
            )
            global_obs = concat_local_obs_into_global_obs(next_obs)
            return episode_stats, (next_obs, reward, termination, truncs, return_infos, global_obs)

        @jax.jit
        def _select_action(params, obs, obs_critic, _rng):
            """
            Select actions for all agents based on current policy.
            
            Args:
                params: Network parameters
                obs: Local observations
                obs_critic: Global observations
                _rng: Random number generator key
                
            Returns:
                actions: Selected actions
                log_probs: Log probabilities of actions
                values: Value function estimates
            """
            actions = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
            log_probs = jnp.zeros((config["NUM_ENVS"], num_agents))
            values = jnp.zeros((config["NUM_ENVS"], num_agents))
            
            for agent in range(num_agents):
                # Get action logits and value from network
                actor_logits, value = networks[agent].apply(
                    params[agent].params, obs[:, agent, :], obs_critic[:, agent, :]
                )
                # Sample action from categorical distribution
                pi = distrax.Categorical(logits=actor_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                _, _rng = jax.random.split(_rng)

                # Store actions, log probs, and values
                actions = actions.at[:, agent].set(action)
                log_probs = log_probs.at[:, agent].set(log_prob)
                values = values.at[:, agent].set(value)

            return actions, log_probs, values

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            """
            Select deterministic actions for evaluation.
            
            Args:
                params: Network parameters
                obs: Observations
                _rng: Random number generator key
                
            Returns:
                actions: Selected actions (modes of distributions)
            """
            obs_dim = obs.shape[-1]
            # [num_envs, num_agents, obs_dim]
            obs = obs.reshape((-1, num_agents, obs_dim))
            num_eval_envs = obs.shape[0]
            actions = jnp.zeros((num_eval_envs, num_agents), dtype=jnp.int32)
            
            # Create dummy global observations
            dummy_obs_critic = jnp.zeros((num_eval_envs, num_agents, num_agents * obs_dim))
            for agent in range(num_agents):
                actor_logits, _ = networks[agent].apply(
                    params[agent].params, obs[:, agent, :], dummy_obs_critic[:, agent, :]
                )
                pi = distrax.Categorical(logits=actor_logits)
                # Use mode for deterministic evaluation
                action = pi.mode()
                actions = actions.at[:, agent].set(action)

            return (actions,)

        # Create function to update all agents
        update_all_agents_fn = make_update_all_agents_fn(networks, num_agents, update_fns)
        
        @jax.jit
        def save_transition_to_single_transition_array(transitions: list) -> Transition:
            """
            Organize transitions into agent-wise batches.
            
            Args:
                transitions: List of transitions
                
            Returns:
                Dictionary mapping agents to their transition batches
            """
            traj_batches = {}
            for agent in range(num_agents):
                done = jnp.array([t.done[:, agent] for t in transitions])
                action = jnp.array([t.action[:, agent] for t in transitions])
                value = jnp.array([t.value[:, agent] for t in transitions])
                reward = jnp.array([t.reward[:, agent] for t in transitions])
                log_prob = jnp.array([t.log_prob[:, agent] for t in transitions])
                obs = jnp.array([t.obs[:, agent, :] for t in transitions])
                global_obs = jnp.array([t.global_obs[:, agent, :] for t in transitions])
                returned_episode_returns = jnp.array(
                    [t.info.returned_episode_returns[:, agent] for t in transitions]
                )
                returned_episode_lengths = jnp.array(
                    [t.info.returned_episode_lengths[:, agent] for t in transitions]
                )
                info = TransitionInfo(returned_episode_returns, returned_episode_lengths)
                traj_batches[agent] = Transition(
                    done, action, value, reward, log_prob, obs, global_obs, info
                )

            return traj_batches

        def _update_step(runner_state, unused):
            """
            Perform a single training update step.
            
            Args:
                runner_state: Current runner state
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated runner state and metrics
            """
            def _env_step(runner_state):
                """
                Perform a single environment step and collect transition.
                
                Args:
                    runner_state: Current runner state
                    
                Returns:
                    Updated runner state and transition
                """
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_global_obs,
                    rng,
                    episode_stats,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # Select actions
                action, log_prob, value = _select_action(
                    train_state, last_obs, last_global_obs, _rng
                )

                # Step environment
                episode_stats, (obsv, reward, termination, truncs, info, global_obs) = (
                    step_env_wrapped(episode_stats, action)
                )

                # Process observations
                obsv = obsv.reshape((config["NUM_ENVS"], num_agents, -1))
                global_obs = global_obs.reshape((config["NUM_ENVS"], num_agents, -1))
                done = jnp.logical_or(termination, truncs).reshape(
                    (config["NUM_ENVS"], num_agents)
                )
                reward = reward.reshape((config["NUM_ENVS"], num_agents))
                
                # Store transition
                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    log_prob,
                    last_obs,
                    last_global_obs,
                    info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    global_obs,
                    rng,
                    episode_stats,
                )
                return runner_state, transition

            # Collect transitions over multiple steps
            transitions = []
            for _ in range(config["NUM_STEPS"]):
                runner_state, transition = _env_step(runner_state)
                transitions.append(transition)

            # Organize transitions by agent
            traj_batch = save_transition_to_single_transition_array(transitions)

            # Update networks
            train_state, env_state, last_obs, last_global_obs, rng, episode_stats = runner_state
            train_state, metric, rng = update_all_agents_fn(
                traj_batch, train_state, last_obs, last_global_obs, rng, episode_stats
            )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_global_obs,
                rng,
                episode_stats,
            )
            return runner_state, metric

        # Initialize runner state
        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_state, obsv, global_obsv, _rng, episode_stats)

        # Set up checkpointers for each agent
        checkpointers = {}
        for agent in range(num_agents):
            agent_key = f"agent_{agent}"
            checkpointers[agent_key] = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
            )

        # Perform zero-shot evaluation if loading from checkpoint
        if zero_shot_eval:
            print("running zero shot eval")
            eval_data = run_eval_jax(
                cfg=config,
                agent_state=runner_state[0],
                writer=wb_run,
                acting_fns=_select_action_eval,
                eval_seed=42,
                global_step=0,
                capture_video=True,
                recurrent=False,
                shared_weights=True,
                parallel=config.get("EVAL_PARALLEL", True),
            )

        # Initialize training metrics
        training_metrics = {agent: None for agent in range(num_agents)}

        # Calculate team diversity metrics if requested
        calculate_team_diversity_metrics = config.get("CALCULATE_TEAM_DIVERSITY_METRICS", False)
        if calculate_team_diversity_metrics:
            _keys = jax.random.PRNGKey(config['SEED'])
            calculate_team_diversity(networks, runner_state[0], _keys, num_agents, centralised_critic=True, param_sharing=False)

        # Initialize training variables
        start_time = time.time()
        global_step = 0
        next_eval_step = config["EVAL_INTERVAL"]
        next_capture_video_step = config.get("CAPTURE_VIDEO_INTERVAL", None)
        next_checkpoint_step = config.get("CHECKPOINT_INTERVAL", None)

        # Early return if only evaluating
        eval_only = config.get("EVAL_ONLY", False)
        if eval_only:
            return {"metrics": {}, "eval_metrics": [(0, eval_data)]}, env

        # Initialize evaluation metrics
        eval_metrics = []

        # Main training loop
        for update in range(config["NUM_UPDATES"]):
            final_update = update == config["NUM_UPDATES"] - 1
            update_time_start = time.time()
            
            # Perform update
            runner_state, ret_metric = _update_step(runner_state, None)
            global_step += 1 * config["NUM_STEPS"] * config["NUM_ENVS"]

            # Log training speed periodically
            if update % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                sps_update = int(
                    config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                    / (time.time() - update_time_start)
                )
                print("SPS:", sps, sps_update)
                wb_run.log({"charts/SPS": sps}, global_step)
                wb_run.log({"charts/SPS_update": sps_update}, global_step)

            # Update training metrics
            for agent in range(num_agents):
                if training_metrics[agent] is None:
                    training_metrics[agent] = initialize_metrics_storage(config, ret_metric[agent].keys())
                else:
                    training_metrics[agent] = update_metrics(training_metrics[agent], ret_metric[agent], update)

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
                    agent_state=runner_state[0],
                    writer=wb_run,
                    acting_fns=_select_action_eval,
                    eval_seed=42,
                    global_step=global_step,
                    capture_video=capture_video,
                    recurrent=False,
                    shared_weights=True,
                    parallel=config.get("EVAL_PARALLEL", True),
                )
                next_eval_step += config["EVAL_INTERVAL"]
                eval_metrics.append((global_step, eval_data))

            # Save checkpoints periodically
            if final_update or (next_checkpoint_step and global_step >= next_checkpoint_step):
                for agent in range(num_agents):
                    agent_key = f"agent_{agent}"
                    agent_identity = f"{agent_key}_{config['SEED']}_seed"
                    model_path = f"{config['CHP_DIR']}/{config['EXP_NAME']}_{global_step}_steps_{update}_updates.{agent_identity}"
                    save_args = orbax_utils.save_args_from_target(runner_state[0][agent].params)
                    checkpointers[agent_key].save(model_path, runner_state[0][agent].params, save_args=save_args)
                    print(f"model saved to {model_path} at step {global_step}")
                    next_checkpoint_step += config["CHECKPOINT_INTERVAL"]

        return {"runner_state": runner_state, "metrics": training_metrics, "eval_metrics": eval_metrics}, env

    return train


@hydra.main(version_base=None, config_path="config", config_name="mappo_ff_nps_vmas_dispersion")
def main(config):
    """
    Main entry point for training.
    
    Args:
        config: Hydra configuration
    """
    print("starting training")
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
    config["CHP_DIR"] = f"{wandb.run.dir}/models/{config['RUN_NAME']}"

    # Load checkpoint if specified
    if config.get("CHECKPOINT_NAME") is not None:
        artifact = run.use_artifact(config.get("CHECKPOINT_NAME"), type="checkpoint")
        artifact_dir = artifact.download()
        checkpoint_load_dir = f"{artifact_dir}/{config.get('CHECKPOINT_FOLDER')}"
        config["CHECKPOINT_LOAD_DIR"] = checkpoint_load_dir
        print(f"checkpoint_load_dir: {checkpoint_load_dir}")

    # Set up RNG and environment kwargs
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Add environment kwargs to config
    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])

    # Run training
    train = make_train(config)
    out, env = train(rngs[0], run)

    # Log metrics
    for agent in out["metrics"].keys():
        log_train_metrics(config, out["metrics"][agent], run, agent_id=agent)

    log_eval_metrics(config, out["eval_metrics"], run)

    # Upload checkpoints to wandb
    eval_only = config.get("EVAL_ONLY", False)
    if config["WANDB_MODE"] == "online" and not eval_only:
        print("pushing checkpoints")
        artifact = wandb.Artifact(name=f'checkpoint_{config["RUN_NAME"]}', type="checkpoint")
        artifact.add_dir(local_path=config["CHP_DIR"])
        run.log_artifact(artifact)

    # Clean up
    print("closing env")
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()