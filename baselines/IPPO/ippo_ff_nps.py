"""
IPPO with Feedforward Networks for Non-Parameter Sharing (NPS) + Non-Jax Cpu Envs
"""

# Environment and multiprocessing setup
import multiprocessing as mp
from collections import defaultdict
import gym

# Ensure proper multiprocessing method
forkserver_available = "forkserver" in mp.get_all_start_methods()
start_method = "forkserver" if forkserver_available else "spawn"
mp.set_start_method(start_method, force=True)


# Standard libraries
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from enum import Enum

# JAX and related libraries
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.struct import dataclass
from flax.training import orbax_utils
from flax.training.train_state import TrainState

# RL and optimization
import distrax
import optax

# Utilities and visualization
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb

# Checkpoint management
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

# Project imports
from baselines.utils.eval import run_eval_jax
from baselines.IPPO.ppo import get_update_fn_ff_ppo as get_update_fn
from baselines.utils.utils import (
    calculate_team_diversity,
    create_action_functions,
    log_eval_metrics,
    log_train_metrics,
)
from baselines.utils.wrappers import make_env


@dataclass
class EpisodeStatistics:
    """Tracks statistics for episodes across multiple environments."""
    episode_returns: jnp.array       # Current episode returns
    episode_lengths: jnp.array       # Current episode lengths
    returned_episode_returns: jnp.array  # Returns of completed episodes
    returned_episode_lengths: jnp.array  # Lengths of completed episodes


class ActionType(Enum):
    """Supported action space types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ActorCritic(nn.Module):
    """
    Actor-Critic network architecture for IPPO agents.
    
    Each agent uses its own instance of this network with independent parameters.
    """
    action_type: ActionType         # Type of action space (discrete/continuous)
    action_dim: int                 # Dimensionality of action space
    num_agents: int                 # Number of agents in environment
    activation: str = "tanh"        # Activation function ("tanh" or "relu")
    actor_layers: List[int] = (64, 64)  # Hidden layer sizes for actor network
    critic_layers: List[int] = (64, 64)  # Hidden layer sizes for critic network

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the actor-critic network.
        
        Args:
            x: Input observation tensor
            
        Returns:
            actor_output: Action distribution parameters 
            critic_value: Value function estimate
        """
        # Select activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Helper function to build MLP with specified layer sizes
        def build_network(x, layer_sizes: List[int], activation):
            for size in layer_sizes:
                x = nn.Dense(
                    size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
                x = activation(x)
            return x

        # Actor network - policy function
        hidden = build_network(x, self.actor_layers, activation)
        if self.action_type is ActionType.DISCRETE:
            # Output logits for categorical distribution
            actor_output = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(hidden)
        else:  # continuous actions
            # Output mean and log standard deviation for Gaussian policy
            actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(hidden)
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            actor_output = (actor_mean, actor_logtstd)

        # Critic network - value function
        critic = build_network(x, self.critic_layers, activation)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return actor_output, jnp.squeeze(critic, axis=-1)


class TransitionInfo(NamedTuple):
    """Additional information stored with each transition."""
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@dataclass
class Transition:
    """Stores a single step transition for training."""
    done: jnp.ndarray        # Terminal state flags
    action: jnp.ndarray      # Actions taken
    value: jnp.ndarray       # Value estimates
    reward: jnp.ndarray      # Rewards received
    log_prob: jnp.ndarray    # Log probabilities of actions
    obs: jnp.ndarray         # Observations
    info: TransitionInfo     # Episode statistics


# Metrics tracking functions
def initialize_metrics_storage(config: Dict, metric_keys: List[str]) -> Dict:
    """
    Initialize metrics storage with zeros arrays for each key.
    
    Args:
        config: Configuration dictionary with NUM_UPDATES defined
        metric_keys: List of metric names to track
        
    Returns:
        Dict mapping metrics names to zero-filled arrays
    """
    return {k: np.zeros(config["NUM_UPDATES"]) for k in metric_keys}


@jax.jit
def update_metrics(metrics: Dict, new_values: Dict, update_idx: int) -> Dict:
    """
    Update metrics arrays with new values at specified index.
    
    Args:
        metrics: Current metrics dictionary
        new_values: New values to insert
        update_idx: Index at which to insert new values
        
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
    Construct the training function based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train: Function that performs training
    """
    # Verify configuration is valid for NPS
    assert (
        config["env"]["ENV_KWARGS"].get("one_hot_encode_agent_id") is not True
    ), "one_hot_ids should be false for NPS"
    
    # Create vectorized environment
    env, possible_agents, action_dim, num_actions, observation_size = make_env(
        config["ENV_NAME"], num_envs=config["NUM_ENVS"], **config["TRAIN_ENV_KWARGS"]
    )
    
    # Calculate derived parameters
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # Update wandb config with calculated values
    wandb.config.update(
        {
            "MINIBATCH_SIZE": config["MINIBATCH_SIZE"],
            "NUM_UPDATES": config["NUM_UPDATES"],
        },
        allow_val_change=True,
    )

    # Parse action space type
    action_space_type = config["ACTION_SPACE_TYPE"].lower()
    if action_space_type == ActionType.DISCRETE.value:
        action_type = ActionType.DISCRETE
    elif action_space_type == ActionType.CONTINUOUS.value:
        action_type = ActionType.CONTINUOUS
    else:
        raise ValueError(f"Unknown action space type: {config['ACTION_SPACE_TYPE']}")

    # Learning rate scheduler
    def linear_schedule(count):
        """Linear learning rate decay schedule."""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Function to update all agent networks
    def make_update_all_agents_fn(networks: dict, num_agents: int, update_fns: dict):
        """
        Create a function that updates all agent networks.
        
        Args:
            networks: Dictionary of agent networks
            num_agents: Number of agents
            update_fns: Dictionary of update functions per agent
            
        Returns:
            Function that updates all agents' networks
        """
        @jax.jit
        def update_all_agents(
            traj_batches, train_state, last_obs_batch, rng, episode_stats
        ):
            metrics = {}
            for agent in range(num_agents):
                traj_batch = traj_batches[agent]
                update_state, loss_info = update_fns[agent](
                    traj_batch, last_obs_batch[:, agent, :], train_state[agent], rng
                )

                train_state[agent] = update_state[0]

                # Collect episode statistics and metrics
                metric = {
                    "returned_episode_returns": traj_batch.info.returned_episode_returns,
                    "returned_episode_lengths": traj_batch.info.returned_episode_lengths,
                }
                rng = update_state[-1]

                # Compute mean metrics for logging
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
            wb_run: Weights & Biases run object for logging
            
        Returns:
            Dictionary of results and trained environment
        """
        # Initialize networks, train states, and update functions for each agent
        networks = {}
        train_states = {}
        update_fns = {}
        zero_shot_eval = False
        param_count = 0
        
        for agent in range(len(possible_agents)):
            # Initialize network architecture
            network_x = ActorCritic(
                action_dim=action_dim,
                activation=config["ACTIVATION"],
                actor_layers=config.get("ACTOR_LAYERS"),
                critic_layers=config.get("CRITIC_LAYERS"),
                num_agents=env.num_agents,
                action_type=action_type,
            )
            
            # Initialize network parameters
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(observation_size)
            network_params = network_x.init(_rng, init_x)
            param_count += sum(x.size for x in jax.tree_util.tree_leaves(network_params))
            
            # Configure optimizer with optional learning rate annealing and gradient clipping
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    (
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"])
                        if config["MAX_GRAD_NORM"] is not None
                        else optax.identity()
                    ),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    (
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"])
                        if config["MAX_GRAD_NORM"] is not None
                        else optax.identity()
                    ),
                    optax.adam(config["LR"], eps=1e-5),
                )

            # JIT compile network for efficiency
            network_x.apply = jax.jit(network_x.apply)
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

                # Load parameters from checkpoint
                loaded_checkpoints = load_checkpointer.restore(
                    full_checkpoint_path, item=train_state_x.params
                )
                train_state_x = TrainState.create(
                    apply_fn=train_state_x.apply_fn, params=loaded_checkpoints, tx=tx
                )

                # Enable zero-shot evaluation when loading checkpoints
                zero_shot_eval = True

            networks[agent] = network_x
            train_states[agent] = train_state_x
            update_fns[agent] = get_update_fn(config, network_x)

        
        wb_run.log({"num_params": param_count}, commit=False)
        
        # Initialize episode statistics
        num_agents = len(env.agents)
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

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        int_seed = jax.random.randint(
            reset_rng, shape=(1,), minval=1, maxval=1000000
        ).item()
        obsv, infos = env.reset(seed=int_seed)

        # Reshape observations to [num_envs, num_agents, obs_dim]
        obsv = obsv.reshape((config["NUM_ENVS"], num_agents, -1))
        env_state = {}
        env_step = env.step

        def step_env_wrapped(episode_stats: EpisodeStatistics, action: Any) -> Any:
            """Take a step in the environment with the given action."""
            next_obs, reward, termination, truncs, info = env_step(action.flatten())
            return next_obs, reward, termination, truncs, info

        @jax.jit
        def update_episode_stats(episode_stats, reward, termination, truncs, obsv):
            """Update episode statistics based on environment step results."""
            # Determine if episode is done (either terminated or truncated)
            done = jnp.logical_or(termination, truncs)
            
            # Update running totals for current episodes
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            
            # Store returns and lengths of completed episodes
            returned_episode_returns = jnp.where(
                done, new_episode_return, episode_stats.returned_episode_returns
            )
            returned_episode_lengths = jnp.where(
                done, new_episode_length, episode_stats.returned_episode_lengths
            )

            # Update episode statistics, resetting counters for done episodes
            episode_stats = episode_stats.replace(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=returned_episode_returns,
                returned_episode_lengths=returned_episode_lengths,
            )

            # Reshape return info to match expected format
            return_infos = TransitionInfo(
                returned_episode_returns=returned_episode_returns.reshape(
                    (config["NUM_ENVS"], num_agents, -1)
                ),
                returned_episode_lengths=returned_episode_lengths.reshape(
                    (config["NUM_ENVS"], num_agents, -1)
                ),
            )

            # Reshape observations and metrics for multi-agent format
            obsv = obsv.reshape((config["NUM_ENVS"], num_agents, -1))
            done = done.reshape((config["NUM_ENVS"], num_agents))
            reward = reward.reshape((config["NUM_ENVS"], num_agents))
            return episode_stats, return_infos, done, obsv, reward

        @jax.jit
        def _select_action(params, obs, _rng):
            """
            Select actions for all agents using their respective policies.
            
            Args:
                params: Dictionary of agent policy parameters
                obs: Current observations
                _rng: Random number generator key
                
            Returns:
                actions: Selected actions for all agents
                log_probs: Log probabilities of selected actions
                values: Value function estimates
            """
            # Initialize output arrays
            actions = (
                jnp.zeros((config["NUM_ENVS"], num_agents, action_dim))
                if action_type is ActionType.CONTINUOUS
                else jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
            )
            log_probs = jnp.zeros((config["NUM_ENVS"], num_agents))
            values = jnp.zeros((config["NUM_ENVS"], num_agents))
            
            # Select action for each agent
            for agent in range(num_agents):
                # Get policy and value outputs from agent's network
                actor_output, value = networks[agent].apply(
                    params[agent].params, obs[:, agent, :]
                )
                
                # Create action distribution based on action type
                if action_type is ActionType.DISCRETE:
                    pi = distrax.Categorical(logits=actor_output)
                else:  # continuous
                    mean, log_std = actor_output
                    pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

                # Sample action and get log probability
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Split RNG key for next iteration
                _, _rng = jax.random.split(_rng)

                # Store action, log probability, and value
                if action_type is ActionType.CONTINUOUS:
                    actions = actions.at[:, agent, :].set(action)
                else:
                    actions = actions.at[:, agent].set(action)

                log_probs = log_probs.at[:, agent].set(log_prob)
                values = values.at[:, agent].set(value)

            return actions, log_probs, values

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            """
            Select actions for evaluation (with optional deterministic policy).
            
            Args:
                params: Dictionary of agent policy parameters
                obs: Current observations
                _rng: Random number generator key
                
            Returns:
                actions: Selected actions for all agents
            """
            obs_dim = obs.shape[-1]
            # Reshape to [num_envs, num_agents, obs_dim]
            obs = obs.reshape((-1, num_agents, obs_dim))
            num_eval_envs = obs.shape[0]
            
            # Initialize actions array
            actions = (
                jnp.zeros((num_eval_envs, num_agents, num_actions))
                if action_type is ActionType.CONTINUOUS
                else jnp.zeros((num_eval_envs, num_agents), dtype=jnp.int32)
            )
            
            # Select action for each agent
            for agent in range(num_agents):
                # Get policy output from agent's network
                actor_output, _ = networks[agent].apply(
                    params[agent].params, obs[:, agent, :]
                )
                
                # Create action distribution
                if action_type is ActionType.DISCRETE:
                    pi = distrax.Categorical(logits=actor_output)
                else:  # continuous
                    mean, log_std = actor_output
                    pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

                # Use deterministic policy if configured, otherwise sample
                if config.get("EVAL_DETERMINISTIC") is False:
                    action = pi.sample(seed=_rng)
                else:
                    action = pi.mode()

                # Store action
                if action_type is ActionType.CONTINUOUS:
                    actions = actions.at[:, agent, :].set(action)
                else:
                    actions = actions.at[:, agent].set(action)
            return (actions,)

        # Create function to update all agents
        update_all_agents_fn = make_update_all_agents_fn(
            networks, num_agents, update_fns
        )

        @jax.jit
        def save_transition_to_single_transition_array(transitions: list) -> Transition:
            """
            Convert list of transitions to dictionary of agent-specific trajectories.
            
            Args:
                transitions: List of Transition objects
                
            Returns:
                Dictionary mapping agent IDs to their trajectory data
            """
            traj_batches = {}
            for agent in range(num_agents):
                # Extract per-agent data from transitions
                done = jnp.array([t.done[:, agent] for t in transitions])
                action = jnp.array([t.action[:, agent] for t in transitions])
                value = jnp.array([t.value[:, agent] for t in transitions])
                reward = jnp.array([t.reward[:, agent] for t in transitions])
                log_prob = jnp.array([t.log_prob[:, agent] for t in transitions])
                obs = jnp.array([t.obs[:, agent, :] for t in transitions])
                returned_episode_returns = jnp.array(
                    [t.info.returned_episode_returns[:, agent] for t in transitions]
                )
                returned_episode_lengths = jnp.array(
                    [t.info.returned_episode_lengths[:, agent] for t in transitions]
                )
                
                # Construct batch for this agent
                info = TransitionInfo(
                    returned_episode_returns, returned_episode_lengths
                )
                traj_batches[agent] = Transition(
                    done, action, value, reward, log_prob, obs, info
                )

            return traj_batches

        def _update_step(runner_state, unused):
            """
            Perform a single training update step.
            
            Args:
                runner_state: Current training state
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated runner state and metrics
            """
            train_state, env_state, last_obs, rng, episode_stats = runner_state
            transitions = []

            # Collect trajectory data by stepping through environment
            for _ in range(config["NUM_STEPS"]):
                # Select actions using current policies
                action, log_prob, value = _select_action(train_state, last_obs, rng)

                # Step the environment
                obsv, reward, termination, truncs, info = step_env_wrapped(env, action)

                # Update statistics
                episode_stats, info, done, obsv, reward = update_episode_stats(
                    episode_stats, reward, termination, truncs, obsv
                )

                # Store transition data
                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    log_prob,
                    last_obs,
                    info,
                )
                transitions.append(transition)

                # Update observation for next step
                last_obs = obsv

            # Prepare trajectories for update
            traj_batches = save_transition_to_single_transition_array(transitions)

            # Update all agent policies
            train_state, metrics, rng = update_all_agents_fn(
                traj_batches, train_state, last_obs, rng, episode_stats
            )

            runner_state = (train_state, env_state, last_obs, rng, episode_stats)
            return runner_state, metrics

        # Initialize runner state
        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_state, obsv, _rng, episode_stats)

        # Initialize checkpointers for each agent
        checkpointers = {}
        for agent in range(num_agents):
            agent_key = f"agent_{agent}"
            checkpointers[agent_key] = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
            )

        # Perform zero-shot evaluation if loading from checkpoint
        if zero_shot_eval:
            # Determine whether to capture video based on eval episodes
            capture_video = config.get("EVAL_EPISODES") <= 40
            
            # Run evaluation
            eval_data = run_eval_jax(
                cfg=config,
                agent_state=runner_state[0],
                writer=wb_run,
                acting_fns=_select_action_eval,
                eval_seed=42,
                global_step=0,
                capture_video=capture_video,
                recurrent=False,
                shared_weights=False,
                parallel=config.get("EVAL_PARALLEL", True),
            )

        # Calculate team diversity metrics if configured
        calculate_team_diversity_metrics = config.get(
            "CALCULATE_TEAM_DIVERSITY_METRICS", False
        )
        if calculate_team_diversity_metrics:
            _keys = jax.random.PRNGKey(config["SEED"])
            calculate_team_diversity(
                networks, runner_state[0], _keys, num_agents, param_sharing=False
            )

        # Initialize tracking variables
        eval_metrics = []
        training_metrics = {agent: None for agent in range(num_agents)}
        start_time = time.time()
        global_step = 0
        next_eval_step = config["EVAL_INTERVAL"]
        next_capture_video_step = config.get("CAPTURE_VIDEO_INTERVAL", None)
        next_checkpoint_step = config.get("CHECKPOINT_INTERVAL", None)

        # Skip training if eval-only mode
        eval_only = config.get("EVAL_ONLY", False)
        if eval_only:
            return {"metrics": {}, "eval_metrics": [(0, eval_data)]}, env

        # Main training loop
        for update in range(config["NUM_UPDATES"]):
            final_update = update == config["NUM_UPDATES"] - 1
            
            # Perform update step and track time
            update_time_start = time.time()
            runner_state, ret_metric = _update_step(runner_state, None)
            global_step += config["NUM_STEPS"] * config["NUM_ENVS"]

            # Periodically log progress and performance
            if update % 100 == 0:
                print(f"update: {update}/{config['NUM_UPDATES']}")
                
                # Calculate steps per second
                sps = int(global_step / (time.time() - start_time))
                sps_update = int(
                    config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                    / (time.time() - update_time_start)
                )
                print("SPS:", sps, sps_update)
                
                # Log to wandb
                wb_run.log({"charts/SPS": sps}, global_step)
                wb_run.log({"charts/SPS_update": sps_update}, global_step)

            # Update metrics tracking
            for agent in range(num_agents):
                if training_metrics[agent] is None:
                    # Initialize metrics storage on first update
                    training_metrics[agent] = initialize_metrics_storage(
                        config, ret_metric[agent].keys()
                    )
                else:
                    # Update metrics with new values
                    training_metrics[agent] = update_metrics(
                        training_metrics[agent], ret_metric[agent], update
                    )

            # Periodic evaluation
            record_final_episode = config.get("CAPTURE_VIDEO_INTERVAL") and final_update
            if (global_step >= next_eval_step) or record_final_episode:
                # Determine if we should capture video
                if (
                    next_capture_video_step and global_step >= next_capture_video_step
                ) or record_final_episode:
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
                    shared_weights=False,
                    parallel=config.get("EVAL_PARALLEL", True),
                )
                
                next_eval_step += config["EVAL_INTERVAL"]
                eval_metrics.append((global_step, eval_data))

            # Save checkpoints at intervals or final update
            if (final_update) or (
                next_checkpoint_step and global_step >= next_checkpoint_step
            ):
                for agent in range(num_agents):
                    agent_key = f"agent_{agent}"
                    agent_identity = f"{agent_key}_{config['SEED']}_seed"
                    
                    # Create checkpoint path
                    model_path = f"{config['CHP_DIR']}/{config['EXP_NAME']}_{global_step}_steps_{update}_updates.{agent_identity}"
                    
                    # Save model parameters
                    save_args = orbax_utils.save_args_from_target(
                        runner_state[0][agent].params
                    )
                    checkpointers[agent_key].save(
                        model_path, runner_state[0][agent].params, save_args=save_args
                    )
                    print(f"model saved to {model_path} at step {global_step}")
                    
                    # Update next checkpoint step
                    next_checkpoint_step += config["CHECKPOINT_INTERVAL"]

        # Return final results
        return {
            "runner_state": runner_state,
            "metrics": training_metrics,
            "eval_metrics": eval_metrics,
        }, env

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="ippo_ff_nps_vmas_dispersion",
)
def main(config):
    """
    Main entry point for training.
    
    Args:
        config: Hydra configuration
    """
    print("starting training")
    config = OmegaConf.to_container(config, resolve=True)

    # Initialize wandb run
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

    # Initialize RNG
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Update environment kwargs
    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])

    # Create and run training function
    train = make_train(config)
    out, env = train(rngs[0], run)

    # Log metrics
    for agent in out["metrics"].keys():
        log_train_metrics(config, out["metrics"][agent], run, agent_id=agent)

    log_eval_metrics(config, out["eval_metrics"], run)

    # Clean up
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()