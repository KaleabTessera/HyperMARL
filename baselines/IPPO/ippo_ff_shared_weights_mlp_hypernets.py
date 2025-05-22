"""
MLP Hypernetworks for IPPO with shared weights.


HyperMARL generates both actor and critic parameters using a linear hypernetwork. Identical to linear HyperMARL but with MLP.
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
from typing import Any, Dict, List, NamedTuple, Tuple
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
from baselines.IPPO.ppo import get_update_fn_ff_ppo as get_update_fn
from baselines.utils.utils import (
    calculate_team_diversity,
    create_action_functions,
    extract_and_save_agent_embeddings,
    log_eval_metrics,
    log_train_metrics,
    log_videos,
)
from baselines.utils.eval import run_eval_jax
from baselines.utils.wrappers import make_env


@dataclass
class EpisodeStatistics:
    """Tracks statistics for episodes across multiple environments."""
    episode_returns: jnp.array        # Current episode returns
    episode_lengths: jnp.array        # Current episode lengths
    returned_episode_returns: jnp.array  # Returns of completed episodes
    returned_episode_lengths: jnp.array  # Lengths of completed episodes


class ActionType(Enum):
    """Supported action space types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class HyperNetType(Enum):
    """Types of hypernetworks for different parts of the policy."""
    ACTOR = 0   # Hypernetwork for actor networks
    CRITIC = 1  # Hypernetwork for critic networks


def check_rows_orthogonal(A):
    """
    Check if rows of a matrix are orthogonal to each other.
    
    Args:
        A: Matrix to check
        
    Returns:
        Boolean indicating if rows are orthogonal
    """
    dot_products = jnp.dot(A, A.T)
    off_diagonal_elements = dot_products - jnp.diag(jnp.diag(dot_products))
    is_orthogonal = jnp.all(jnp.isclose(off_diagonal_elements, l0, atol=0.0001))
    return is_orthogonal


class MLPHyperNetwork(nn.Module):
    """
    MLP HyperNetwork for generating weights and biases of target networks.
    
    Unlike linear hypernetworks, this uses multi-layer perceptrons to generate
    the parameters, allowing for more complex mappings between agent embeddings
    and policy weights.
    """
    output_dims: List[Tuple[int, int]]  # List of (input_dim, output_dim) for each layer
    hypernet_type: HyperNetType         # Type of hypernetwork (ACTOR or CRITIC)
    init_scale: float = np.sqrt(2)      # Default initialization scale for ReLU
    use_bias: bool = True               # Whether to use bias in the hypernetwork
    hidden_dims: List[int] = (64,)      # Hidden layer sizes for the MLP
    
    @staticmethod
    def hypernet_init(gain, fan_in, fan_out):
        """
        Custom initialization function for the hypernetwork weights.
        Simply takes an initialization function and applies it to each agent's generated weights e.g. orthogonal init weights are commonly used, so this ensures that this generates orthogonal weights for each agent at init.   

        Args:
            gain: Scaling factor for weights
            fan_in: Input dimension
            fan_out: Output dimension

        Returns:
            Initialization function
        """
        def weight_init(key, shape, dtype):
            init = jax.nn.initializers.orthogonal(gain)
            batched_init = jax.vmap(init, in_axes=(0, None, None))
            
            # Create a batch of keys (one per agent)
            batch_size = shape[0]
            keys = jax.random.split(key, num=batch_size)
            
            # Generate orthogonal weights for each agent
            weights = batched_init(keys, (fan_in, fan_out), dtype)
            return weights.reshape(shape)
        return weight_init

    @nn.compact
    def __call__(self, x):
        """
        Generate weights and biases for each layer of the target network.
        
        Args:
            x: Agent embeddings
            
        Returns:
            weight_heads: Generated weights for each layer
            bias_heads: Generated biases for each layer
        """
        weight_heads = []
        bias_heads = []
        for i, (input_dim, output_dim) in enumerate(self.output_dims):
            weight_dim = input_dim * output_dim
            bias_dim = output_dim
            
            is_final_layer = i == len(self.output_dims) - 1

            # Determine gain for initialization based on layer type and position
            if is_final_layer and self.hypernet_type == HyperNetType.ACTOR:
                gain = 0.01  # Lower gain for final actor layer (common in PPO)
            elif is_final_layer and self.hypernet_type == HyperNetType.CRITIC:
                gain = 1.0   # Standard gain for critic output
            else:
                gain = self.init_scale  # Hidden layer gain
                
            # MLP for weights generation
            weight_mlp = x
            for hidden_dim in self.hidden_dims:
                weight_mlp = nn.Dense(hidden_dim, use_bias=self.use_bias)(weight_mlp)
                weight_mlp = nn.relu(weight_mlp)
            weight_head = nn.Dense(
                weight_dim,
                use_bias=self.use_bias,
                kernel_init=self.hypernet_init(gain, input_dim, output_dim),
                bias_init=nn.initializers.zeros,
            )(weight_mlp)
            
            # MLP for biases generation
            bias_mlp = x
            for hidden_dim in self.hidden_dims:
                bias_mlp = nn.Dense(hidden_dim, use_bias=self.use_bias)(bias_mlp)
                bias_mlp = nn.relu(bias_mlp)
            bias_head = nn.Dense(
                bias_dim,
                use_bias=self.use_bias,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(bias_mlp)
            
            weight_heads.append(weight_head)
            bias_heads.append(bias_head)
        
        return weight_heads, bias_heads


class ActorCritic(nn.Module):
    """
    ActorCritic network using MLP hypernetworks to generate agent-specific parameters.
    
    The architecture uses MLP hypernetworks to generate weights for both the actor
    and critic networks from agent embeddings, allowing parameter sharing while 
    supporting complex agent-specific specialization.
    """
    action_dim: int            # Dimension of the action space
    num_agents: int            # Number of agents in the environment
    actor_layers: List[int]    # Hidden layer sizes for the actor network
    critic_layers: List[int]   # Hidden layer sizes for the critic network
    embedding_dim: int         # Dimension of the agent embeddings
    observation_dim: int       # Dimension of the observation space
    init_scale: float          # Scale for weight initialization
    activation: str = "tanh"   # Activation function to use
    use_agent_id_embeddings: bool = False  # Whether to use learned agent ID embeddings
    use_bias_in_hypernet: bool = False     # Whether to use bias in the hypernetwork
    is_continuous: bool = False            # Whether the action space is continuous
    hypernet_hidden_dims: List[int] = (64,)  # Hidden layer sizes for MLP hypernetworks

    def setup(self):
        """Initialize model components."""
        # Set up activation function
        self.activation_fn = jax.nn.relu if self.activation == "relu" else jax.nn.tanh
        
        # Initialize agent embeddings - either learned or one-hot
        self.agent_embeddings = self.param(
            "agent_embeddings",
            nn.initializers.orthogonal(self.init_scale),
            (self.num_agents, self.embedding_dim),
        ) if self.use_agent_id_embeddings else jnp.eye(self.num_agents)
        
        # Compute output dimensions for actor and critic networks
        self.actor_output_dims = self._compute_output_dims(self.actor_layers, self.action_dim)
        self.critic_output_dims = self._compute_output_dims(self.critic_layers, 1)

        # Initialize hypernetworks for actor and critic
        self.actor_hypernet = MLPHyperNetwork(
            output_dims=self.actor_output_dims,
            init_scale=self.init_scale,
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.ACTOR,
            hidden_dims=self.hypernet_hidden_dims
        )
        self.critic_hypernet = MLPHyperNetwork(
            output_dims=self.critic_output_dims,
            init_scale=self.init_scale,
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.CRITIC,
            hidden_dims=self.hypernet_hidden_dims
        )
        
        # Initialize log_std parameter for continuous actions
        if self.is_continuous:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

    def _compute_output_dims(self, layers, final_dim):
        """
        Compute the dimensions of each layer in the network.
        
        Args:
            layers: List of hidden layer sizes
            final_dim: Output dimension of the final layer
            
        Returns:
            List of (input_dim, output_dim) tuples for each layer
        """
        output_dims = []
        input_dim = self.observation_dim
        for layer_size in layers + (final_dim,):
            output_dims.append((input_dim, layer_size))
            input_dim = layer_size
        return output_dims

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the actor-critic network.
        
        Args:
            x: Input tensor containing observation and agent ID
            
        Returns:
            actor_outputs: Policy distribution parameters
            critic_values: Value function estimates
        """
        # Split input into observation and agent ID
        obs = x[..., :-self.num_agents]
        agent_id = jnp.argmax(x[..., -self.num_agents:], axis=-1)

        actor_outputs, critic_outputs = self._apply_networks(obs, agent_id)
        return actor_outputs, jnp.squeeze(critic_outputs, axis=-1)

    def _apply_networks(self, obs, agent_id):
        """
        Apply the hypernetworks to generate actor and critic outputs.
        
        Args:
            obs: Observation tensor
            agent_id: Agent IDs
            
        Returns:
            actor_outputs: Policy distribution parameters
            critic_outputs: Value function estimates
        """
        batch_size = obs.shape[0]
            
        # Pre-compute all hypernet outputs for all agents
        actor_weights, actor_biases = self.actor_hypernet(self.agent_embeddings)
        critic_weights, critic_biases = self.critic_hypernet(self.agent_embeddings)

        def apply_hypernet(obs, weights, biases):
            """
            Apply hypernetwork-generated weights to an input.
            
            Args:
                obs: Observation tensor
                weights: List of weight matrices
                biases: List of bias vectors
                
            Returns:
                Network output
            """
            x = obs
            # Apply all hidden layers with activation
            for w, b in zip(weights[:-1], biases[:-1]):
                x = self.activation_fn(jnp.matmul(x, w.reshape(x.shape[-1], -1)) + b)
            # Apply final layer without activation
            return jnp.matmul(x, weights[-1].reshape(x.shape[-1], -1)) + biases[-1]

        # Vectorize apply_hypernet over all agents
        vmap_apply_hypernet = jax.vmap(apply_hypernet, in_axes=(None, 0, 0))
        
        # Apply networks for all agents simultaneously
        # Shape: (num_agents, batch_size, action_dim) for actor
        # Shape: (num_agents, batch_size, 1) for critic
        actor_outputs_all = vmap_apply_hypernet(obs, actor_weights, actor_biases)
        critic_outputs_all = vmap_apply_hypernet(obs, critic_weights, critic_biases)
        
        # Select outputs for the specific agents in the batch
        batch_indices = jnp.arange(batch_size)
        actor_outputs = actor_outputs_all[agent_id, batch_indices]
        critic_outputs = critic_outputs_all[agent_id, batch_indices]

        # For continuous action spaces, return mean and log_std
        if self.is_continuous:
            return (actor_outputs, self.log_std), critic_outputs
        else:
            return actor_outputs, critic_outputs


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
    # Verify configuration is valid for parameter sharing
    assert config["env"]["ENV_KWARGS"].get("one_hot_encode_agent_id"), "one_hot_ids must be True"

    # Create vectorized environment
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
    
    # Update wandb config with calculated values
    wandb.config.update(
        {
            "MINIBATCH_SIZE": config["MINIBATCH_SIZE"],
            "NUM_UPDATES": config["NUM_UPDATES"]
        }, 
        allow_val_change=True
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

    def train(rng, wb_run=None):
        """
        Main training function.
        
        Args:
            rng: JAX random number generator key
            wb_run: Weights & Biases run object for logging
            
        Returns:
            Dictionary of results and trained environment
        """
        # True observation size without agent IDs
        true_obs_size = observation_size - env.num_agents
        
        # Initialize network architecture
        network = ActorCritic(
            action_dim=action_dim,
            activation=config["ACTIVATION"],
            actor_layers=config.get("ACTOR_LAYERS"),
            critic_layers=config.get("CRITIC_LAYERS"),
            num_agents=env.num_agents,
            embedding_dim=config.get("HYPERNET_EMBEDDING_DIM", None),
            use_agent_id_embeddings=config.get("USE_AGENT_ID_EMBEDDINGS", False),
            init_scale=config.get("INIT_SCALE", np.sqrt(2)),
            use_bias_in_hypernet=config.get("USE_BIAS_IN_HYPERNET", True),
            observation_dim=true_obs_size,
            is_continuous=action_type is ActionType.CONTINUOUS,
            hypernet_hidden_dims=config.get("HYPERNET_HIDDEN_DIMS", (64,))
        )
        
        print(
            f"Using agent ID embeddings: {config.get('USE_AGENT_ID_EMBEDDINGS', False)}, "
            f"Embedding dim: {config.get('HYPERNET_EMBEDDING_DIM', None)}"
        )
        
        # Initialize network parameters
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, observation_size))
        network_params = network.init(_rng, init_x)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
        
        

        # Log number of parameters
        wb_run.log({"num_params": param_count}, commit=False)

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

        # JIT compile network for efficiency
        network.apply = jax.jit(network.apply)

        # Load checkpoint if specified
        zero_shot_eval = False
        if config.get("CHECKPOINT_LOAD_DIR") is not None:
            print(f"Loading from {config.get('CHECKPOINT_LOAD_DIR')}")
            load_checkpointer = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename="checkpoints")
            )
            # Load parameters from checkpoint
            loaded_checkpoints = load_checkpointer.restore(
                config.get("CHECKPOINT_LOAD_DIR"), item=train_state.params
            )
            train_state = TrainState.create(
                apply_fn=train_state.apply_fn, params=loaded_checkpoints, tx=tx
            )
            zero_shot_eval = True

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

        # Preallocate arrays for transitions
        transitions = Transition(
            done=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=bool
            ),
            action=np.zeros(
                # Add extra dim if you can have more than 1 action e.g. continuous actions mean and logstd
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents) + 
                ((num_actions,) if num_actions == 2 else ()),
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

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        int_seed = jax.random.randint(
            reset_rng, shape=(1,), minval=1, maxval=1000000
        ).item()
        obsv, infos = env.reset(seed=int_seed)
        env_state = {}
        env_step = env.step

        def step_env_wrapped(action: Any) -> Any:
            """Take a step in the environment with the given action."""
            next_obs, reward, termination, truncs, info = env_step(action.flatten())
            return next_obs, reward, termination, truncs, infos

        @jax.jit
        def update_episode_stats(episode_stats, reward, done):
            """Update episode statistics based on environment step results."""
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
            return episode_stats.replace(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=returned_episode_returns,
                returned_episode_lengths=returned_episode_lengths,
            )

        @jax.jit
        def _select_action(params, obs, _rng):
            """
            Select actions using current policy.
            
            Args:
                params: Policy parameters
                obs: Current observations
                _rng: Random number generator key
                
            Returns:
                action: Selected actions
                log_prob: Log probabilities of selected actions
                value: Value function estimates
            """
            actor_output, value = network.apply(params, obs)
            
            # Create action distribution based on action type
            if action_type is ActionType.DISCRETE:
                pi = distrax.Categorical(logits=actor_output) 
            else:  # continuous
                actor_mean, actor_logtstd = actor_output
                pi = distrax.MultivariateNormalDiag(
                    loc=actor_mean, 
                    scale_diag=jnp.exp(actor_logtstd)
                )
                
            # Sample action and get log probability
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            
            return action, log_prob, value

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            """
            Select actions for evaluation (with optional deterministic policy).
            
            Args:
                params: Policy parameters
                obs: Current observations
                _rng: Random number generator key
                
            Returns:
                actions: Selected actions
            """
            actor_output, _ = network.apply(params, obs)
            
            # Create action distribution based on action type
            if action_type is ActionType.DISCRETE:
                pi = distrax.Categorical(logits=actor_output)    
            else:  # continuous
                actor_mean, actor_logtstd = actor_output
                pi = distrax.MultivariateNormalDiag(
                    loc=actor_mean, 
                    scale_diag=jnp.exp(actor_logtstd)
                )

            # Use deterministic policy if configured, otherwise sample
            if config.get("EVAL_DETERMINISTIC") is False:
                action = pi.sample(seed=_rng)
            else:
                action = pi.mode()
                
            return (action,)

        # Get PPO update function
        calculate_advantage_and_update_ppo = get_update_fn(config, network)

        # Define single update step
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

            # Collect trajectory data by stepping through environment
            for t in range(config["NUM_STEPS"]):
                # Select actions using current policy
                rng, _rng = jax.random.split(rng)
                action, log_prob, value = _select_action(
                    train_state.params, last_obs, _rng
                )

                # Convert JAX arrays to NumPy for environment interaction
                np_action = np.array(action)
                obsv, reward, termination, truncs, info = step_env_wrapped(np_action)

                # Update statistics
                done = np.logical_or(termination, truncs)
                episode_stats = update_episode_stats(episode_stats, reward, done)

                # Store transitions in preallocated arrays
                transitions.done[t] = done
                transitions.action[t] = action
                transitions.value[t] = value
                transitions.reward[t] = reward
                transitions.log_prob[t] = log_prob
                transitions.obs[t] = last_obs
                transitions.info.returned_episode_returns[t] = (
                    episode_stats.returned_episode_returns
                )
                transitions.info.returned_episode_lengths[t] = (
                    episode_stats.returned_episode_lengths
                )

                # Update observation for next step
                last_obs = obsv

            # Update policy using collected trajectory
            update_state, loss_info = calculate_advantage_and_update_ppo(
                transitions, last_obs, train_state, rng
            )
            train_state = update_state[0]
            
            # Collect metrics for logging
            metric = {
                "returned_episode_returns": episode_stats.returned_episode_returns,
                "returned_episode_lengths": episode_stats.returned_episode_lengths,
            }
            rng = update_state[-1]

            # Compute mean metrics for logging
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info}

            runner_state = (train_state, env_state, last_obs, rng, episode_stats)
            return runner_state, metric

        # Initialize runner state
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, episode_stats)

        # Initialize checkpointer for model saving
        checkpointers = checkpointer.Checkpointer(
            PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
        )

        # Perform zero-shot evaluation if loading from checkpoint
        if zero_shot_eval:
            print("running zero shot eval")
            # Determine whether to capture video based on eval episodes
            capture_video = config.get("EVAL_EPISODES") <= 40
            
            # Run evaluation
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
            )

        # Calculate team diversity metrics if configured
        calculate_team_diversity_metrics = config.get(
            "CALCULATE_TEAM_DIVERSITY_METRICS", False
        )
        if calculate_team_diversity_metrics:
            _keys = jax.random.PRNGKey(config["SEED"])
            calculate_team_diversity(
                network, runner_state[0], _keys, num_agents, param_sharing=True
            )

        # Initialize tracking variables
        training_metrics = None
        eval_metrics = []
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

            # Initialize or update metrics storage
            if training_metrics is None:
                training_metrics = initialize_metrics_storage(config, ret_metric.keys())
            training_metrics = update_metrics(training_metrics, ret_metric, update)

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

            # Extract and save agent embeddings
            if update % config.get("SAVE_EMBEDDINGS_INTERVAL", 5000) == 0 or final_update:
                extract_and_save_agent_embeddings(train_state.params, global_step, config, wb_run)
    
            # Save checkpoints at intervals or final update
            if (final_update) or (
                next_checkpoint_step and global_step >= next_checkpoint_step
            ):
                agent_identity = f"agent_{config['SEED']}_seed"
                
                # Create checkpoint path
                model_path = f"{config['CHP_DIR']}/{config['EXP_NAME']}_{global_step}_steps_{update}_updates.{agent_identity}"
                
                # Save model parameters
                save_args = orbax_utils.save_args_from_target(runner_state[0].params)
                checkpointers.save(
                    model_path, runner_state[0].params, save_args=save_args
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
    config_name="ippo_ff_shared_weights_vmas_dispersion",
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
    log_train_metrics(config, out["metrics"], run)
    log_eval_metrics(config, out["eval_metrics"], run)
    
    # Clean up
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()