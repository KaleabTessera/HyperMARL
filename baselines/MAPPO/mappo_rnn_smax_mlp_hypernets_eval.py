"""
MLP Hypernetworks for shared IPPO with Recurrent Networks for SMAX.

HyperMARL only generates the actor and critic
feedforward weights, not the GRU weights. 

Initially built off this version https://github.com/FLAIROx/JaxMARL/blob/main/baselines/MAPPO/mappo_rnn_smax.py
"""

from enum import Enum
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import List, Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from baselines.utils.smax_utils import log_experiment_results
from jaxmarl.environments.spaces import Box


class HyperNetType(Enum):
    """
    Enum to distinguish between different hypernetwork types.
    
    Different types may use different initialization strategies.
    """
    EMBEDDING = 0
    ACTOR = 1
    CRITIC = 2


class SMAXAppendAgentID(JaxMARLWrapper):
    """
    Wrapper to provide agent IDs and world state observations for the centralized critic.
    
    Adds one-hot encoded agent IDs to observations and creates a global world state
    representation for the critic.
    
    Attributes:
        env: The SMAX environment to wrap
        obs_with_agent_id: Whether to append agent IDs to observations
        id_in_world_state: Whether to include agent IDs in world state
    """
    
    def __init__(self,
                 env: HeuristicEnemySMAX,
                 obs_with_agent_id=True,
                 id_in_world_state=False):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        self.id_in_world_state = id_in_world_state
        
        if self.obs_with_agent_id:
            # Only add IDs to observations for hypernetwork input, not used in forward passes
            self.obs_size = self.obs_size + self._env.num_allies
            self.state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id
            self.observation_spaces = {
                i: Box(low=0.0, high=1.0, shape=(self.obs_size,)) for i in self.agents
            }
            
        if self.id_in_world_state:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id
        else:
            # No ID in world state, just use environment state
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
    
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        """
        Reset the environment and append agent IDs to observations.
        
        Args:
            key: JAX random key
            
        Returns:
            obs: Augmented observations with agent IDs
            env_state: Environment state
        """
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        
        # Add agent IDs to each agent's observation
        agent_ids = jnp.eye(self._env.num_allies)
        for i, agent_key in enumerate(self._env.agents):
            obs[agent_key] = jnp.concatenate([obs[agent_key], agent_ids[i]])
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        """
        Step the environment and append agent IDs to observations.
        
        Args:
            key: JAX random key
            state: Current environment state
            action: Actions to take
            
        Returns:
            obs: Augmented observations with agent IDs
            env_state: New environment state
            reward: Rewards received
            done: Done flags
            info: Additional information
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state_fn(obs, state)
        
        # Add agent IDs to each agent's observation
        agent_ids = jnp.eye(self._env.num_allies)
        for i, agent_key in enumerate(self._env.agents):
            obs[agent_key] = jnp.concatenate([obs[agent_key], agent_ids[i]])
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        """
        Create world state without agent IDs.
        
        Args:
            obs: Current observations
            state: Current environment state
            
        Returns:
            World state for each agent
        """
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state
        
    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        """
        Create world state with agent IDs.
        
        Args:
            obs: Current observations
            state: Current environment state
            
        Returns:
            World state with agent IDs for each agent
        """
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)
        
    def world_state_size(self):
        """Get size of world state."""
        return self._world_state_size 
    
    def _get_obs_size(self):
        """Get size of observations."""
        return self.obs_size
    
    def observation_space(self, agent: str):
        """Get observation space for an agent."""
        return self.observation_spaces[agent]


class ScannedRNN(nn.Module):
    """
    Scanned RNN module that applies GRU cells across a sequence.
    
    Handles proper state reset based on episode termination flags,
    maintaining recurrent state across sequences of observations.
    """
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """
        Apply GRU cell to input sequence.
        
        Args:
            carry: Recurrent state
            x: Tuple of (input, reset flags)
            
        Returns:
            New recurrent state and output
        """
        rnn_state = carry
        ins, resets = x
        # Reset state when episode terminates
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """
        Initialize recurrent state.
        
        Args:
            batch_size: Number of sequences in batch
            hidden_size: Size of hidden state
            
        Returns:
            Initial recurrent state
        """
        # Use a dummy key since the default state init fn is just zeros
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class MLPHyperNetwork(nn.Module):
    """
    MLP HyperNetwork for generating weights and biases of target networks.
    
    Uses multi-layer perceptrons to transform agent embeddings into network parameters,
    allowing for more complex weight generation.
    
    Attributes:
        output_dims: List of (input_dim, output_dim) tuples for each layer
        hypernet_type: Type of hypernetwork (actor or critic)
        init_scale: Initialization scale factor
        use_bias: Whether to use bias in the hypernetwork
        hidden_dims: Sizes of hidden layers in the MLP hypernetwork
        shared_hypernet_base: Whether to use shared base layers for all outputs
    """
    output_dims: List[Tuple[int, int]]
    hypernet_type: HyperNetType 
    init_scale: float = np.sqrt(2)  # Default when using ReLU
    use_bias: bool = True 
    hidden_dims: List[int] = (64,)  # Default hidden layer sizes for the MLP
    shared_hypernet_base: bool = False
    
    @staticmethod
    def hypernet_init(gain, fan_in, fan_out):
        """
        Custom weight initialization for hypernetworks.
        
        Args:
            gain: Gain factor for initialization
            fan_in: Input dimension
            fan_out: Output dimension
            
        Returns:
            Weight initialization function
        """
        def weight_init(key, shape, dtype):
            init = jax.nn.initializers.orthogonal(gain)
            batched_init = jax.vmap(init, in_axes=(0, None, None))
            
            batch_size = shape[0]
            keys = jax.random.split(key, num=batch_size)
        
            weights = batched_init(keys, (fan_in, fan_out), dtype)
            return weights.reshape(shape)
        return weight_init

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the hypernetwork.
        
        Args:
            x: Agent embeddings
            
        Returns:
            weight_heads: Generated weights for each layer
            bias_heads: Generated biases for each layer
        """
        # Use base shared layers for hypernetwork if specified
        if self.shared_hypernet_base:
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(
                    hidden_dim, 
                    use_bias=self.use_bias,
                    kernel_init=nn.initializers.orthogonal(np.sqrt(2))
                )(x)
                x = nn.relu(x)
 
        weight_heads = []
        bias_heads = []
        
        # Generate weights and biases for each target layer
        for i, (input_dim, output_dim) in enumerate(self.output_dims):
            weight_dim = input_dim * output_dim
            bias_dim = output_dim
            
            is_final_layer = i == len(self.output_dims) - 1

            # Adjust gain based on layer position and network type
            if is_final_layer and self.hypernet_type == HyperNetType.ACTOR:
                gain = 0.01  # Small gain for actor output layer
            elif is_final_layer and self.hypernet_type == HyperNetType.CRITIC:
                gain = 1.0  # Standard gain for critic output layer
            else:
                gain = self.init_scale
                
            # MLP for weights
            weight_mlp = x
            if not self.shared_hypernet_base:
                for hidden_dim in self.hidden_dims:
                    weight_mlp = nn.Dense(
                        hidden_dim, 
                        use_bias=self.use_bias,
                        kernel_init=nn.initializers.orthogonal(np.sqrt(2))
                    )(weight_mlp)
                    weight_mlp = nn.relu(weight_mlp)
                    
            weight_head = nn.Dense(
                weight_dim,
                use_bias=self.use_bias,
                kernel_init=self.hypernet_init(gain, input_dim, output_dim),
                bias_init=nn.initializers.zeros,
            )(weight_mlp)
            
            # MLP for biases
            bias_mlp = x
            if not self.shared_hypernet_base:
                for hidden_dim in self.hidden_dims:
                    bias_mlp = nn.Dense(
                        hidden_dim, 
                        use_bias=self.use_bias,
                        kernel_init=nn.initializers.orthogonal(np.sqrt(2))
                    )(bias_mlp)
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


class ActorRNN(nn.Module):
    """
    Actor network with recurrent architecture and hypernetwork-generated parameters.
    
    Uses GRU cells to maintain state across time steps and hypernetworks to
    generate parameters for each agent.
    
    Attributes:
        action_dim: Dimension of action space
        config: Configuration dictionary
        num_agents: Number of agents in the environment
    """
    action_dim: Sequence[int]
    config: Dict
    num_agents: int

    def setup(self):
        """Initialize model components."""
        self.embedding_dim = self.config.get("HYPERNET_EMBEDDING_DIM", 64)
        self.init_scale = self.config.get("HYPERNET_INIT_SCALE", jnp.sqrt(2))
        self.use_agent_id_embeddings = self.config.get("USE_AGENT_ID_EMBEDDINGS", True)
        self.use_bias_in_hypernet = self.config.get("USE_BIAS_IN_HYPERNET", True)
        self.activation_fn = jax.nn.relu
        
        # Initialize agent embeddings (learnable or one-hot)
        self.agent_embeddings = self.param(
            "agent_embeddings",
            nn.initializers.orthogonal(self.init_scale),
            (self.num_agents, self.embedding_dim),
        ) if self.use_agent_id_embeddings else jnp.eye(self.num_agents)
        
        hidden_dim = self.config["GRU_HIDDEN_DIM"]
        fc_dim = self.config["FC_DIM_SIZE"]
        
        # Define network architecture
        self.output_dims = [
            (fc_dim, hidden_dim),  # After RNN
            (hidden_dim, self.action_dim)  # Final actor output
        ]

        # Create hypernetwork to generate network parameters
        hyper_hidden_dims = self.config.get("HYPERNET_HIDDEN_DIMS", (64,))
        self.hypernet = MLPHyperNetwork(
            output_dims=self.output_dims,
            init_scale=self.init_scale, 
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.ACTOR,
            hidden_dims=hyper_hidden_dims,
            shared_hypernet_base=self.config.get("SHARED_HYPERNET_BASE", False)
        )

    @nn.compact
    def __call__(self, hidden, x):
        """
        Forward pass through the actor network.
        
        Args:
            hidden: Initial recurrent state
            x: Tuple of (observations, dones, available actions)
            
        Returns:
            hidden: Updated recurrent state
            pi: Action probability distribution
        """
        obs, dones, avail_actions = x
        
        # Extract agent IDs from one-hot encoding
        one_hot = obs[..., -self.num_agents:]
        agent_id = jnp.argmax(one_hot, axis=-1)
        obs = obs[..., :-self.num_agents]
        
        # Initial embedding layer
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], 
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), 
            bias_init=nn.initializers.constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        
        # Apply RNN
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Generate parameters with hypernetwork
        weights, biases = self.hypernet(self.agent_embeddings)
        
        # Apply hypernetwork-generated parameters
        def apply_hypernet(x, weights, biases):
            """Apply hypernetwork-generated parameters to input."""
            for w, b in zip(weights[:-1], biases[:-1]):
                x = self.activation_fn(jnp.matmul(x, w.reshape(x.shape[-1], -1)) + b)
            return jnp.matmul(x, weights[-1].reshape(x.shape[-1], -1)) + biases[-1] 

        # Apply the same function across all agents
        vmap_apply_hypernet = jax.vmap(apply_hypernet, in_axes=(None, 0, 0))
        all_actor_out = vmap_apply_hypernet(embedding, weights, biases)
        
        # Select outputs for each agent based on agent_id
        actor_out = all_actor_out[agent_id, jnp.arange(embedding.shape[0])[:, None], jnp.arange(embedding.shape[1])]

        # Mask unavailable actions
        unavail_actions = 1 - avail_actions
        action_logits = actor_out - (unavail_actions * 1e10)

        # Create categorical distribution
        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    """
    Critic network with recurrent architecture and hypernetwork-generated parameters.
    
    Uses GRU cells to maintain state across time steps and hypernetworks to
    generate parameters for each agent's value function.
    
    Attributes:
        config: Configuration dictionary
        num_agents: Number of agents in the environment
    """
    config: Dict
    num_agents: int

    def setup(self):
        """Initialize model components."""
        self.embedding_dim = self.config.get("HYPERNET_EMBEDDING_DIM", 64)
        self.init_scale = self.config.get("HYPERNET_INIT_SCALE", jnp.sqrt(2))
        self.use_agent_id_embeddings = self.config.get("USE_AGENT_ID_EMBEDDINGS", True)
        self.use_bias_in_hypernet = self.config.get("USE_BIAS_IN_HYPERNET", True)
        self.activation_fn = jax.nn.relu
        
        # Initialize agent embeddings (learnable or one-hot)
        self.agent_embeddings = self.param(
            "agent_embeddings",
            nn.initializers.orthogonal(self.init_scale),
            (self.num_agents, self.embedding_dim),
        ) if self.use_agent_id_embeddings else jnp.eye(self.num_agents)
        
        hidden_dim = self.config["GRU_HIDDEN_DIM"]
        fc_dim = self.config["FC_DIM_SIZE"]
        
        # Define network architecture
        self.output_dims = [
            (fc_dim, hidden_dim),  # After RNN
            (hidden_dim, 1)  # Final critic output
        ]

        # Create hypernetwork to generate network parameters
        hyper_hidden_dims = self.config.get("HYPERNET_HIDDEN_DIMS", (64,))
        self.hypernet = MLPHyperNetwork(
            output_dims=self.output_dims,
            init_scale=self.init_scale,
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.CRITIC,
            hidden_dims=hyper_hidden_dims,
            shared_hypernet_base=self.config.get("SHARED_HYPERNET_BASE", False)
        )

    @nn.compact
    def __call__(self, hidden, x, obs_with_agent_id):
        """
        Forward pass through the critic network.
        
        Args:
            hidden: Initial recurrent state
            x: Tuple of (world state, dones)
            obs_with_agent_id: Observations with agent IDs for agent identification
            
        Returns:
            hidden: Updated recurrent state
            critic: Value function estimates
        """
        # Extract agent IDs from one-hot encoding
        one_hot = obs_with_agent_id[..., -self.num_agents:]
        agent_id = jnp.argmax(one_hot, axis=-1)
        
        world_state, dones = x
        
        # Initial embedding layer
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], 
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), 
            bias_init=nn.initializers.constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        # Apply RNN
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Generate parameters with hypernetwork
        weights, biases = self.hypernet(self.agent_embeddings)
        
        # Apply hypernetwork-generated parameters
        def apply_hypernet(x, weights, biases):
            """Apply hypernetwork-generated parameters to input."""
            for w, b in zip(weights[:-1], biases[:-1]):
                x = self.activation_fn(jnp.matmul(x, w.reshape(x.shape[-1], -1)) + b)
            return jnp.matmul(x, weights[-1].reshape(x.shape[-1], -1)) + biases[-1] 

        # Apply the same function across all agents
        vmap_apply_hypernet = jax.vmap(apply_hypernet, in_axes=(None, 0, 0))
        all_critic_out = vmap_apply_hypernet(embedding, weights, biases)
        
        # Select outputs for each agent based on agent_id
        critic = all_critic_out[agent_id, jnp.arange(embedding.shape[0])[:, None], jnp.arange(all_critic_out.shape[2])]

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """
    Stores transition information for a single environment step.
    
    Attributes:
        global_done: Global done flag
        done: Done flags for each agent
        action: Actions taken by each agent
        value: Value estimates
        reward: Rewards received
        log_prob: Log probabilities of actions
        obs: Observations (local)
        world_state: Global world state
        info: Additional information
        avail_actions: Available actions mask
    """
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    """
    Convert a dictionary of agent observations to a batched array.
    
    Args:
        x: Dictionary mapping agent IDs to observations
        agent_list: List of agent IDs
        num_actors: Total number of actors
        
    Returns:
        Batched array of shape (num_actors, ...)
    """
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """
    Convert a batched array back to a dictionary of agent observations.
    
    Args:
        x: Batched array of shape (num_actors, ...)
        agent_list: List of agent IDs
        num_envs: Number of environments
        num_actors: Total number of actors
        
    Returns:
        Dictionary mapping agent IDs to observations
    """
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class CustomTrainState(TrainState):
    """
    Extended TrainState to track number of updates.
    
    Attributes:
        n_updates: Number of updates performed
    """
    n_updates: int = 0


def make_train(config):
    """
    Create the training function based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train: Function that performs training
    """
    # Initialize environment
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    
    # Calculate derived parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    # Apply wrappers
    env = SMAXAppendAgentID(env, True, id_in_world_state=config.get("ID_IN_WORLD_STATE", False))
    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        """Linear learning rate decay schedule."""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        """
        Main training function.
        
        Args:
            rng: JAX random number generator key
            
        Returns:
            Dictionary of training results
        """
        original_seed = rng[0]
        
        # Initialize networks
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config, num_agents=env.num_agents)
        critic_network = CriticRNN(config=config, num_agents=env.num_agents)
        
        # Initialize network parameters
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        # Using ac_init_x[0] only to get agent ID
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x, ac_init_x[0])

        # Count parameters
        actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_network_params))
        critic_param_count = sum(x.size for x in jax.tree_util.tree_leaves(critic_network_params))
        param_count = actor_param_count + critic_param_count
        
        # Configure optimizers with optional learning rate annealing
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            
        # Create train states
        actor_train_state = CustomTrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = CustomTrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # Create test environment for evaluation
        scenario = map_name_to_scenario(config["MAP_NAME"])
        test_env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
        test_env = SMAXAppendAgentID(test_env, True, id_in_world_state=config.get("ID_IN_WORLD_STATE", False))
        test_env = SMAXLogWrapper(test_env)

        def run_eval(rng, train_states):
            """
            Evaluate policy during training.
            
            Args:
                rng: JAX random key
                train_states: Current training states
                
            Returns:
                Evaluation metrics
            """
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            
            actor_params, critic_params = train_states[0].params, train_states[1].params
            
            def _eval_step(step_state, unused):
                """
                Perform a single evaluation step.
                
                Args:
                    step_state: Current evaluation state
                    unused: Unused parameter for JAX compatibility
                    
                Returns:
                    Updated evaluation state and metrics
                """
                actor_params, critic_params, env_state, last_obs, last_done, ac_hstate, cr_hstate, rng = step_state
                
                # Select action (deterministic policy during evaluation)
                num_actors_eval = config["TEST_NUM_ENVS"] * env.num_agents
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors_eval)
                )
                obs_batch = batchify(last_obs, env.agents, num_actors_eval)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                ac_hstate, pi = actor_network.apply(actor_params, ac_hstate, ac_in)
                action = pi.mode()  # Use mode for deterministic evaluation
                
                # Convert actions to environment format
                env_act = unbatchify(
                    action, env.agents, config["TEST_NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["TEST_NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # Process results
                infos = jax.tree_map(lambda x: x.reshape((num_actors_eval)), info)
                rewards = batchify(reward, env.agents, num_actors_eval).squeeze()
                done_batch = batchify(done, env.agents, num_actors_eval).squeeze()
                
                # Update critic state (not used for action selection in eval, but maintaining for consistency)
                world_state = obsv["world_state"].swapaxes(0,1)
                world_state = world_state.reshape((num_actors_eval,-1))
                cr_in = (
                    world_state[None, :],
                    done_batch[np.newaxis, :],
                )
                cr_hstate, _ = critic_network.apply(critic_params, cr_hstate, cr_in, ac_in[0])
                
                step_state = (actor_params, critic_params, env_state, obsv, done_batch, ac_hstate, cr_hstate, rng)
                return step_state, (rewards, done_batch, infos)

            # Initialize evaluation
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            init_obs, env_state = jax.vmap(test_env.reset, in_axes=0)(keys)
            
            num_eval_actors = config["TEST_NUM_ENVS"] * env.num_agents
            init_dones = jnp.zeros((num_eval_actors), dtype=bool)
            rng, _rng = jax.random.split(rng)
            ac_hstate = ScannedRNN.initialize_carry(
                num_eval_actors, config["GRU_HIDDEN_DIM"]
            )
            cr_hstate = ScannedRNN.initialize_carry(
                num_eval_actors, config["GRU_HIDDEN_DIM"]
            )
            
            # Run evaluation episodes
            step_state = (
                actor_params,
                critic_params,
                env_state,
                init_obs,
                init_dones,
                ac_hstate,
                cr_hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _eval_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            
            # Calculate evaluation metrics
            metrics = jax.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return metrics
        
        # Main training loop
        def _update_step(update_runner_state, unused):
            """
            Perform a single training update step.
            
            Args:
                update_runner_state: Current runner state and update counter
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated runner state, update counter, and metrics
            """
            # Unpack runner state
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                """
                Perform a single environment step and collect transition.
                
                Args:
                    runner_state: Current runner state
                    unused: Unused parameter for JAX compatibility
                    
                Returns:
                    Updated runner state and transition
                """
                train_states, env_state, last_obs, last_done, hstates, rng, test_state = runner_state

                # Select action
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)  # Stochastic action during training
                log_prob = pi.log_prob(action)
                
                # Convert actions to environment format
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # Get value estimates
                world_state = last_obs["world_state"].swapaxes(0,1)  
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in, ac_in[0])

                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                # Process results
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                
                # Store transition
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )
                
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng, test_state)
                return runner_state, transition

            # Collect trajectories
            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # Calculate advantages using GAE
            train_states, env_state, last_obs, last_done, hstates, rng, test_state = runner_state
            
            # Get value of last observation for bootstrapping
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            
            # Only to get agent ID
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            
            _, last_val = critic_network.apply(
                train_states[1].params, hstates[1], cr_in, last_obs_batch[np.newaxis, :]
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                """
                Calculate Generalized Advantage Estimation.
                
                Args:
                    traj_batch: Batch of transitions
                    last_val: Value estimate of final state
                    
                Returns:
                    advantages: Advantage estimates
                    targets: Value targets
                """
                def _get_advantages(gae_and_next_value, transition):
                    """Calculate advantage for a single transition."""
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Update networks
            def _update_epoch(update_state, unused):
                """
                Perform a single PPO update epoch.
                
                Args:
                    update_state: Current update state
                    unused: Unused parameter for JAX compatibility
                    
                Returns:
                    Updated update state and loss information
                """
                def _update_minibatch(train_states, batch_info):
                    """
                    Update networks with a single minibatch.
                    
                    Args:
                        train_states: Current train states
                        batch_info: Information for the current batch
                        
                    Returns:
                        Updated train states and loss information
                    """
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        """Calculate actor loss."""
                        # Rerun network to get updated policy distribution
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Calculate PPO objective
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)  # Normalize advantages
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        # Calculate diagnostics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        """Calculate critic loss."""
                        # Rerun critic network
                        _, value = critic_network.apply(
                            critic_params, 
                            init_hstate.squeeze(), 
                            (traj_batch.world_state, traj_batch.done), 
                            traj_batch.obs
                        ) 
                        
                        # Calculate value loss with clipping
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    # Compute gradients and update actor
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    
                    # Compute gradients and update critic
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
                    # Apply gradients
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    # Collect loss information
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                # Unpack update state
                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    test_state
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # Reshape initial hidden states
                init_hstates = jax.tree_map(lambda x: jnp.reshape(
                    x, (1, config["NUM_ACTORS"], -1)
                ), init_hstates)
                
                # Prepare batch data
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                
                # Shuffle data for each epoch
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                # Split into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                # Update all minibatches
                train_states, loss_info = jax.lax.scan(
                    _update_minibatch, train_states, minibatches
                )
                
                update_state = (
                    train_states,
                    jax.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    test_state
                )
                return update_state, loss_info

            # Perform multiple update epochs
            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
                test_state
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            # Save specific sample of ratio for detailed monitoring
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            
            # Extract updated states and collect metrics
            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-2]

            def callback(metric, original_seed):
                """
                Log metrics to wandb.
                
                Args:
                    metric: Dictionary of metrics
                    original_seed: Original RNG seed
                """
                # Add identifier per RNG
                metric.update(
                    {f"rng{int(original_seed)}/{k}": v for k, v in metric.items() if k != 'loss'}
                )

                # Prepare the log dictionary
                log_dict = {
                    # Test metrics are already masked by the returned_episode mask
                    f"rng{int(original_seed)}/test_returns": metric["test_returned_episode_returns"],
                    f"rng{int(original_seed)}/test_win_rate": metric["test_returned_won_episode"],
                    f"rng{int(original_seed)}/returns": metric["returned_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    f"rng{int(original_seed)}/win_rate": metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    f"rng{int(original_seed)}/env_step": metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                }

                # Handle the nested 'loss' dictionary
                if 'loss' in metric:
                    for loss_key, loss_value in metric['loss'].items():
                        log_dict[f"rng{int(original_seed)}/loss/{loss_key}"] = loss_value

                wandb.log(log_dict)

            # Update train state counters
            train_states = (
                train_states[0].replace(n_updates=train_states[0].n_updates + 1),
                train_states[1].replace(n_updates=train_states[1].n_updates + 1)
            )
            metric["update_steps"] = update_steps
            
            # Run evaluation periodically
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_states[0].n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: run_eval(_rng, train_states),
                    lambda _: test_state,
                    operand=None,
                )
                metric.update({"test_" + k: v for k, v in test_state.items()})
                
            # Log metrics
            jax.debug.callback(callback, metric, original_seed)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng, test_state)
            return (runner_state, update_steps), metric

        # Initialize evaluation
        rng, _rng = jax.random.split(rng)
        test_state = run_eval(_rng, (actor_train_state, critic_train_state))
        
        # Initialize runner state
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
            test_state
        )
        
        # Run training
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_rnn_smax")
def main(config):
    """
    Main entry point for training.
    
    Args:
        config: Hydra configuration
    """
    config = OmegaConf.to_container(config)
    from datetime import datetime
    now = datetime.now()
    name = f"mappo_rnn_shared_smax_hypernets_{config['MAP_NAME']}_{now:%Y-%m-%d_%H-%M-%S}"
    tags = ["MAPPO", "RNN", "MLP", "Hypernets"] if config.get("EXP_TAGS") is None else config.get("EXP_TAGS")
    
    # Initialize wandb
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
        name=name,
        reinit=True,
        save_code=True,
    )
    
    # Initialize RNG
    rng = jax.random.PRNGKey(config["SEED"])
    
    # Run single or multiple seeds
    if config.get("NUM_SEEDS") is None:
        train_jit = jax.jit(make_train(config), device=jax.devices()[0])
        out = train_jit(rng)
    else:
        rngs = jax.random.split(rng, config["NUM_SEEDS"])    
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)
        
        # Log results across seeds
        log_experiment_results(config, out, axis=(0, 2, 3, 4))
    
    # Save model if requested
    try:
        import orbax
        orbax_installed = True
    except ImportError:
        orbax_installed = False

    if orbax_installed and config['SAVE_PATH'] is not None:
        import os 
        from orbax.checkpoint import checkpointer
        from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler
        from flax.training import orbax_utils
        
        # Set up checkpointer
        checkpointers = checkpointer.Checkpointer(
            PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
        )

        # Extract parameters
        params = out['runner_state'][0][0][0].params
        save_dir = f"{wandb.run.dir}/models/{run.name}"  # type: ignore
        path = f'{save_dir}/final'
        
        # Save checkpoint
        save_args = orbax_utils.save_args_from_target(params)
        checkpointers.save(
            path, params, save_args=save_args
        )
        print(f"Model saved to {save_dir} {path}")

        # Upload to wandb as artifact
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_dir(path)
        run.log_artifact(artifact)  # type: ignore
    else:
        if not orbax_installed:
            print("Orbax is not installed. Skipping checkpoint saving.")
        elif config['SAVE_PATH'] is None:
            print("SAVE_PATH is not set. Skipping checkpoint saving.") 


if __name__ == "__main__":
    main()