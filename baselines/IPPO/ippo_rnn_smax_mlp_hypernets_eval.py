"""
MLP Hypernetworks for shared IPPO with Recurrent Networks for SMAX.

HyperMARL only generates the actor and critic
feedforward weights, not the GRU weights. 

Initially built off this version https://github.com/FLAIROx/JaxMARL/blob/main/baselines/IPPO/ippo_rnn_smax.py
"""

from enum import Enum
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import List, Sequence, NamedTuple, Any, Dict, Tuple
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf

from baselines.utils.smax_utils import log_experiment_results
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

import wandb
import functools
import matplotlib.pyplot as plt
from jaxmarl.environments.spaces import Box
from functools import partial

class SMAXAppendAgentID(JaxMARLWrapper):
    """
    Wrapper that appends one-hot agent IDs to observations.
    
    This wrapper enables parameter sharing by making agent identity explicit
    in the observation space, allowing the network to distinguish between agents.
    """
    
    def __init__(self,
                 env: HeuristicEnemySMAX,
                 obs_with_agent_id=True,):
        """
        Initialize the wrapper.
        
        Args:
            env: The SMAX environment to wrap
            obs_with_agent_id: Whether to append agent IDs to observations
        """
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        
        if self.obs_with_agent_id:
            # Extend observation space to include agent IDs
            self.obs_size = self.obs_size + self._env.num_allies
            self.state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id
            self.observation_spaces = {
                i: Box(low=0.0, high=1.0, shape=(self.obs_size,)) for i in self.agents
            }
        else:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        """
        Reset the environment and append agent IDs to observations.
        
        Args:
            key: Random key for environment reset
            
        Returns:
            Observations with agent IDs and environment state
        """
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        
        # Add agent IDs to each agent's observation
        agent_ids = jnp.eye(self._env.num_allies)
        for i, agent_key in enumerate(self._env.agents):
            obs[agent_key] = jnp.concatenate([obs[agent_key], agent_ids[i]])
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        """
        Step the environment and append agent IDs to observations.
        
        Args:
            key: Random key for environment step
            state: Current environment state
            action: Actions to take
            
        Returns:
            Observations with agent IDs, new state, rewards, dones, and info
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
        Return world state without agent IDs.
        
        Args:
            obs: Current observations
            state: Current environment state
            
        Returns:
            World state repeated for each agent
        """
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state
        
    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        """
        Return world state with agent IDs.
        
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
        """Return the size of the world state."""
        return self._world_state_size 
    
    def _get_obs_size(self):
        """Return the size of agent observations."""
        return self.obs_size
    
    def observation_space(self, agent: str):
        """Get observation space for a given agent."""
        return self.observation_spaces[agent]


class HyperNetType(Enum):
    """Types of hypernetworks for different network components."""
    EMBEDDING = 0  # For generating embeddings
    ACTOR = 1      # For generating actor network parameters
    CRITIC = 2     # For generating critic network parameters


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
    is_orthogonal = jnp.all(jnp.isclose(off_diagonal_elements, 0, atol=0.0001))
    return is_orthogonal


class MLPHyperNetwork(nn.Module):
    """
    MLP HyperNetwork for generating weights and biases of target networks.
    
    Uses multi-layer perceptrons to generate weights for both actor and critic
    networks, allowing for more complex agent specialization than linear hypernetworks.
    """
    output_dims: List[Tuple[int, int]]  # List of (input_dim, output_dim) for each layer
    hypernet_type: HyperNetType        # Type of hypernetwork (ACTOR or CRITIC)
    init_scale: float = np.sqrt(2)     # Default initialization scale for ReLU
    use_bias: bool = True              # Whether to use bias in the hypernetwork
    hidden_dims: List[int] = (64,)     # Hidden layer sizes for the MLP
    
    @staticmethod
    def hypernet_init(gain, fan_in, fan_out):
        """
        Custom initialization function for the hypernetwork weights.
        Uses orthogonal initialization with a specified gain.
        
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
            
            batch_size = shape[0]
            keys = jax.random.split(key, num=batch_size)
        
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
            
            # MLP for biases generation
            bias_mlp = x
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
    

class ScannedRNN(nn.Module):
    """
    RNN module that can be scanned over a sequence.
    
    Implements a GRU cell that maintains hidden state across sequence steps,
    with support for episode boundaries (resets).
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
        Process a single timestep with the RNN.
        
        Args:
            carry: Previous hidden state
            x: Tuple of (inputs, resets)
            
        Returns:
            New hidden state and output
        """
        rnn_state = carry
        ins, resets = x
        # Reset hidden state when episodes terminate
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """
        Initialize RNN hidden state.
        
        Args:
            batch_size: Number of parallel sequences
            hidden_size: Size of hidden state
            
        Returns:
            Initial hidden state
        """
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    """
    Actor-Critic network with recurrent layers and hypernetworks.
    
    Combines GRU units for temporal dependencies with hypernetworks to
    generate agent-specific parameters from agent embeddings.
    """
    action_dim: Sequence[int]          # Action space dimension
    config: Dict                       # Configuration dictionary
    num_agents: int                    # Number of agents
    observation_dim: int               # Observation dimension (not used)

    def setup(self):
        """Initialize model components."""
        # Extract hypernetwork configuration
        self.embedding_dim = self.config.get("HYPERNET_EMBEDDING_DIM", 64)
        self.init_scale = self.config.get("HYPERNET_INIT_SCALE", np.sqrt(2))
        self.use_agent_id_embeddings = self.config.get("USE_AGENT_ID_EMBEDDINGS", True)
        self.use_bias_in_hypernet = self.config.get("USE_BIAS_IN_HYPERNET", True)
        self.activation_fn = jax.nn.relu
        
        # Initialize agent embeddings - either learned or one-hot
        self.agent_embeddings = self.param(
            "agent_embeddings",
            nn.initializers.orthogonal(self.init_scale),
            (self.num_agents, self.embedding_dim),
        ) if self.use_agent_id_embeddings else jnp.eye(self.num_agents)
        
        # Set up network dimensions
        hidden_dim = self.config["GRU_HIDDEN_DIM"]
        fc_dim = self.config["FC_DIM_SIZE"]
        
        # Actor network architecture
        self.actor_output_dims = [
            (fc_dim, hidden_dim),       # After RNN
            (hidden_dim, self.action_dim)  # Final actor output
        ]

        # Configure hypernetworks
        hyper_hidden_dims = self.config.get("HYPERNET_HIDDEN_DIMS", (64,))
        self.actor_hypernet = MLPHyperNetwork(
            output_dims=self.actor_output_dims,
            init_scale=self.init_scale, 
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.ACTOR,
            hidden_dims=hyper_hidden_dims
        )
        
        # Critic network architecture
        self.critic_output_dims = [
            (fc_dim, hidden_dim),       # After RNN
            (hidden_dim, 1)             # Final critic output
        ]
        
        self.critic_hypernet = MLPHyperNetwork(
            output_dims=self.critic_output_dims,
            init_scale=self.init_scale,
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.CRITIC,
            hidden_dims=hyper_hidden_dims
        )

    @nn.compact
    def __call__(self, hidden, x):
        """
        Forward pass through the actor-critic network.
        
        Args:
            hidden: RNN hidden state
            x: Tuple of (observations, dones, available_actions)
            
        Returns:
            new_hidden: Updated RNN hidden state
            pi: Action probability distribution
            value: Value function estimate
        """
        obs, dones, avail_actions = x
        
        # Extract agent IDs from observations
        one_hot = obs[..., -self.num_agents:]
        agent_id = jnp.argmax(one_hot, axis=-1)
        obs = obs[..., :-self.num_agents]  # Remove agent ID from observation
        
        # Initial observation embedding
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], 
            kernel_init=nn.initializers.orthogonal(np.sqrt(2)), 
            bias_init=nn.initializers.constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # Process sequence with RNN
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Generate actor weights using hypernetwork
        actor_weights, actor_biases = self.actor_hypernet(self.agent_embeddings)
        critic_weights, critic_biases = self.critic_hypernet(self.agent_embeddings)
        
        def apply_hypernet(x, weights, biases):
            """
            Apply hypernetwork-generated weights to an input.
            
            Args:
                x: Input features
                weights: List of weight matrices
                biases: List of bias vectors
                
            Returns:
                Network output
            """
            for w, b in zip(weights[:-1], biases[:-1]):
                x = self.activation_fn(jnp.matmul(x, w.reshape(x.shape[-1], -1)) + b)
            # Final layer without activation
            return jnp.matmul(x, weights[-1].reshape(x.shape[-1], -1)) + biases[-1] 

        # Apply hypernetworks to generate actor and critic outputs for all agents
        vmap_apply_hypernet = jax.vmap(apply_hypernet, in_axes=(None, 0, 0))
        all_actor_out = vmap_apply_hypernet(embedding, actor_weights, actor_biases)
        
        # Select outputs for the specific agents in the batch
        actor_out = all_actor_out[agent_id, jnp.arange(embedding.shape[0])[:, None], jnp.arange(embedding.shape[1])]

        # Mask unavailable actions with large negative values
        unavail_actions = 1 - avail_actions
        action_logits = actor_out - (unavail_actions * 1e10)

        # Create categorical distribution for discrete actions
        pi = distrax.Categorical(logits=action_logits)

        # Apply critic hypernetwork
        all_critic_out = vmap_apply_hypernet(embedding, critic_weights, critic_biases)
        critic = all_critic_out[agent_id, jnp.arange(embedding.shape[0])[:, None], jnp.arange(all_critic_out.shape[2])]

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """Stores a single step transition for training."""
    global_done: jnp.ndarray     # Done flag for all agents
    done: jnp.ndarray            # Done flags per agent
    action: jnp.ndarray          # Actions taken
    value: jnp.ndarray           # Value estimates
    reward: jnp.ndarray          # Rewards received
    log_prob: jnp.ndarray        # Log probabilities of actions
    obs: jnp.ndarray             # Observations
    info: jnp.ndarray            # Additional info
    avail_actions: jnp.ndarray   # Available actions mask


class CustomTrainState(TrainState):
    """Extended TrainState with update counter."""
    n_updates: int = 0


def batchify(x: dict, agent_list, num_actors):
    """
    Convert a dictionary of agent observations to a batched array.
    
    Args:
        x: Dictionary mapping agent IDs to observations
        agent_list: List of agent IDs
        num_actors: Total number of actors
        
    Returns:
        Batched array of observations
    """
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """
    Convert a batched array to a dictionary of agent-specific arrays.
    
    Args:
        x: Batched array
        agent_list: List of agent IDs
        num_envs: Number of environments
        num_actors: Total number of actors
        
    Returns:
        Dictionary mapping agent IDs to their arrays
    """
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    """
    Construct the training function based on configuration.
    
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
    env = SMAXAppendAgentID(env, True)
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
        
        # Initialize network
        observation_size = env.observation_space(env.agents[0]).shape[0]
        true_obs_size = observation_size - env.num_agents
        network = ActorCriticRNN(
            env.action_space(env.agents[0]).n, 
            config=config, 
            num_agents=env.num_agents, 
            observation_dim=true_obs_size
        )
        
        # Initialize network parameters
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], observation_size)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
        
         
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
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        
        # Evaluation function
        def run_eval(rng, train_state):
            """
            Run evaluation with current policy.
            
            Args:
                rng: Random key
                train_state: Current training state
                
            Returns:
                Dictionary of evaluation metrics
            """
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            
            params = train_state.params
            def _eval_step(step_state, unused):
                """Perform a single evaluation step."""
                params, env_state, last_obs, last_done, hstate, rng = step_state
                
                # Select actions
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
                hstate, pi, value = network.apply(params, hstate, ac_in)
                action = pi.mode()  # Deterministic actions for evaluation
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
                infos = jax.tree_map(lambda x: x.reshape((num_actors_eval)), info)
                rewards = batchify(reward, env.agents, num_actors_eval).squeeze()
                done_batch = batchify(done, env.agents, num_actors_eval).squeeze()
                step_state = (params, env_state, obsv, done_batch, hstate, rng)
                return step_state, (rewards, done_batch, infos)

            # Initialize evaluation environment
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            
            # Create test environment
            test_env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
            test_env = SMAXAppendAgentID(test_env, True)
            test_env = SMAXLogWrapper(test_env) 
            
            init_obs, env_state = jax.vmap(test_env.reset, in_axes=0)(keys)
            
            num_eval_actors = config["TEST_NUM_ENVS"] * env.num_agents
            init_dones = jnp.zeros((num_eval_actors), dtype=bool)
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                num_eval_actors, config["GRU_HIDDEN_DIM"]
            )
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            
            # Run evaluation steps
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _eval_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            
            # Calculate metrics, filtering by completed episodes
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
        
        # Training update step
        def _update_step(update_runner_state, unused):
            """
            Perform a single training update step.
            
            Args:
                update_runner_state: Current runner state and update counter
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated runner state and metrics
            """
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                """
                Perform a single environment step and collect transition data.
                
                Args:
                    runner_state: Current runner state
                    unused: Unused parameter for JAX compatibility
                    
                Returns:
                    Updated runner state and transition data
                """
                train_state, env_state, last_obs, last_done, hstate, rng, test_state = runner_state

                # SELECT ACTION
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
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
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
                    info,
                    avail_actions,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, test_state)
                return runner_state, transition

            # Collect trajectory over multiple steps
            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE using GAE
            train_state, env_state, last_obs, last_done, hstate, rng, test_state = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                avail_actions,
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                """
                Calculate Generalized Advantage Estimation.
                
                Args:
                    traj_batch: Batch of trajectory data
                    last_val: Final value estimates
                    
                Returns:
                    advantages: Advantage estimates
                    targets: Value targets
                """
                def _get_advantages(gae_and_next_value, transition):
                    """Calculate GAE for a single timestep."""
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

            # UPDATE NETWORK with PPO
            def _update_epoch(update_state, unused):
                """
                Perform a single PPO epoch.
                
                Args:
                    update_state: Current update state
                    unused: Unused parameter for JAX compatibility
                    
                Returns:
                    Updated state and loss info
                """
                def _update_minbatch(train_state, batch_info):
                    """Update on a single minibatch."""
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        """PPO loss function."""
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS with clipping
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS with clipping
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        # Normalize advantages
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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

                        # Calculate diagnostic metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        # Combined loss
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    # Calculate gradients and update parameters
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                # Unpack update state
                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    test_state
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # Reshape hidden state for minibatching
                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                
                # Shuffle data for minibatching
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

                # Update on all minibatches
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    test_state
                )
                return update_state, total_loss

            # Run multiple epochs of updates
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
                test_state
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # Process metrics
            metric = traj_batch.info
            metric = jax.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            
            # Collect loss info
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            
            rng = update_state[-2]

            # Callback for logging
            def callback(metric,original_seed):
                """Log metrics to wandb."""
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

            # Increment update counter
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metric["update_steps"] = update_steps
            
            # Run periodic evaluation
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: run_eval(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metric.update({"test_" + k: v for k, v in test_state.items()})
                
            # Log metrics
            jax.debug.callback(callback, metric, original_seed)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, test_state)
            return (runner_state, update_steps), metric

        # Initial evaluation
        rng, _rng = jax.random.split(rng)
        test_state = run_eval(_rng, train_state)
        
        # Initialize runner state
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
            test_state
        )
        
        # Run training
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_smax")
def main(config):
    """
    Main entry point for training.
    
    Args:
        config: Hydra configuration
    """
    config = OmegaConf.to_container(config)
    from datetime import datetime
    now = datetime.now()
    name = f"ippo_rnn_shared_smax_mlp_hypernets_{config['MAP_NAME']}_{now:%Y-%m-%d_%H-%M-%S}"
    tags = ["IPPO", "RNN","MLP","Hypernets"] if config.get("EXP_TAGS") is None else config.get("EXP_TAGS")
    
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
    
    # Set up RNG
    rng = jax.random.PRNGKey(config["SEED"])
    
    # Run training (single seed or multiple seeds)
    if config.get("NUM_SEEDS") is None:
        train_jit = jax.jit(make_train(config), device=jax.devices()[0])
        out = train_jit(rng)
    else:
        rngs = jax.random.split(rng, config["NUM_SEEDS"])    
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)
        
        # Log results for all seeds
        log_experiment_results(config, out, axis=(0, 2, 3, 4))
        
    # Save the model if orbax is available
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

        # Extract final parameters
        params = out['runner_state'][0][0].params
        save_dir = f"{wandb.run.dir}/models/{run.name}"  # type: ignore
        path = f'{save_dir}/final'
        
        # Save checkpoint
        save_args = orbax_utils.save_args_from_target(params)
        checkpointers.save(
            path, params, save_args=save_args
        )
        print(f"model saved to {save_dir} {path}")

        # Upload to wandb as an artifact   
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