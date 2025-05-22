
import os
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from supersuit.vector.concat_vec_env import ConcatVecEnv
from supersuit.vector.constructors import call_wrap
from supersuit.vector.multiproc_vec import ProcConcatVec
from supersuit.vector.vector_constructors import vec_env_args
import gym


def MakeCPUAsyncConstructor(max_num_cpus):
    """
    Create a constructor for multi-CPU vectorized environments.
    
    This function creates an appropriate environment constructor based on
    the number of CPUs requested, enabling efficient parallel execution.
    
    Args:
        max_num_cpus: Maximum number of CPUs to use for environment processes
        
    Returns:
        Environment constructor function
    """
    if max_num_cpus == 0 or max_num_cpus == 1:
        return ConcatVecEnv
    else:
        def constructor(env_fn_list, obs_space, act_space):
            # Ensure environments don't use GPU
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            # Get example environment to determine properties
            example_env = env_fn_list[0]()
            envs_per_env = getattr(example_env, "num_envs", 1)

            # Distribute environments across CPUs
            num_fns = len(env_fn_list)
            envs_per_cpu = (num_fns + max_num_cpus - 1) // max_num_cpus

            # Split environment functions across CPUs
            env_cpu_div = []
            num_envs_alloced = 0
            while num_envs_alloced < num_fns:
                start_idx = num_envs_alloced
                end_idx = min(num_fns, start_idx + envs_per_cpu)
                env_cpu_div.append(env_fn_list[start_idx:end_idx])
                num_envs_alloced = end_idx

            # Create vectorized environment
            cat_env_fns = [call_wrap(ConcatVecEnv, env_fns) for env_fns in env_cpu_div]
            return ProcConcatVec(
                cat_env_fns,
                obs_space,
                act_space,
                num_fns * envs_per_env,
                example_env.metadata,
            )

        return constructor


def convert_one_hot_to_ids(one_hot_encoded_ids):
    """
    Convert one-hot encoded agent IDs to integer IDs.
    
    Args:
        one_hot_encoded_ids: One-hot encoded agent identifiers
        
    Returns:
        Integer agent IDs
    """
    return jnp.argmax(one_hot_encoded_ids, axis=-1)


def change_agent_id(
    num_agents,
    agent_id_size,
    key,
    one_hot_encode=False,
    orthogonal_encode=False,
    random_uniform_encode=False,
    int_id_encode=False,
):
    """
    Create agent identifiers using different encoding methods.
    
    This function supports various ways to create agent identifiers,
    which can significantly affect learning in parameter-sharing architectures.
    
    Args:
        num_agents: Number of agents
        agent_id_size: Size of the agent ID vector
        key: JAX random key
        one_hot_encode: Use one-hot encoding
        orthogonal_encode: Use orthogonal vectors
        random_uniform_encode: Use random uniform vectors
        int_id_encode: Use integer IDs with padding
        
    Returns:
        Agent ID vectors
    """
    if one_hot_encode:
        ret_agent_id = jax.nn.one_hot(jnp.arange(num_agents), agent_id_size)
    elif orthogonal_encode:
        ret_agent_id = jax.random.orthogonal(key, agent_id_size)[:num_agents]
    elif random_uniform_encode:
        ret_agent_id = jax.random.normal(key, (num_agents, agent_id_size))
    elif int_id_encode:
        arr = jnp.expand_dims(jnp.arange(num_agents), axis=-1)
        # Pad with zeros to reach required size
        padding_size = agent_id_size - 1
        ret_agent_id = jnp.pad(
            arr, ((0, 0), (0, padding_size)), mode="constant", constant_values=0
        )
    return ret_agent_id


class EncodedAgentIDWrapper:
    """
    Wrapper that adds encoded agent IDs to observations.
    
    This wrapper appends agent identifiers to each agent's observations,
    enabling parameter-sharing architectures to distinguish between agents.
    
    Attributes:
        env: The environment to wrap
        agents: List of agent IDs
        num_agents: Number of agents
        agent_id_size: Size of agent ID vectors
        encode_agent_id: Method to encode agent IDs
        one_hot_encode_agent_id: Whether to use one-hot encoding
    """
    def __init__(
        self, env, agent_id_size, encode_agent_id=None, one_hot_encode_agent_id=False
    ):
        self.env = env
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.agent_id_size = agent_id_size
        self.encode_agent_id = encode_agent_id
        self.one_hot_encode_agent_id = one_hot_encode_agent_id
        self.action_space = env.action_space
        self.observation_space = self._create_observation_space(env.observation_space)
        self.encoded_ids = self._create_encoded_ids(agent_id_size, encode_agent_id)

    def _create_observation_space(self, observation_space):
        """Create modified observation space that includes agent IDs."""
        from gym.spaces import Box

        new_spaces = []
        for i in range(self.num_agents):
            low = np.concatenate(
                [observation_space[i].low, np.zeros(self.agent_id_size)]
            )
            high = np.concatenate(
                [observation_space[i].high, np.ones(self.agent_id_size)]
            )
            new_spaces.append(Box(low=low, high=high))
        return tuple(new_spaces)

    def _create_encoded_ids(self, agent_id_size, encode_agent_id):
        """Create encoded agent IDs using the specified method."""
        if encode_agent_id:
            key = jax.random.PRNGKey(42)
            return change_agent_id(
                self.num_agents,
                agent_id_size,
                key,
                one_hot_encode=(encode_agent_id == "one_hot"),
                orthogonal_encode=(encode_agent_id == "orthogonal"),
                random_uniform_encode=(encode_agent_id == "random_uniform"),
                int_id_encode=(encode_agent_id == "int_id"),
            )
        return None

    def _convert_actions_ids(self, observations):
        """Convert one-hot agent IDs to the specified encoding."""
        if not self.encode_agent_id:
            return observations

        x_obs = observations[..., : -self.num_agents]
        agent_ids_int = convert_one_hot_to_ids(observations[..., -self.num_agents :])
        updated_ids = self.encoded_ids[agent_ids_int]
        return np.concatenate([x_obs, updated_ids], axis=-1)

    def step(self, actions):
        """
        Take one step in the environment.
        
        Args:
            actions: Agent actions
            
        Returns:
            Tuple of (observations, rewards, dones, truncations, info)
        """
        observations, rewards, agent_dones, agent_truncs, infos = self.env.step(actions)
        
        # Add agent ID to observations if needed
        if self.one_hot_encode_agent_id:
            observations = self.concat_agent_id(observations, num_envs=self.env.num_envs, num_agents=self.num_agents)

        observations = self._convert_actions_ids(observations)
        return observations, rewards, agent_dones, agent_truncs, infos

    @staticmethod
    def concat_agent_id(observations, num_envs, num_agents):
        """Concatenate one-hot agent IDs to observations."""
        # Create one-hot encoded agent IDs
        agent_ids = np.eye(num_agents)
            
        # Repeat agent IDs for each environment
        agent_ids = np.tile(agent_ids, (num_envs, 1))
            
        # Concatenate agent IDs to observations
        observations = np.concatenate([observations, agent_ids], axis=1)
        return observations

    def reset(self, *args, **kwargs):
        """
        Reset the environment.
        
        Args:
            *args: Arguments to pass to the environment reset
            **kwargs: Keyword arguments to pass to the environment reset
            
        Returns:
            Tuple of (observations, info)
        """
        observations, info = self.env.reset(*args, **kwargs)

        if self.one_hot_encode_agent_id:
            observations = self.concat_agent_id(observations, num_envs=self.env.num_envs, num_agents=self.num_agents)

        observations = self._convert_actions_ids(observations)
        return observations, info

    def close(self):
        """Close the environment."""
        return self.env.close()

    def render(self, *args, **kwargs):
        """Render the environment."""
        return self.env.render(*args, **kwargs)


class VmasNumpyVecEnv:
    """
    Wrapper for VMAS environments to provide compatibility with NumPy.
    
    Handles conversion between PyTorch tensors (used by VMAS) and NumPy arrays
    (used by most RL frameworks).
    
    Attributes:
        env: VMAS environment
        agents: List of agent IDs
        num_agents: Number of agents
        num_envs: Number of parallel environments
        auto_reset: Whether to automatically reset completed environments
    """
    def __init__(self, env, auto_reset=True):
        global torch
        import torch

        self.env = env
        self.agents = env.agents
        self.num_agents = len(self.agents)
        self.num_envs = env.num_envs
        self.metadata = env.metadata
        self.render = env.render
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.auto_reset = auto_reset

        # Pre-allocate arrays to avoid memory reallocation
        self.obs_dim = self.observation_space[0].shape[-1]
        self.obs_array = np.zeros((self.num_agents, self.num_envs, self.obs_dim))
        self.reward_array = np.zeros((self.num_agents, self.num_envs))
        self.agent_dones = np.zeros(self.num_envs * self.num_agents, dtype=bool)
        self.agent_truncs = np.zeros(self.num_envs * self.num_agents, dtype=bool)

        # Pre-allocate tensors for actions
        self.actions_tensor = [
            torch.zeros(
                (self.num_envs, self.env.get_agent_action_size(agent)),
                device="cpu",
                dtype=torch.float32,
            )
            for agent in self.agents
        ]

    def step(self, actions):
        """
        Take one step in the environment.
        
        Handles conversion between NumPy and PyTorch for actions and observations.
        
        Args:
            actions: Agent actions as NumPy arrays
            
        Returns:
            Tuple of (observations, rewards, dones, truncations, info)
        """
        # Convert JAX/NumPy actions to PyTorch
        actions = np.array(actions)
        actions = actions.reshape(self.num_envs, self.num_agents, -1)

        # Convert actions to the format expected by VMAS
        for i in range(self.num_agents):
            self.actions_tensor[i].copy_(torch.from_numpy(actions[:, i]))

        obs, rewards, dones, infos = self.env.step(self.actions_tensor)

        # Use pre-allocated arrays for observations and rewards
        np.stack(obs, out=self.obs_array)
        np.stack(rewards, out=self.reward_array)

        # Reshape observations to match expected format
        observations = self.obs_array.transpose(1, 0, 2).reshape(-1, self.obs_dim)
        rewards = self.reward_array.T.ravel()

        # Convert environment dones to agent dones
        self.agent_dones = np.array(np.repeat(dones, self.num_agents))
        truncs = self.env.steps >= self.env.max_steps
        self.agent_truncs = np.array(np.repeat(truncs, self.num_agents))

        # Auto-reset completed environments if configured
        if self.auto_reset:
            should_reset = np.logical_or(dones, truncs)
            if should_reset.any():
                reset_indices = np.where(should_reset)[0]
                for env_index in reset_indices:
                    terminal_obs = self.env.reset_at(env_index)

        return observations, rewards, self.agent_dones, self.agent_truncs, infos

    def reset(self, *args, **kwargs):
        """
        Reset the environment.
        
        Args:
            *args: Arguments to pass to the environment reset
            **kwargs: Keyword arguments to pass to the environment reset
            
        Returns:
            Tuple of (observations, info)
        """
        obs, info = self.env.reset(*args, **kwargs, return_info=True)

        # Convert observations to NumPy arrays
        np.stack(obs, out=self.obs_array)
        observations = self.obs_array.transpose(1, 0, 2).reshape(-1, self.obs_dim)

        return observations, info

    def close(self):
        """Close the environment."""
        pass

    def render(self, mode="rgb_array", agent_index_focus=None):
        """Render the environment."""
        return self.env.render(mode=mode, agent_index_focus=agent_index_focus)


class ClipAction:
    """
    Wrapper that clips action values to specified range.
    
    Useful for continuous action spaces to ensure actions stay within bounds.
    
    Attributes:
        env: The environment to wrap
        low: Lower bound for actions
        high: Upper bound for actions
    """
    def __init__(self, env, low=-1.0, high=1.0):
        self.env = env
        self.low = low
        self.high = high

    def step(self, actions):
        """Clip actions and step the environment."""
        clipped_actions = np.clip(actions, self.low, self.high)
        return self.env.step(clipped_actions)

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)


@dataclass
class NormalizationState:
    """
    Tracks running statistics for observation normalization.
    
    Attributes:
        mean: Running mean of observations
        var: Running variance of observations
        count: Number of samples processed
    """
    mean: np.ndarray
    var: np.ndarray
    count: float


class NormalizeVecObservation:
    """
    Normalizes observations using running statistics.
    
    Keeps track of observation mean and variance and normalizes
    observations to have approximately zero mean and unit variance.
    
    Attributes:
        env: The environment to wrap
        norm_state: Normalization statistics
    """
    def __init__(self, env):
        self.env = env
        self.norm_state = None

    def reset(self, *args, **kwargs):
        """Reset environment and initialize or update normalization state."""
        obs, info = self.env.reset(*args, **kwargs)
        if self.norm_state is None:
            self.norm_state = NormalizationState(
                mean=np.zeros_like(obs),
                var=np.ones_like(obs),
                count=1e-4
            )
        self._update_normalization(obs)
        return self._normalize_obs(obs), info

    def step(self, action):
        """Take step and normalize new observations."""
        obs, reward, done, trunc, info = self.env.step(action)
        self._update_normalization(obs)
        return self._normalize_obs(obs), reward, done, trunc, info

    def _update_normalization(self, obs):
        """Update running statistics with new observations."""
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]
        
        # Update running mean
        delta = batch_mean - self.norm_state.mean
        tot_count = self.norm_state.count + batch_count
        new_mean = self.norm_state.mean + delta * batch_count / tot_count
        
        # Update running variance using Welford's algorithm
        m_a = self.norm_state.var * self.norm_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.norm_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        # Update normalization state
        self.norm_state = NormalizationState(
            mean=new_mean,
            var=new_var,
            count=tot_count
        )

    def _normalize_obs(self, obs):
        """Normalize observations with current statistics."""
        return (obs - self.norm_state.mean) / np.sqrt(self.norm_state.var + 1e-8)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)


@dataclass
class NormalizationStateReward:
    """
    Tracks running statistics for reward normalization.
    
    Attributes:
        mean: Running mean of returns
        var: Running variance of returns
        count: Number of samples processed
        return_val: Current return values
    """
    mean: float
    var: float
    count: float
    return_val: np.ndarray


class NormalizeVecReward:
    """
    Normalizes rewards using running statistics.
    
    Keeps track of discounted returns and normalizes rewards
    to have approximately unit variance.
    
    Attributes:
        env: The environment to wrap
        gamma: Discount factor for returns
        norm_state: Normalization statistics
    """
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.norm_state = None

    def reset(self, *args, **kwargs):
        """Reset environment and initialize or update normalization state."""
        obs, info = self.env.reset(*args, **kwargs)
        batch_count = obs.shape[0]
        if self.norm_state is None:
            self.norm_state = NormalizationStateReward(
                mean=0.0,
                var=1.0,
                count=1e-4,
                return_val=np.zeros((batch_count,))
            )
        return obs, info

    def step(self, action):
        """Take step and normalize rewards."""
        obs, reward, done, trunc, info = self.env.step(action)
        
        # Update return tracking
        self.norm_state.return_val = self.norm_state.return_val * self.gamma * (1 - done) + reward

        # Update normalization statistics
        batch_mean = np.mean(self.norm_state.return_val)
        batch_var = np.var(self.norm_state.return_val)
        batch_count = obs.shape[0]
        
        # Update running mean
        delta = batch_mean - self.norm_state.mean
        tot_count = self.norm_state.count + batch_count
        new_mean = self.norm_state.mean + delta * batch_count / tot_count
        
        # Update running variance using Welford's algorithm
        m_a = self.norm_state.var * self.norm_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.norm_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        # Update normalization state
        self.norm_state = NormalizationStateReward(
            mean=new_mean,
            var=new_var,
            count=tot_count,
            return_val=self.norm_state.return_val
        )

        # Normalize the reward
        normalized_reward = reward / np.sqrt(self.norm_state.var + 1e-8)

        return obs, normalized_reward, done, trunc, info

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)


def concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0, base_class="gymnasium"):
    """
    Create a vectorized environment with multiple copies of an environment.
    
    Args:
        vec_env: Environment constructor
        num_vec_envs: Number of environments to create
        num_cpus: Number of CPUs to use
        base_class: Base class for environments ("gymnasium" or "gym")
        
    Returns:
        Vectorized environment
    """
    num_cpus = min(num_cpus, num_vec_envs)
    print("num_cpus", num_cpus)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))
    return vec_env


def make_env(
    env_name: str,
    one_hot_encode_agent_id: bool = False,
    num_envs: int = 1,
    num_cpus: Optional[int] = None,
    capture_video: bool = False,
    video_location: Optional[str] = None,
    int_ids: bool = False,
    **kwargs: Any,
) -> Tuple[Any, list[str], int, int, int]:
    """
    Create a multi-agent environment with consistent interface.
    
    This function handles the creation of environments from different frameworks
    (PettingZoo, VMAS, RWare) with a unified interface suitable for MARL algorithms.
    
    Args:
        env_name: Name of the environment
        one_hot_encode_agent_id: Whether to append one-hot agent IDs to observations
        num_envs: Number of parallel environments
        num_cpus: Number of CPUs to use for parallelization
        capture_video: Whether to capture videos during evaluation
        video_location: Directory to save videos
        int_ids: Whether to use integer IDs instead of one-hot encoding
        **kwargs: Additional environment-specific arguments
        
    Returns:
        Tuple of (env, agents, action_dim, num_actions, observation_size)
    """
    if num_cpus is None:
        num_cpus = num_envs

    # Different environment frameworks have different APIs
    # Handle each environment type separately
    
    if env_name == "simple_spread_v3":
        # PettingZoo MPE environments
        import supersuit as ss
        from pettingzoo.mpe import simple_spread_v3

        # Keys specific to PettingZoo environments
        pz_env_keys = [
            "N",
            "local_ratio",
            "max_cycles",
            "continuous_actions",
            "render_mode",
        ]
        env_kwargs = {k: v for k, v in kwargs.items() if k in pz_env_keys}

        # Special case for customized MPE variants
        if (kwargs.get("see_other_agents")) is not None or (
            kwargs.get("agents_collide")
        ) is not None:
            # Import necessary components for custom MPE environment
            from gymnasium.utils import EzPickle
            import numpy as np
            from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
            from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
            from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
            from pettingzoo.utils.conversions import parallel_wrapper_fn
            from pettingzoo.mpe.simple_spread.simple_spread import Scenario

            see_other_agents = kwargs.get("see_other_agents", True)
            agents_collide = kwargs.get("agents_collide", True)

            print("see_other_agents", see_other_agents)
            print("agents_collide", agents_collide)

            # Create custom scenario with configurable agent visibility and collision
            class CustomScenario(Scenario):
                def make_world(self, N=3):
                    world = World()
                    # set any world properties first
                    world.dim_c = 2
                    num_agents = N
                    num_landmarks = N
                    world.collaborative = True
                    # add agents
                    world.agents = [Agent() for i in range(num_agents)]
                    for i, agent in enumerate(world.agents):
                        agent.name = f"agent_{i}"
                        agent.collide = agents_collide
                        agent.silent = True
                        agent.size = 0.15
                    # add landmarks
                    world.landmarks = [Landmark() for i in range(num_landmarks)]
                    for i, landmark in enumerate(world.landmarks):
                        landmark.name = "landmark %d" % i
                        landmark.collide = False
                        landmark.movable = False
                    return world

                def observation(self, agent, world):
                    # get positions of all entities in this agent's reference frame
                    entity_pos = []
                    for entity in world.landmarks:  # world.entities:
                        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                    # communication of all other agents
                    comm = []
                    other_pos = []
                    for other in world.agents:
                        if other is agent:
                            continue
                        comm.append(other.state.c)
                        other_pos.append(other.state.p_pos - agent.state.p_pos)
                    if see_other_agents:
                        return np.concatenate(
                            [agent.state.p_vel]
                            + [agent.state.p_pos]
                            + entity_pos
                            + other_pos
                            + comm
                        )
                    else:
                        return np.concatenate(
                            [agent.state.p_vel]
                            + [agent.state.p_pos]
                            + entity_pos
                            + comm
                        )

            # Create custom environment class
            class raw_env_v2(SimpleEnv, EzPickle):
                def __init__(
                    self,
                    N=3,
                    local_ratio=0.5,
                    max_cycles=25,
                    continuous_actions=False,
                    render_mode=None,
                ):
                    EzPickle.__init__(
                        self,
                        N=N,
                        local_ratio=local_ratio,
                        max_cycles=max_cycles,
                        continuous_actions=continuous_actions,
                        render_mode=render_mode,
                    )
                    assert (
                        0.0 <= local_ratio <= 1.0
                    ), "local_ratio is a proportion. Must be between 0 and 1."
                    scenario = CustomScenario()
                    world = scenario.make_world(N)
                    SimpleEnv.__init__(
                        self,
                        scenario=scenario,
                        world=world,
                        render_mode=render_mode,
                        max_cycles=max_cycles,
                        continuous_actions=continuous_actions,
                        local_ratio=local_ratio,
                    )
                    self.metadata["name"] = "simple_spread_v3"

            env = make_env(raw_env_v2)
            parallel_env = parallel_wrapper_fn(env)
            env = parallel_env(**env_kwargs)
        else:
            # Use standard simple_spread environment
            env = simple_spread_v3.parallel_env(**env_kwargs)

        # Apply custom appearance if specified
        if (
            kwargs.get("agent_colours")
            or kwargs.get("landmark_colours")
            or kwargs.get("deterministic_agent_pos")
        ):
            update_reset(
                env,
                kwargs.get("agent_colours"),
                kwargs.get("landmark_colours"),
                kwargs.get("deterministic_agent_pos"),
            )

        # Get agent IDs
        possible_agents = env.possible_agents
        
        # Apply agent ID encoding if requested
        if one_hot_encode_agent_id:
            print("one hot encoding")
            from supersuit import agent_indicator_v0
            env = agent_indicator_v0(env, type_only=False)

        # Convert to vectorized environment
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env.render_mode = "rgb_array"
        
        # Create multiple parallel environments if requested
        if num_envs > 1:
            env = concat_vec_envs_v1(
                env, num_envs, num_cpus=num_cpus, base_class="gymnasium"
            )

        # Set common attributes
        env.single_observation_space = env.observation_space
        env.single_action_space = env.action_space
        env.is_vector_env = True
        
        # Apply video recording wrapper if requested
        if capture_video:
            import gymnasium
            env = gymnasium.wrappers.RecordVideo(
                env,
                video_location,
                episode_trigger=lambda x: x >= 0,
            )

        # Get environment dimensions
        num_actions = 1
        observation_size = env.observation_space.shape[0]
        env.agents = possible_agents
        action_dim = env.action_space.n

    elif env_name == "multiwalker_v9":
        # PettingZoo SISL environments
        import supersuit as ss
        from pettingzoo.sisl import multiwalker_v9

        # Create environment
        env = multiwalker_v9.parallel_env(**kwargs)
        env = ss.clip_actions_v0(env)
        
        # Get agent IDs
        possible_agents = env.possible_agents
        
        # Apply agent ID encoding if requested
        if one_hot_encode_agent_id:
            from supersuit import agent_indicator_v0
            env = agent_indicator_v0(env, type_only=False)

        # Convert to vectorized environment
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env.render_mode = "rgb_array"
        
        # Create multiple parallel environments if requested
        if num_envs > 1:
            env = concat_vec_envs_v1(
                env, num_envs, num_cpus=num_cpus, base_class="gymnasium"
            )

        # Set common attributes
        env.single_observation_space = env.observation_space
        env.single_action_space = env.action_space
        env.is_vector_env = True
        
        # Apply video recording wrapper if requested
        if capture_video:
            import gymnasium
            env = gymnasium.wrappers.RecordVideo(
                env,
                video_location,
                episode_trigger=lambda x: x >= 0,
            )

        # Get environment dimensions
        action_dim = env.action_space.shape[0]
        num_actions = action_dim
        observation_size = env.observation_space.shape[0]
        env.agents = possible_agents
        
    elif "rware" in env_name:
        # Robotic Warehouse environments
        import rware
        import gym
        import gymnasium.vector
        import numpy as np
        import supersuit as ss
        from gymnasium import spaces
        from gymnasium.vector.utils import concatenate, create_empty_array
        from pettingzoo.utils.env import ParallelEnv

        # Create custom wrapper for RWare environments
        class RwareVectorEnv(ParallelEnv):
            def __init__(self, env: Any) -> None:
                self.env = env
                n_agents = env.n_agents
                self.possible_agents = [f"agent_{i}" for i in range(n_agents)]

                self.metadata = env.metadata
                self.agents = self.possible_agents

                # Create observation and action spaces
                self.observation_spaces = spaces.Dict(
                    {
                        agent: _convert_space(env.observation_space[agent_index])
                        for agent_index, agent in enumerate(self.possible_agents)
                    }
                )
                self.action_spaces = spaces.Dict(
                    {
                        agent: _convert_space(env.action_space[agent_index])
                        for agent_index, agent in enumerate(self.possible_agents)
                    }
                )

                # Add int IDs to agents if requested
                if int_ids:
                    self.observation_spaces = spaces.Dict(
                        {
                            key: spaces.Box(
                                low=np.concatenate([space.low, [space.low[0]]], axis=0),
                                high=np.concatenate(
                                    [space.high, [space.high[0]]], axis=0
                                ),
                                shape=(space.shape[0] + 1,),
                                dtype=space.dtype,
                            )
                            for key, space in self.observation_spaces.items()
                        }
                    )
                self.num_envs = len(self.possible_agents)

                self._max_episode_steps = self.env.unwrapped.max_steps
                self._elapsed_steps = 0

            def concat_obs(self, obs_list: list[np.array]) -> np.ndarray:
                # Concatenate observations
                first_agent = self.possible_agents[0]
                obs_space = self.observation_space(first_agent)
                return concatenate(
                    obs_space,
                    obs_list,
                    create_empty_array(obs_space, self.num_envs),
                )

            def step_async(self, actions: np.ndarray) -> None:
                self._saved_actions = actions

            def step_wait(self) -> Any:
                return self.step(self._saved_actions)

            def reset(
                self, seed: Optional[int] = None, options: Any = None
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
                # Reset the environment
                self.env.seed(seed)
                _observations = self.env.reset()

                # Handle deterministic agent positioning if requested
                deterministic_agent_pos = kwargs.get("deterministic_agent_pos", None)
                if deterministic_agent_pos:
                    from rware.warehouse import Agent, Shelf, Direction

                    # Set up deterministic agent positions
                    available_pos = np.arange(
                        self.env.grid_size[0] * self.env.grid_size[1]
                    )
                    # Place agents in a row
                    manual_positions = [
                        available_pos[0],
                        available_pos[0] + 1,
                        available_pos[0] + 2,
                        available_pos[0] + 3,
                    ]
                    # Use positions for actual agents
                    agent_locs = manual_positions[: self.env.n_agents]
                    agent_locs = np.unravel_index(agent_locs, self.env.grid_size)
                    # Set direction
                    agent_dirs = [
                        Direction.DOWN,
                        Direction.DOWN,
                        Direction.DOWN,
                        Direction.DOWN,
                    ][: self.env.n_agents]
                    
                    # Reset agent counter and create agents
                    Agent.counter = 0
                    self.env.agents = [
                        Agent(x, y, dir_, self.env.msg_bits)
                        for y, x, dir_ in zip(*agent_locs, agent_dirs)
                    ]

                    # Update environment state
                    self.env.unwrapped.agents = self.env.agents
                    self.env.unwrapped._recalc_grid()
                    
                    # Setup request queue
                    self.env.request_queue = list(
                        np.random.choice(
                            self.env.shelfs,
                            size=self.env.request_queue_size,
                            replace=False,
                        )
                    )

                    # Get observations
                    _observations = tuple(
                        [
                            self.env.unwrapped._make_obs(agent)
                            for agent in self.env.agents
                        ]
                    )

                # Add integer IDs if requested
                if int_ids:
                    _observations = tuple(
                        np.hstack((arr, np.array([i])))
                        for i, arr in enumerate(_observations)
                    )

                # Format observations
                observations = self.concat_obs(_observations)
                infs: Dict[str, Any] = {}
                for agent in self.possible_agents:
                    infs[agent] = {}

                self._elapsed_steps = 0
                obs_dict = {}
                for agent_index, agent in enumerate(self.possible_agents):
                    obs_dict[agent] = observations[agent_index]
                return obs_dict, infs

            def step(self, actions: Dict[str, np.array]) -> Tuple[
                Dict[str, np.ndarray],
                Dict[str, float],
                Dict[str, bool],
                Dict[str, bool],
                Dict[str, np.ndarray],
            ]:
                # Take a step in the environment
                action_array = actions.values()
                observations, rewards, terms, infos = self.env.step(action_array)

                # Add integer IDs if requested
                if int_ids:
                    observations = tuple(
                        np.hstack((arr, np.array([i])))
                        for i, arr in enumerate(observations)
                    )

                env_done = np.all(terms)

                # Determine truncation
                if env_done and (self._elapsed_steps >= (self._max_episode_steps - 1)):
                    truns = [True] * len(terms)
                else:
                    truns = [False] * len(terms)

                # Format outputs
                obs_dict, rewards_dict, terms_dict, truns_dict, infos_dict = (
                    {},
                    {},
                    {},
                    {},
                    {},
                )
                for agent_index, agent in enumerate(self.possible_agents):
                    obs_dict[agent] = observations[agent_index]
                    rewards_dict[agent] = rewards[agent_index]
                    terms_dict[agent] = terms[agent_index]
                    truns_dict[agent] = truns[agent_index]

                    if infos and infos.get(agent):
                        infos_dict[agent] = infos[agent]
                    else:
                        infos_dict[agent] = {}

                self._elapsed_steps += 1

                return (
                    obs_dict,
                    rewards_dict,
                    terms_dict,
                    truns_dict,
                    infos_dict,
                )

            def render(self, mode: str = "rgb_array") -> Any:
                return self.env.render(mode)

            def close(self) -> Any:
                return self.env.close()

        # Create environment with global/individual rewards as specified
        if kwargs.get("reward_type") == "global":
            from rware.warehouse import RewardType
            env = gym.make(env_name, reward_type=RewardType.GLOBAL)
            print("correct!")
        else:
            env = gym.make(env_name)
            
        # Set agent colors if specified
        if kwargs.get("agent_colours"):
            change_colour_agents_rware(env, kwargs.get("agent_colours"))

        # Wrap environment
        env = RwareVectorEnv(env)
        possible_agents = env.possible_agents

        # Add agent IDs if requested
        if one_hot_encode_agent_id:
            from supersuit import agent_indicator_v0
            env = agent_indicator_v0(env, type_only=False)

        # Get environment dimensions
        action_dim = env.action_spaces["agent_0"].n
        num_actions = 1
        observation_size = env.observation_spaces["agent_0"].shape[0]
        env.agents = env.possible_agents

        # Set common attributes
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.render_mode = "rgb_array"
        env.unwrapped.render_mode = "rgb_array"

        # Convert to vectorized environment
        env = ss.pettingzoo_env_to_vec_env_v1(env)

        # Create multiple parallel environments if requested
        if num_envs > 1:
            env = concat_vec_envs_v1(
                env, num_envs, num_cpus=num_cpus, base_class="gymnasium"
            )
        env.single_observation_space = env.observation_space
        env.single_action_space = env.action_space
        env.is_vector_env = True

        # Apply video recording wrapper if requested
        if capture_video:
            import gymnasium
            env = gymnasium.wrappers.RecordVideo(
                env,
                video_location,
                episode_trigger=lambda x: x >= 0,
            )

        env.is_vector_env = True
        env.agents = possible_agents

    elif "vmas" in env_name:
        # VMAS environments
        from vmas import make_env as vmas_make_env
        import vmas

        # Extract scenario name from environment ID
        scenario = env_name.split("-")[-1]

        # Filter irrelevant kwargs
        keys_to_remove = ["one_hot_encode_agent_id"]
        env_kwargs = {k: v for k, v in kwargs.items() if k not in keys_to_remove}

        # Create VMAS environment
        env = vmas_make_env(
            scenario=scenario,
            num_envs=num_envs,
            device="cpu",
            **env_kwargs,
        )
        
        # Modify agent initial positions if requested
        if kwargs.get("change_agent_initial_pos_apart"):
            import torch
            
            def reset_world_at_change_agent_initial_pos(self, env_index: int = None):
                """Modified reset function to initialize agents at different positions."""
                for agent_index, agent in enumerate(self.world.agents):
                    # Initialize agents at random positions within range
                    agent.set_pos(
                        torch.zeros(
                            self.world.dim_p,
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.pos_range,
                            self.pos_range,
                        ),
                        batch_index=env_index,
                    )
                for landmark in self.world.landmarks:
                    landmark.set_pos(
                        torch.zeros(
                            (
                                (1, self.world.dim_p)
                                if env_index is not None
                                else (self.world.batch_dim, self.world.dim_p)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.pos_range,
                            self.pos_range,
                        ),
                        batch_index=env_index,
                    )
                    if env_index is None:
                        landmark.eaten = torch.full(
                            (self.world.batch_dim,), False, device=self.world.device
                        )
                        landmark.just_eaten = torch.full(
                            (self.world.batch_dim,), False, device=self.world.device
                        )
                        landmark.reset_render()
                    else:
                        landmark.eaten[env_index] = False
                        landmark.just_eaten[env_index] = False
                        landmark.is_rendering[env_index] = True

            # Monkey patch the reset function
            env.scenario.reset_world_at = types.MethodType(
                reset_world_at_change_agent_initial_pos, env.scenario
            )

        # Wrap environment for PyTorch/NumPy conversion
        env = VmasNumpyVecEnv(env, auto_reset=kwargs.get("auto_reset", False))

        # Add agent IDs if requested
        encode_agent_ids = kwargs.get("encode_agent_ids")
        if encode_agent_ids or one_hot_encode_agent_id:
            # Use one-hot encoding
            one_hot_encode_agent_id = True
            # Default is same size as num agents
            agent_id_size = kwargs.get("agent_id_size", env.num_agents)
            env = EncodedAgentIDWrapper(
                env,
                agent_id_size=agent_id_size,
                one_hot_encode_agent_id=one_hot_encode_agent_id,
                encode_agent_id=encode_agent_ids,
            )

        # Get agent IDs and environment dimensions
        possible_agents = env.agents
        observation_size = env.observation_space[0].shape[0]
        
        # Handle continuous vs. discrete actions
        if env_kwargs.get("continuous_actions") is True:
            action_dim = env.action_space[0].shape[0]
            num_actions = action_dim
            # Clip actions to valid range
            env = ClipAction(
                env, 
                low=env.action_space[0].low[0], 
                high=env.action_space[0].high[0]
            )
            # Apply normalization if requested
            if kwargs.get("NORMALIZE_ENV"):
                gamma = kwargs.get("GAMMA", 0.99)
                print(f"Normalizing env: {gamma}")
                env = NormalizeVecObservation(env)
                env = NormalizeVecReward(env, gamma=gamma)
        else:
            action_dim = env.action_space[0].n
            num_actions = 1

        # Setup video capture if requested
        if capture_video:
            import pyglet
            import os
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            # Configure pyglet for headless rendering
            pyglet.options["headless"] = True

            # Function to save video from frames
            def save_video(
                name: str, frame_list: List[np.array], fps: int, folder: str = "."
            ):
                # Ensure the folder exists
                os.makedirs(folder, exist_ok=True)

                # Construct the full path for the video file
                video_name = os.path.join(folder, name + ".mp4")

                # Get the shape of the first frame
                height, width = frame_list[0].shape[:2]

                # Create the video using MoviePy
                clip = ImageSequenceClip(frame_list, fps=fps)
                clip.write_videofile(
                    video_name,
                    fps=fps,
                    verbose=False,
                    logger=None,
                    ffmpeg_params=["-s", f"{width}x{height}"],
                )

            # Wrapper to capture frames for video recording
            class RenderWrapper:
                def __init__(self, env, save_render=True, visualize_render=False):
                    self.env = env
                    self.save_render = save_render
                    self.visualize_render = visualize_render
                    self.frame_list = []

                def step(self, actions):
                    obs, rewards, terms, truncs, infos = self.env.step(actions)

                    # Render if enabled
                    frame = self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=self.visualize_render,
                    )
                    if self.save_render:
                        self.frame_list.append(frame)

                    return obs, rewards, terms, truncs, infos

                def reset(self, *args, **kwargs):
                    return self.env.reset(*args, **kwargs)

                def render(self, mode="rgb_array", agent_index_focus=None):
                    return self.env.render(
                        mode=mode, agent_index_focus=agent_index_focus
                    )

                def close(self):
                    self.save_video(env_name, fps=30)
                    return self.env.close()

                def save_video(self, scenario_name, fps):
                    if self.save_render:
                        save_video(
                            scenario_name,
                            self.frame_list,
                            fps=fps,
                            folder=video_location,
                        )
                
                # Pass through attributes to the wrapped environment
                def __getattr__(self, name):
                    return getattr(self.env, name)

            # Apply render wrapper
            env = RenderWrapper(env)

    # Set common attributes
    env.num_agents = len(possible_agents)
    return env, possible_agents, action_dim, num_actions, observation_size


def update_reset(
    env: Any,
    agent_colours: Dict[str, list[int]],
    landmark_colours: Dict[str, list[int]],
    deterministic_agent_pos: bool = False,
) -> None:
    """
    Override the environment reset function to customize agent and landmark colors.
    
    Used for PettingZoo MPE environments to make agents visually distinguishable
    and to set deterministic positions if needed.
    
    Args:
        env: PettingZoo environment
        agent_colours: Dictionary mapping agent names to RGB colors
        landmark_colours: Dictionary mapping landmark names to RGB colors
        deterministic_agent_pos: Whether to use deterministic agent positions
    """
    def reset_world(world, np_random):
        # Set agent colors
        for i, agent in enumerate(world.agents):
            if agent_colours and agent_colours.get(agent.name):
                # PZ multiplies by 200 in rendering
                agent.color = np.array(agent_colours[agent.name]) / 200
            else:
                agent.color = np.array([0.35, 0.35, 0.85])

        # Set landmark colors
        for i, landmark in enumerate(world.landmarks):
            if landmark_colours and landmark_colours.get(landmark.name):
                landmark.color = np.array(landmark_colours[landmark.name]) / 200
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])

        # Set agent positions
        if deterministic_agent_pos:
            all_agent_pos = np_random.uniform(-1, +1, world.dim_p)

        for i, agent in enumerate(world.agents):
            if deterministic_agent_pos:
                # Position agents next to each other with fixed spacing
                agent.state.p_pos = np.clip(all_agent_pos + (i * 0.25), -1, 1)
            else:
                agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Set landmark positions
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # Override the reset_world function
    env.unwrapped.scenario.reset_world = reset_world


def _convert_space(space: gym.Space) -> gymnasium.Space:
    """
    Convert a gym space to a gymnasium space.
    
    Handles the conversion between different versions of spaces
    for compatibility with various environments.
    
    Args:
        space: gym space to convert
        
    Returns:
        Equivalent gymnasium space
    """
    from gymnasium.spaces import (
        Box,
        Dict,
        Discrete,
        Graph,
        MultiBinary,
        MultiDiscrete,
        Sequence,
        Text,
        Tuple,
    )

    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(spaces=tuple(map(_convert_space, space.spaces)))
    elif isinstance(space, gym.spaces.Dict):
        return Dict(spaces={k: _convert_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, gym.spaces.Sequence):
        return Sequence(space=_convert_space(space.feature_space))
    elif isinstance(space, gym.spaces.Graph):
        return Graph(
            node_space=_convert_space(space.node_space),
            edge_space=_convert_space(space.edge_space),
        )
    elif isinstance(space, gym.spaces.Text):
        return Text(
            max_length=space.max_length,
            min_length=space.min_length,
            charset=space._char_str,
        )
    else:
        raise NotImplementedError(
            f"Cannot convert space of type {space}. Please upgrade your code to gymnasium."
        )


def change_colour_agents_rware(env: Any, agent_colours: Dict[str, list[int]]) -> None:
    """
    Customize agent colors in Robotic Warehouse environment.
    
    Modifies the rendering to use custom colors for each agent.
    
    Args:
        env: RWare environment
        agent_colours: Dictionary mapping agent names to RGB colors
    """
    import math
    import pyglet
    import pyglet.gl as gl
    from pyglet.gl import GL_POLYGON, glColor3ub
    from rware.warehouse import Direction
    from rware.rendering import Viewer

    # Custom renderer class with agent color customization
    class AgentsDiffColourViewer(Viewer):
        def _draw_agents(self, env):
            batch = pyglet.graphics.Batch()
            radius = self.grid_size / 3
            resolution = 6
            _AGENT_DIR_COLOR = (0, 0, 0)

            for agent_index, agent in enumerate(env.agents):
                col, row = agent.x, agent.y
                row = self.rows - row - 1  # pyglet rendering is reversed

                # Make a circle
                verts = []
                for i in range(resolution):
                    angle = 2 * math.pi * i / resolution
                    x = (
                        radius * math.cos(angle)
                        + (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                    )
                    y = (
                        radius * math.sin(angle)
                        + (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                    )
                    verts += [x, y]
                circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))

                # Use custom colors if provided
                if agent_colours and agent_colours.get(f"agent_{agent_index}"):
                    colour = agent_colours[f"agent_{agent_index}"]
                else:
                    # Default color is blue
                    colour = (0, 175, 248)
                
                # Draw agent
                glColor3ub(*colour)
                circle.draw(GL_POLYGON)

            # Draw direction indicators
            for agent in env.agents:
                col, row = agent.x, agent.y
                row = self.rows - row - 1  # pyglet rendering is reversed

                batch.add(
                    2,
                    gl.GL_LINES,
                    None,
                    (
                        "v2f",
                        (
                            (self.grid_size + 1) * col
                            + self.grid_size // 2
                            + 1,  # CENTER X
                            (self.grid_size + 1) * row
                            + self.grid_size // 2
                            + 1,  # CENTER Y
                            (self.grid_size + 1) * col
                            + self.grid_size // 2
                            + 1
                            + (
                                radius
                                if agent.dir.value == Direction.RIGHT.value
                                else 0
                            )  # DIR X
                            + (
                                -radius
                                if agent.dir.value == Direction.LEFT.value
                                else 0
                            ),  # DIR X
                            (self.grid_size + 1) * row
                            + self.grid_size // 2
                            + 1
                            + (
                                radius if agent.dir.value == Direction.UP.value else 0
                            )  # DIR Y
                            + (
                                -radius
                                if agent.dir.value == Direction.DOWN.value
                                else 0
                            ),  # DIR Y
                        ),
                    ),
                    ("c3B", (*_AGENT_DIR_COLOR, *_AGENT_DIR_COLOR)),
                )
            batch.draw()

    # Initialize renderer
    try:
        env.unwrapped.render()
    except:
        pass

    # Replace renderer with custom one
    env.unwrapped.renderer = AgentsDiffColourViewer(env.unwrapped.grid_size)