import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from baselines.utils.wrappers import make_env


def run_eval_jax(
    cfg,
    agent_state,
    writer,
    acting_fns,
    eval_seed,
    global_step,
    capture_video,
    seed_reset=True,
    recurrent=False,
    recurrent_init_fn=None,
    shared_weights=False,
    parallel=False,
):
    """
    Main evaluation function that dispatches to either parallel or sequential execution.
    
    This function serves as the entry point for policy evaluation, choosing between
    parallel (faster) or sequential (more detailed) evaluation modes based on configuration.
    
    Args:
        cfg: Configuration dictionary with evaluation parameters
        agent_state: Trained agent parameters
        writer: Logging writer (e.g., WandB)
        acting_fns: Functions to select actions from policies
        eval_seed: Random seed for evaluation
        global_step: Current training step for logging
        capture_video: Whether to record videos of evaluation episodes
        seed_reset: Whether to reset environment with new seeds each episode
        recurrent: Whether the policy is recurrent (using LSTM/GRU)
        recurrent_init_fn: Function to initialize recurrent states
        shared_weights: Whether agents share network weights
        parallel: Whether to run evaluation in parallel
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if parallel:
        return run_eval_parallel_jax(
            cfg,
            agent_state,
            writer,
            acting_fns,
            eval_seed,
            global_step,
            capture_video,
            seed_reset,
            recurrent,
            recurrent_init_fn,
            shared_weights,
        )
    else:
        return run_eval_sequential_jax(
            cfg,
            agent_state,
            writer,
            acting_fns,
            eval_seed,
            global_step,
            capture_video,
            seed_reset,
            recurrent,
            recurrent_init_fn,
            shared_weights,
        )


def run_eval_sequential_jax(
    cfg,
    agent_state,
    writer,
    acting_fns,
    eval_seed,
    global_step,
    capture_video,
    seed_reset=True,
    recurrent=False,
    recurrent_init_fn=None,
    shared_weights=False,
):
    """
    Run evaluation sequentially (one episode at a time).
    
    This mode is useful for detailed analysis and when capturing videos
    of individual episodes is important.
    
    Args:
        cfg: Configuration dictionary
        agent_state: Trained agent parameters
        writer: Logging writer
        acting_fns: Functions to select actions
        eval_seed: Random seed for evaluation
        global_step: Current training step for logging
        capture_video: Whether to record videos
        seed_reset: Whether to reset with new seeds each episode
        recurrent: Whether the policy is recurrent
        recurrent_init_fn: Function to initialize recurrent states
        shared_weights: Whether agents share network weights
        
    Returns:
        Dictionary containing evaluation metrics
    """
    total_eval_episodes = cfg["EVAL_EPISODES"]
    store_full_data = cfg.get("STORE_FULL_EVAL_DATA", False)

    # Create evaluation environment
    env, possible_agents, _, _, _ = make_env(
        cfg["ENV_NAME"] if not cfg.get("TEST_ENV_NAME") else cfg["TEST_ENV_NAME"],
        render_mode="rgb_array",
        capture_video=capture_video,
        video_location=(
            f"./videos/{cfg['RUN_NAME']}/{global_step}" if capture_video else None
        ),
        num_envs=1,  # Sequential evaluation uses a single environment
        **cfg["TEST_ENV_KWARGS"],
    )

    num_agents = len(possible_agents)
    rng = jax.random.PRNGKey(eval_seed)

    # Run evaluation episodes
    episode_results = run_sequential_episodes(
        env,
        agent_state,
        acting_fns,
        rng,
        cfg,
        num_agents,
        seed_reset,
        recurrent,
        recurrent_init_fn,
        shared_weights,
        total_eval_episodes,
        store_full_data,
    )

    env.close()
    return episode_results


def run_sequential_episodes(
    env,
    agent_state,
    acting_fns,
    rng,
    cfg,
    num_agents,
    seed_reset,
    recurrent,
    recurrent_init_fn,
    shared_weights,
    total_eval_episodes,
    store_full_data,
):
    """
    Run multiple evaluation episodes sequentially.
    
    Executes multiple episodes one after another and collects performance metrics.
    
    Args:
        env: Environment to evaluate in
        agent_state: Trained agent parameters
        acting_fns: Functions to select actions
        rng: JAX random number generator
        cfg: Configuration dictionary
        num_agents: Number of agents in the environment
        seed_reset: Whether to reset with new seeds each episode
        recurrent: Whether the policy is recurrent
        recurrent_init_fn: Function to initialize recurrent states
        shared_weights: Whether agents share network weights
        total_eval_episodes: Number of episodes to evaluate
        store_full_data: Whether to store complete trajectory data
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize arrays to store results
    episode_returns = np.zeros((total_eval_episodes, num_agents))
    episode_lengths = np.zeros(total_eval_episodes, dtype=np.int32)

    # If requested, allocate storage for complete trajectory data
    if store_full_data:
        all_obs = [[] for _ in range(total_eval_episodes)]
        all_actions = [[] for _ in range(total_eval_episodes)]
        all_rewards = [[] for _ in range(total_eval_episodes)]

    # Execute each episode
    for episode in range(total_eval_episodes):
        # Generate new RNG key for episode
        rng, reset_rng = jax.random.split(rng)
        
        # Reset environment (with random seed if configured)
        if seed_reset:
            obs, _ = env.reset(
                seed=jax.random.randint(reset_rng, (), 0, 1_000_000).item()
            )
        else:
            obs, _ = env.reset()

        # Initialize recurrent state if needed
        lstm_state = initialize_lstm_state(
            recurrent, recurrent_init_fn, num_agents, cfg, shared_weights
        )

        step = 0
        done = False

        # Run episode until termination
        while not done:
            # Generate RNG key for action selection
            rng, action_rng = jax.random.split(rng)
            
            # Get actions from policy
            actions, lstm_state = get_actions(
                agent_state,
                acting_fns,
                obs,
                lstm_state,
                recurrent,
                shared_weights,
                action_rng,
            )

            # Step environment
            next_obs, rewards, terms, truncs, _ = env.step(actions)

            # Record returns and increment episode length
            episode_returns[episode] += rewards
            episode_lengths[episode] += 1

            # Store full trajectory data if requested
            if store_full_data:
                all_obs[episode].append(obs)
                all_actions[episode].append(actions)
                all_rewards[episode].append(rewards)

            # Check if episode is complete
            done = np.any(terms) or np.any(truncs)
            obs = next_obs
            step += 1

    # Compile results
    episode_results = {
        "returns": episode_returns,
        "lengths": episode_lengths,
    }

    # Add full trajectory data if collected
    if store_full_data:
        episode_results.update(
            {
                "obs": [np.array(obs) for obs in all_obs],
                "actions": [np.array(actions) for actions in all_actions],
                "rewards": [np.array(rewards) for rewards in all_rewards],
            }
        )

    return episode_results


def run_eval_parallel_jax(
    cfg,
    agent_state,
    writer,
    acting_fns,
    eval_seed,
    global_step,
    capture_video,
    seed_reset=True,
    recurrent=False,
    recurrent_init_fn=None,
    shared_weights=False,
):
    """
    Run evaluation in parallel (multiple episodes simultaneously).
    
    This mode is more efficient for collecting statistics across many episodes
    by leveraging vectorized environments.
    
    Args:
        cfg: Configuration dictionary
        agent_state: Trained agent parameters
        writer: Logging writer
        acting_fns: Functions to select actions
        eval_seed: Random seed for evaluation
        global_step: Current training step for logging
        capture_video: Whether to record videos
        seed_reset: Whether to reset with new seeds each episode
        recurrent: Whether the policy is recurrent
        recurrent_init_fn: Function to initialize recurrent states
        shared_weights: Whether agents share network weights
        
    Returns:
        Dictionary containing evaluation metrics
    """
    total_eval_episodes = cfg["EVAL_EPISODES"]
    store_full_data = cfg.get("STORE_FULL_EVAL_DATA", False)

    # Create evaluation environment (vectorized)
    env, possible_agents, _, _, _ = make_env(
        cfg["ENV_NAME"] if not cfg.get("TEST_ENV_NAME") else cfg["TEST_ENV_NAME"],
        render_mode="rgb_array",
        capture_video=capture_video,
        video_location=(
            f"./videos/{cfg['RUN_NAME']}/{global_step}" if capture_video else None
        ),
        num_envs=total_eval_episodes,  # Parallel evaluation uses multiple environments
        **cfg["TEST_ENV_KWARGS"],
    )

    num_agents = len(possible_agents)
    rng = jax.random.PRNGKey(eval_seed)

    # Run evaluation episodes in parallel
    episode_results = run_parallel_episodes(
        env,
        agent_state,
        acting_fns,
        rng,
        cfg,
        num_agents,
        seed_reset,
        recurrent,
        recurrent_init_fn,
        shared_weights,
        total_eval_episodes,
        store_full_data,
    )

    env.close()
    return episode_results


def run_parallel_episodes(
    env,
    agent_state,
    acting_fns,
    rng,
    cfg,
    num_agents,
    seed_reset,
    recurrent,
    recurrent_init_fn,
    shared_weights,
    total_eval_episodes,
    store_full_data,
):
    """
    Run multiple evaluation episodes in parallel.
    
    Executes multiple episodes simultaneously using vectorized environments
    for faster evaluation.
    
    Args:
        env: Vectorized environment to evaluate in
        agent_state: Trained agent parameters
        acting_fns: Functions to select actions
        rng: JAX random number generator
        cfg: Configuration dictionary
        num_agents: Number of agents in the environment
        seed_reset: Whether to reset with new seeds each episode
        recurrent: Whether the policy is recurrent
        recurrent_init_fn: Function to initialize recurrent states
        shared_weights: Whether agents share network weights
        total_eval_episodes: Number of episodes to evaluate
        store_full_data: Whether to store complete trajectory data
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Reset all environments
    rng, reset_rng = jax.random.split(rng)
    if seed_reset:
        obs, _ = env.reset(seed=jax.random.randint(reset_rng, (), 0, 1_000_000).item())
    else:
        obs, _ = env.reset()

    # Initialize recurrent state if needed (for all envs at once)
    lstm_state = initialize_lstm_state(
        recurrent,
        recurrent_init_fn,
        num_agents * total_eval_episodes,
        cfg,
        shared_weights,
    )

    # Initialize arrays to store results
    episode_returns = np.zeros((total_eval_episodes, num_agents))
    episode_lengths = np.zeros(total_eval_episodes, dtype=np.int32)

    # If requested, allocate storage for complete trajectory data
    if store_full_data:
        all_obs = [[] for _ in range(total_eval_episodes)]
        all_actions = [[] for _ in range(total_eval_episodes)]
        all_rewards = [[] for _ in range(total_eval_episodes)]

    # Track which environments are still active
    active_envs = np.ones(total_eval_episodes, dtype=bool)

    # Run until all environments are done
    while np.any(active_envs):
        # Generate RNG key for action selection
        rng, action_rng = jax.random.split(rng)
        
        # Get actions from policy
        actions, lstm_state = get_actions(
            agent_state,
            acting_fns,
            obs,
            lstm_state,
            recurrent,
            shared_weights,
            action_rng,
        )

        # Step all active environments
        next_obs, rewards, agent_terms, agent_truncs, _ = env.step(actions)

        # Update metrics only for active environments
        episode_returns[active_envs] += rewards.reshape(-1, num_agents)[active_envs]
        episode_lengths[active_envs] += 1

        # Store full trajectory data if requested
        if store_full_data:
            for i in np.where(active_envs)[0]:
                all_obs[i].append(obs[i])
                all_actions[i].append(actions[i])
                all_rewards[i].append(rewards[i])

        # Determine which environments are done
        # Reshape agent termination flags to (num_envs, num_agents)
        reshaped_agent_dones = agent_terms.reshape(total_eval_episodes, num_agents)
        env_terms = np.any(reshaped_agent_dones, axis=1)
        
        # Reshape agent truncation flags to (num_envs, num_agents)
        reshaped_agent_truncs = agent_truncs.reshape(total_eval_episodes, num_agents)
        env_truncs = np.any(reshaped_agent_truncs, axis=1)
        
        # Update active_envs mask (an environment is inactive if any agent is done)
        active_envs = ~(env_terms | env_truncs)

        # Update observations
        obs = next_obs

    # Compile results
    episode_results = {
        "returns": episode_returns,
        "lengths": episode_lengths,
    }

    # Add full trajectory data if collected
    if store_full_data:
        episode_results.update(
            {
                "obs": [np.array(obs) for obs in all_obs],
                "actions": [np.array(actions) for actions in all_actions],
                "rewards": [np.array(rewards) for rewards in all_rewards],
            }
        )

    return episode_results


@partial(jax.jit, static_argnums=(1, 4, 5))
def get_actions(
    agent_state, acting_fns, obs, lstm_state, recurrent, shared_weights, rng
):
    """
    Get actions from policy network.
    
    Handles different policy types (recurrent/non-recurrent, shared/non-shared weights)
    and returns appropriate actions.
    
    Args:
        agent_state: Trained agent parameters
        acting_fns: Functions to select actions
        obs: Current observations
        lstm_state: Current recurrent state (if applicable)
        recurrent: Whether the policy is recurrent
        shared_weights: Whether agents share network weights
        rng: JAX random number generator
        
    Returns:
        actions: Selected actions
        new_lstm_state: Updated recurrent state (if applicable)
    """
    if recurrent:
        if shared_weights:
            # Shared recurrent network - process all agents at once
            return acting_fns(agent_state, obs, lstm_state, rng)
        else:
            # Separate recurrent networks - process each agent individually
            actions = []
            new_lstm_states = []
            for i in range(len(obs)):
                rng, subrng = jax.random.split(rng)
                action, new_lstm_state = acting_fns(
                    agent_state, obs[i : i + 1], lstm_state[i], subrng
                )
                actions.append(action)
                new_lstm_states.append(new_lstm_state)
            return jnp.concatenate(actions, axis=0), jnp.array(new_lstm_states)
    else:
        # Non-recurrent network - simpler processing
        actions = acting_fns(agent_state, obs, rng)[0]
        return actions, lstm_state


def initialize_lstm_state(
    recurrent, recurrent_init_fn, num_agents, cfg, shared_weights
):
    """
    Initialize recurrent state for policies with memory.
    
    Creates appropriate initial states for LSTM/GRU networks based on
    architecture and sharing configuration.
    
    Args:
        recurrent: Whether the policy is recurrent
        recurrent_init_fn: Function to initialize recurrent states
        num_agents: Number of agents requiring states
        cfg: Configuration dictionary
        shared_weights: Whether agents share network weights
        
    Returns:
        lstm_state: Initial recurrent state (or None if not recurrent)
    """
    if not recurrent:
        return None

    if shared_weights:
        # Shared recurrent network - single state for all agents
        return recurrent_init_fn(num_agents, cfg["LAYER_WIDTH"])
    else:
        # Separate recurrent networks - separate state for each agent
        return jnp.array(
            [recurrent_init_fn(1, cfg["LAYER_WIDTH"]) for _ in range(num_agents)]
        )