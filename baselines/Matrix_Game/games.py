import numpy as np
from enum import Enum
from functools import partial
import jax
from jax import numpy as jnp

# Define GameType enum for clarity
class GameType(Enum):
    SPECIALISATION = 1
    SYNCHRONISATION = 2

# Fake policy for testing purposes
class FakePolicy:
    def __init__(self, action):
        # Store the fixed action this fake policy will return
        self.action = action
    def choose_action(self):
        # Return the chosen action (training mode)
        return self.action
    def choose_greedy_action(self):
        # Return the chosen action (evaluation mode)
        return self.action

@partial(jax.jit, static_argnums=(1))
def compute_specialisation_rewards(action_array, num_possible_actions):
    """Compute rewards for specialisation game using JAX operations."""
    # Count occurrences of each action
    action_counts = jnp.zeros(num_possible_actions, dtype=jnp.int32)
    for i, a in enumerate(action_array):
        action_counts = action_counts.at[a].add(1)
    
    # Specialisation: reward = 1 if unique, 
    # otherwise split equally among group who chose the same action
    rewards = jnp.array([
        jnp.where(action_counts[a] == 1, 1.0, 1.0 / action_counts[a])
        for a in action_array
    ])
    
    return rewards



@partial(jax.jit, static_argnums=(1, 2))
def compute_synchronisation_rewards(action_array, num_possible_actions, num_agents):
    """Compute rewards for synchronisation game using JAX operations."""
    # Count occurrences of each action
    action_counts = jnp.zeros(num_possible_actions, dtype=jnp.int32)
    for i, a in enumerate(action_array):
        action_counts = action_counts.at[a].add(1)
    
      
    # Synchronization with hyperbolic growth: (matches spec hyperbolic decay)
    # - Full consensus (all same): 1.0
    # - Partial consensus: 1.0 / (num_agents - action_counts[a] + 1)
    # This creates exponential growth in rewards as more agents coordinate,
    # providing stronger incentives as the system approaches consensus.
    rewards = jnp.array([
        jnp.where(action_counts[a] == num_agents, 
                 1.0,  # Complete consensus gets maximum reward
                 1.0 / (num_agents - action_counts[a] + 1))  # Hyperbolic scaling
        for a in action_array
    ])
    
    return rewards

# Main play_game function with non-JIT portion for policy interactions
# We can make this a lot faster if we JIT compile everything end-to-end, but this 
# will impact the readability of the code. Will leave for future work.
def play_game(game_type, num_agents, policies, policy_type, eval_mode=False, max_actions=100):
    """
    Draw actions for each agent, then compute per-agent payoffs.
    This implementation separates policy interaction from reward computation,
    allowing the reward calculation to be JIT-compiled.
    
    Args:
        game_type: GameType.SPECIALISATION or GameType.SYNCHRONISATION
        num_agents: Number of agents in the game
        policies: List of policies or single policy object
        policy_type: Type of policy arrangement ('PG-NoPS', etc.)
        eval_mode: If True, use greedy action selection
        max_actions: Maximum possible action value
    
    Returns:
        acts: List of integer actions
        rews: NumPy array of rewards
    """
    # 1) Collect actions from each agent (can't be jitted due to policy calls)
    acts = []
    for i in range(num_agents):
        if policy_type == 'PG-NoPS':
            # In PG-NoPS, policies is a list; choose per-agent policy
            a = policies[i].choose_greedy_action() if eval_mode else policies[i].choose_action()
        else:
            # In other modes, policies encapsulates all agents
            a = policies.choose_greedy_action(i) if eval_mode else policies.choose_action(i)
        acts.append(a)
    
    # Convert to JAX array
    acts_array = jnp.array(acts)
    
    # 2) Compute rewards using JIT-compiled functions
    if game_type == GameType.SPECIALISATION:
        rewards = compute_specialisation_rewards(acts_array, max_actions)
    elif game_type == GameType.SYNCHRONISATION:
        rewards = compute_synchronisation_rewards(acts_array, max_actions, num_agents)
    else:
        raise ValueError(f"Unknown game_type: {game_type}")
    
    # Return original acts list and rewards as NumPy array for compatibility
    return acts, jnp.asarray(rewards)

# --- Test cases to verify correctness ---
def test_specialisation():
    # Test 1: All unique actions -> each gets reward 1.0 - two agents
    
    policies = [FakePolicy(0), FakePolicy(1)]
    acts, rews = play_game(GameType.SPECIALISATION, 2, policies, 'PG-NoPS')
    assert np.allclose(rews, [1.0, 1.0]), f"Unique actions should yield 1.0 each, got {rews}"
    
    # Test 2: Same actions -> split reward
    policies = [FakePolicy(1), FakePolicy(1)]
    _, rews = play_game(GameType.SPECIALISATION, 2, policies, 'PG-NoPS')
    assert np.allclose(rews, [0.5, 0.5]), f"Shared actions should split 1.0, got {rews}"
    
    # Test 3: All unique actions -> each gets reward 1.0
    policies = [FakePolicy(0), FakePolicy(1), FakePolicy(2)]
    acts, rews = play_game(GameType.SPECIALISATION, 3, policies, 'PG-NoPS')
    assert np.allclose(rews, [1.0, 1.0, 1.0]), f"Unique actions should yield 1.0 each, got {rews}"

    # Test 4: Two share and one unique -> sharers split 1.0, unique gets 1.0
    policies = [FakePolicy(1), FakePolicy(1), FakePolicy(2)]
    _, rews = play_game(GameType.SPECIALISATION, 3, policies, 'PG-NoPS')
    assert np.allclose(rews, [1/2, 1/2, 1.0]), f"Shared actions should split 1.0, print {rews}"

    # Test 5: Partial groups -> reward proportional to group size
    policies = [FakePolicy(1), FakePolicy(1), FakePolicy(2), FakePolicy(3)]
    _, rews = play_game(GameType.SPECIALISATION, 4, policies, 'PG-NoPS')
    assert np.allclose(rews, [1.0/2, 1.0/2, 1, 1]), f"Reward should be proportional to action count / num_agents, got {rews}"


def test_synchronisation():
    # Test 1: Different actions -> each gets reward proportional to count/num_agents
    policies = [FakePolicy(0), FakePolicy(1)]
    acts, rews = play_game(GameType.SYNCHRONISATION, 2, policies, 'PG-NoPS')
    assert np.allclose(rews, [0.5, 0.5]), "Individual agents should get 1/num_agents reward"
    
    # Test 2: Same actions -> reward of 1.0 each (full consensus)
    policies = [FakePolicy(1), FakePolicy(1)]
    _, rews = play_game(GameType.SYNCHRONISATION, 2, policies, 'PG-NoPS')
    assert np.allclose(rews, [1.0, 1.0]), f"Full consensus should yield 1.0 each, got {rews}"
    
    # Test 3: Full sync -> all agents get 1.0
    policies = [FakePolicy(1), FakePolicy(1), FakePolicy(1), FakePolicy(1)]
    acts, rews = play_game(GameType.SYNCHRONISATION, 4, policies, 'PG-NoPS')
    assert np.allclose(rews, [1.0, 1.0, 1.0, 1.0]), "Full sync should yield 1.0 each"

    # Test 4: Partial groups -> reward proportional to group size
    policies = [FakePolicy(1), FakePolicy(1), FakePolicy(2), FakePolicy(3)]
    _, rews = play_game(GameType.SYNCHRONISATION, 4, policies, 'PG-NoPS')
    assert np.allclose(rews, [1.0/(4-2+1), 1.0/(4-2+1), 1/4, 1/4]), f"Reward should be proportional to action count / num_agents, got {rews}"

# Execute tests
if __name__ == "__main__":
    test_specialisation()
    test_synchronisation()
    print("All tests passed!")
