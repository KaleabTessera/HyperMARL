import jax
import jax.numpy as jnp
from functools import partial
from jax.nn import one_hot
import numpy as np

from baselines.Matrix_Game.games import GameType, compute_specialisation_rewards, compute_synchronisation_rewards

class MatrixGameWithState:
    def __init__(self, game_type, num_agents, num_actions, max_steps=100, add_ids=False):
        """
        Matrix game environment with state based on previous actions.

        Args:
            game_type: GameType.SPECIALISATION or GameType.SYNCHRONISATION
            num_agents: Number of agents in the game
            num_actions: Size of the discrete action space
            max_steps: Maximum number of steps per episode
            add_ids: Whether to include agent one-hot IDs in observations
        """
        self.game_type = game_type
        self.num_agents = num_agents
        self.K = num_actions
        self.H = max_steps
        # TODO: add ids not used in the env remove, used in the policy
        self.add_ids = add_ids

        # Episode state
        self.t = 0
        self.prev_actions = None        # jnp array of shape (N,)
        self.cum_rewards = None         # jnp array of shape (N,)

    def reset(self):
        """Reset episode state and return initial observations."""
        self.t = 0
        # Use a sentinel value outside the normal action range
        self.prev_actions = jnp.full(self.num_agents, -1, dtype=jnp.int32)
        self.cum_rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)
        
        # Create a special initial state representation - all -1s
        # This will make a state that can NEVER occur during normal play
        initial_state = jnp.full((self.num_agents, self.K), -1.0, dtype=jnp.float32)
        
        return self._make_obs(initial_state, self.add_ids)

    def _make_obs(self, state_onehot, with_ids):
        """Build per-agent observations from state_onehot."""
        if with_ids:
            eye = jnp.eye(self.num_agents, dtype=jnp.float32)
            return [jnp.concatenate([state_onehot[i], eye[i]]) for i in range(self.num_agents)]
        else:
            flat = state_onehot.reshape(-1)
            return [flat] * self.num_agents

    def get_state(self):
        """Return current per-agent observations."""
        # state feature: one-hot of previous actions
        # clip negative sentinel to zero so one_hot maps -1->zero-vector
        state_feat = one_hot(jnp.clip(self.prev_actions, 0), self.K)
        return self._make_obs(state_feat, self.add_ids)

    def _converged(self, a):
        uniq = jnp.unique(a)
        if self.game_type == GameType.SPECIALISATION:
            return uniq.size == self.num_agents
        else:
            return uniq.size == 1

    # @partial(jax.jit, static_argnums=(0,))
    def step(self, actions):
        """
        actions: jnp.ndarray shape (N,), dtype int32
        returns: (obs, rewards, done, info)
        """
        # 1) reward computation
        if self.game_type == GameType.SPECIALISATION:
            r = compute_specialisation_rewards(actions, self.K)
        else:
            r = compute_synchronisation_rewards(actions, self.K, self.num_agents)

        # 2) update state & time
        self.prev_actions = actions
        self.t += 1
        self.cum_rewards = self.cum_rewards + r

        # 3) done / info
        done = (self.t >= self.H) 
        # | self._converged(actions)
        info = {
            'step': self.t,
            'converged': bool(self._converged(actions)),
            'cumulative_rewards': np.array(self.cum_rewards)
        }

        # 4) next observation
        obs = self.get_state()
        return obs, r, bool(done), info

# Example FakePolicy that works with state
class StatefulFakePolicy:
    def __init__(self, action_fn):
        """
        Create a policy that can respond to state.
        
        Args:
            action_fn: Function that takes state and returns an action
        """
        self.action_fn = action_fn
        
    def choose_action(self, state=None):
        # Return action based on state
        return self.action_fn(state) if state is not None else self.action_fn([])
        
    def choose_greedy_action(self, state=None):
        # Same as regular action for fake policy
        return self.choose_action(state)

def run_episode(env, policies):
    """
    Run a complete episode with the given policies.
    
    Args:
        policies: List of policy objects implementing choose_action(state)
        
    Returns:
        final_rewards: Final cumulative rewards for each agent
        history: Dictionary containing episode history and metrics
    """
    obs = env.reset()
    done = False
    step = 0
    history = {
        'actions': [],
        'rewards': [],
        'states': [],
        'steps_taken': 0,
        'convergence_info': {'converged': False, 'reason': "Max steps reached"}
    }
    
    # Add initial state
    history['states'].append(obs)
    
    # Run until done
    while not done:
        # Get actions from policies
        actions = jnp.array([
            policies[i].choose_action(obs[i]) for i in range(env.num_agents)
        ], dtype=jnp.int32)
        
        # Take step in environment
        obs, rewards, done, info = env.step(actions)
        
        # Record history
        history['actions'].append(np.array(actions))
        history['rewards'].append(np.array(rewards))
        history['states'].append(obs)
        step += 1
        
        # Check for convergence
        if info['converged']:
            history['convergence_info'] = {
                'converged': True,
                'reason': "Optimal policy found",
                'step': step
            }
            done = True
    
    history['steps_taken'] = step
    
    return env.cum_rewards, history

def test_convergence():
    print("Testing Specialisation Convergence...")
    
    # Create environment - specialisation game
    env_spec = MatrixGameWithState(GameType.SPECIALISATION, 3, num_actions=3, max_steps=10)
    
    # Define policies that will eventually converge to optimal solution
    # For specialisation, we want each agent to choose a different action
    def agent0_spec(state):
        return 0  # Always choose action 0
        
    def agent1_spec(state):
        return 1  # Always choose action 1
        
    def agent2_spec(state):
        return 2  # Always choose action 2
    
    policies_spec = [
        StatefulFakePolicy(agent0_spec),
        StatefulFakePolicy(agent1_spec),
        StatefulFakePolicy(agent2_spec)
    ]
    
    # Run episode - should converge quickly
    final_rewards, history_spec = run_episode(env_spec, policies_spec)
    
    print(f"Specialisation game converged: {history_spec['convergence_info']['converged']}")
    print(f"Steps taken: {history_spec['steps_taken']}")
    print(f"Convergence reason: {history_spec['convergence_info']['reason']}")
    print(f"Final rewards: {final_rewards}")
    
    # Validate test results
    assert history_spec['convergence_info']['converged'], "Specialisation game should converge"
    assert history_spec['steps_taken'] == 1, "Should converge in single step with optimal policies"
    assert np.allclose(final_rewards, np.ones(3)), "All agents should receive reward 1.0"
    
    print("\nTesting Synchronisation Convergence...")
    
    # Create environment - synchronisation game
    env_sync = MatrixGameWithState(GameType.SYNCHRONISATION, 3, num_actions=3, max_steps=10)
    
    # Define policies that will eventually converge to optimal solution
    # For synchronisation, we want all agents to choose the same action
    def agent_sync(state):
        return 1  # All agents choose action 1
    
    policies_sync = [
        StatefulFakePolicy(agent_sync),
        StatefulFakePolicy(agent_sync),
        StatefulFakePolicy(agent_sync)
    ]
    
    # Run episode - should converge quickly
    final_rewards, history_sync = run_episode(env_sync, policies_sync)
    
    print(f"Synchronisation game converged: {history_sync['convergence_info']['converged']}")
    print(f"Steps taken: {history_sync['steps_taken']}")
    print(f"Convergence reason: {history_sync['convergence_info']['reason']}")
    print(f"Final rewards: {final_rewards}")
    
    # Validate test results
    assert history_sync['convergence_info']['converged'], "Synchronisation game should converge"
    assert history_sync['steps_taken'] == 1, "Should converge in single step with optimal policies"
    assert np.allclose(final_rewards, np.ones(3)), "All agents should receive reward 1.0"
    
    # Test non-convergence with conflicting policies
    print("\nTesting Non-Convergence with Conflicting Policies...")
    
    def agent0_conflict(state):
        return 0
        
    def agent1_conflict(state):
        return 0  # Conflicts with agent0 in specialisation game
        
    def agent2_conflict(state):
        return 2
    
    policies_conflict = [
        StatefulFakePolicy(agent0_conflict),
        StatefulFakePolicy(agent1_conflict),
        StatefulFakePolicy(agent2_conflict)
    ]
    
    env_spec_conflict = MatrixGameWithState(GameType.SPECIALISATION, 3, num_actions=3, max_steps=5)
    final_rewards, history_conflict = run_episode(env_spec_conflict, policies_conflict)
    
    print(f"Specialisation game with conflict converged: {history_conflict['convergence_info']['converged']}")
    print(f"Steps taken: {history_conflict['steps_taken']}")
    print(f"Convergence reason: {history_conflict['convergence_info']['reason']}")
    print(f"Final rewards: {final_rewards}")
    
    # Validate test results
    assert not history_conflict['convergence_info']['converged'], "Game with conflict shouldn't converge"
    expected_rewards = np.array([0.5, 0.5, 1.0]) * history_conflict['steps_taken']
    assert np.allclose(final_rewards, expected_rewards), f"Expected {expected_rewards}, got {final_rewards}"
    
    # Test state-dependent policy
    print("\nTesting State-Dependent Policy...")
    
    # Policy that changes actions based on previous state
    def adaptive_policy(state):
        # Initial state has -1 values, otherwise use a different strategy
        if np.any(np.array(state) < 0):
            return 0
        else:
            return 1
    
    policies_adaptive = [
        StatefulFakePolicy(adaptive_policy),
        StatefulFakePolicy(adaptive_policy),
        StatefulFakePolicy(adaptive_policy)
    ]
    
    env_adaptive = MatrixGameWithState(GameType.SYNCHRONISATION, 3, num_actions=3, max_steps=5)
    final_rewards, history_adaptive = run_episode(env_adaptive, policies_adaptive)
    
    print(f"Steps taken with adaptive policy: {history_adaptive['steps_taken']}")
    print(f"Action sequence: {[a.tolist() for a in history_adaptive['actions']]}")
    
    # First action should be 0, second should be 1
    assert history_adaptive['actions'][0].tolist() == [0, 0, 0], "First action should be 0 for all agents"
    if len(history_adaptive['actions']) > 1:
        assert history_adaptive['actions'][1].tolist() == [1, 1, 1], "Second action should be 1 for all agents"

    # Test early convergence detection
    print("\nTesting Early Convergence Detection...")
    
    # Create an environment that should finish early when convergence is detected
    env_early = MatrixGameWithState(GameType.SPECIALISATION, 3, num_actions=3, max_steps=100)
    # Uncomment this line in the step method to enable early stopping
    # done = (self.t >= self.H) | self._converged(actions)
    
    # Use the optimal policies from first test
    final_rewards, history_early = run_episode(env_early, policies_spec)
    
    print(f"Early convergence detected: {history_early['convergence_info']['converged']}")
    print(f"Steps taken: {history_early['steps_taken']}")
    
    # Test different configurations
    print("\nTesting Different Game Configurations...")
    
    # Test with more agents
    env_large = MatrixGameWithState(GameType.SPECIALISATION, 5, num_actions=5, max_steps=10)
    
    # Create policies for 5 agents
    policies_large = [
        StatefulFakePolicy(lambda s: 0),
        StatefulFakePolicy(lambda s: 1),
        StatefulFakePolicy(lambda s: 2),
        StatefulFakePolicy(lambda s: 3),
        StatefulFakePolicy(lambda s: 4)
    ]
    
    final_rewards, history_large = run_episode(env_large, policies_large)
    print(f"5-agent game converged: {history_large['convergence_info']['converged']}")
    print(f"Final rewards for 5 agents: {final_rewards}")
    assert np.allclose(final_rewards, np.ones(5)), "All 5 agents should receive reward 1.0"
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_convergence()
    print("All tests passed!")