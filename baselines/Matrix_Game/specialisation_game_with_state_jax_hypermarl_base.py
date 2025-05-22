# HyperMARL with Hypernetworks for Matrix Games with State
# Same as specialisation_game_with_state_jax, but with HyperMARL
from enum import Enum
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


import jax
import jax.numpy as jnp
from jax import random
import optax

import flax.linen as nn
import distrax
import hydra
import wandb

from baselines.Matrix_Game.games import GameType
from baselines.Matrix_Game.games_with_state import MatrixGameWithState
from baselines.Matrix_Game.logging_utils import (
    compute_stability_proxy,
    log_all_grad_metrics,
    log_all_stability_proxy,
    log_raw_gradient_conflicts,
    log_raw_gradient_norms,
    plot_combined_results,
    plot_gradient_conflicts,
    plot_stability_proxy,
)


# ------------------------------------------------------------
# hypermarl stuff
# ------------------------------------------------------------


class HyperNetType(Enum):
    ACTOR = 0
    CRITIC = 1


def check_rows_orthogonal(A):
    dot_products = jnp.dot(A, A.T)
    off_diagonal_elements = dot_products - jnp.diag(jnp.diag(dot_products))
    is_orthogonal = jnp.all(jnp.isclose(off_diagonal_elements, 0, atol=0.0001))
    return is_orthogonal


class MLPHyperNetwork(nn.Module):
    """MLP HyperNetwork for generating weights and biases of all layers in the network."""

    output_dims: List[Tuple[int, int]]
    hypernet_type: HyperNetType
    init_scale: float = np.sqrt(2)  # default when using relu
    use_bias: bool = True
    hidden_dims: List[int] = (64,)  # Default hidden layer sizes for the MLP
    use_custom_init: bool = True  # Toggle for custom initialization
    init_type: str = "orthogonal"  # Type of initialization to use

    @staticmethod
    def hypernet_init(gain, fan_in, fan_out, init_type="orthogonal"):
        # print(f"Using {init_type} initialization with gain")
        def weight_init(key, shape, dtype):
            if init_type == "orthogonal":
                init = jax.nn.initializers.orthogonal(gain)
            # lecun normal is the default for dense layers in jax
            elif init_type == "lecun_normal":
                init = jax.nn.initializers.lecun_normal()
            batched_init = jax.vmap(init, in_axes=(0, None, None))

            batch_size = shape[0]
            keys = jax.random.split(key, num=batch_size)

            weights = batched_init(keys, (fan_in, fan_out), dtype)
            return weights.reshape(shape)

        return weight_init

    @nn.compact
    def __call__(self, x):
        weight_heads = []
        bias_heads = []
        for i, (input_dim, output_dim) in enumerate(self.output_dims):
            weight_dim = input_dim * output_dim
            bias_dim = output_dim

            is_final_layer = i == len(self.output_dims) - 1

            if is_final_layer and self.hypernet_type == HyperNetType.ACTOR:
                gain = 0.01
            elif is_final_layer and self.hypernet_type == HyperNetType.CRITIC:
                gain = 1
            else:
                gain = self.init_scale

            # MLP for weights
            weight_mlp = x
            for hidden_dim in self.hidden_dims:
                weight_mlp = nn.Dense(hidden_dim, use_bias=self.use_bias)(weight_mlp)
                weight_mlp = nn.relu(weight_mlp)

            if self.use_custom_init:
                weight_head = nn.Dense(
                    weight_dim,
                    use_bias=self.use_bias,
                    kernel_init=self.hypernet_init(
                        gain, input_dim, output_dim, self.init_type
                    ),
                    bias_init=nn.initializers.zeros,
                )(weight_mlp)
            else:  # default init
                weight_head = nn.Dense(
                    weight_dim,
                    use_bias=self.use_bias,
                    bias_init=nn.initializers.zeros,
                )(weight_mlp)

            # MLP for biases
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


# ------------------------------------------------------------
# Flax policy network and PG classes
# ------------------------------------------------------------


class PolicyNetwork(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        return nn.Dense(self.output_size)(x)


class PGNoPS:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        optim,
        seed=0,
        gamma=0.99,
        target_updates=None,
    ):
        self.net = PolicyNetwork(input_size, hidden_size, output_size)
        self.key = random.PRNGKey(seed)
        dummy = jnp.zeros((1, input_size))
        self.params = self.net.init(self.key, dummy)
        optim_name = optim.get("optim")
        lr = optim.get("learning_rate")
        if optim_name == "adam":
            self.opt = optax.adam(lr)
        elif optim_name == "sgd":
            self.opt = optax.sgd(lr)

        print(f"Using {optim_name} optimizer with learning rate {lr}")
        max_grad_norm = optim.get("max_grad_norm", None)
        if max_grad_norm:
            print(f"using max grad norm {max_grad_norm}")
            self.opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), self.opt)

        self.opt_state = self.opt.init(self.params)
        self.gamma = gamma
        self.num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))

        self.states, self.actions, self.rewards = [], [], []
        self.returns = []
        self.gradient_norms = []
        # not used currently
        self.gradient_conflicts = []
        self.gradient_variances = []
        self.losses = []
        self.all_returns = []
        self.num_updates = 0
        self.target_updates = target_updates

        def loss_fn(p, states, actions, returns):
            logits = self.net.apply(p, states)
            lp = distrax.Categorical(logits=logits).log_prob(actions)
            return -jnp.sum(lp * returns)

        @jax.jit
        def _step(params, opt_state, states, actions, returns):

            loss, grads = jax.value_and_grad(loss_fn)(params, states, actions, returns)
            grads_flat, _ = jax.flatten_util.ravel_pytree(grads)
            norm = jnp.linalg.norm(grads_flat)
            var = jnp.var(grads_flat)
            updates, new_state = self.opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, norm, var, loss

        @jax.jit
        def _choose_action(params, key, state):
            # Ensure state is properly shaped
            state_tensor = jnp.array(state).reshape(1, -1)
            logits = self.net.apply(params, state_tensor)
            key, subkey = random.split(key)
            dist = distrax.Categorical(logits=logits[0])
            a = dist.sample(seed=subkey)
            return a, dist.log_prob(a), key, state_tensor

        @jax.jit
        def _choose_action_greedy(params, state):
            # Ensure state is properly shaped
            state_tensor = jnp.array(state).reshape(1, -1)
            logits = self.net.apply(params, state_tensor)
            return jnp.argmax(logits[0])

        self._update_step = _step
        self._choose_action = _choose_action
        self._choose_action_greedy = _choose_action_greedy

    def choose_action(self, state):
        action, dist, self.key, state = self._choose_action(
            self.params, self.key, state
        )
        self.states.append(state)
        self.actions.append(action)
        return action

    def choose_greedy(self, state):
        return self._choose_action_greedy(self.params, state)

    def store_reward(self, reward, done=False):
        self.rewards.append(reward)
        # self.all_returns.append(reward)
        if done:
            R = 0
            returns = []
            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = np.array(returns, dtype=np.float32)
            # for logging
            self.all_returns.extend(returns.tolist())
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            self.returns.extend(returns)
            self.rewards.clear()

    def update_policy(self):
        if not self.states:
            return False

        # TODO: use batch size param from config
        batch_size = 32
        # not enough transitions to do a full batch
        if len(self.states) < batch_size:
            return False

        self.num_updates += 1
        # TODO: probs dont need to [:batch_size] here
        S = jnp.vstack(self.states[:batch_size])
        A = jnp.array(self.actions[:batch_size])
        R = jnp.array(self.returns[:batch_size])

        self.params, self.opt_state, norm, var, loss = self._update_step(
            self.params, self.opt_state, S, A, R
        )

        # only drop used transitions
        self.states = self.states[batch_size:]
        self.actions = self.actions[batch_size:]
        self.returns = self.returns[batch_size:]
        self.gradient_norms.append(norm)
        self.gradient_variances.append(var)
        self.losses.append(loss)
        return True


class PGFuPS(PGNoPS):
    def __init__(
        self,
        num_agents,
        input_size,
        hidden_size,
        output_size,
        lr,
        seed=0,
        gamma=0.99,
        use_agent_id=False,
        use_state=True,
        compute_conflicts=False,
        target_updates=None,
    ):
        super().__init__(
            input_size, hidden_size, output_size, lr, seed, gamma, target_updates
        )
        self.num_agents = num_agents
        self.gradient_conflicts = []
        self.use_agent_id = use_agent_id
        self.use_state = use_state
        self.compute_conflicts = compute_conflicts
        # Number of times to compute conflicts during training
        self.conflict_intervals = 5

        # Track which updates we should compute conflicts for
        self.conflict_updates = set()
        self.target_updates = int(self.target_updates)
        interval_points = np.linspace(
            1, self.target_updates, self.conflict_intervals, dtype=int
        )
        self.conflict_updates = set(interval_points)

        if self.compute_conflicts:
            print(
                f"Gradient conflicts will be computed at updates: {self.conflict_updates}"
            )

        assert (
            use_state or use_agent_id
        ), "Either use_state or use_agent_id must be True"

    def choose_action(self, agent_id, state=None):
        if self.use_agent_id and self.use_state:
            one_hot = jax.nn.one_hot(agent_id, self.num_agents)
            state = jnp.concatenate([one_hot, jnp.array(state)]).reshape(1, -1)
        elif self.use_agent_id and not self.use_state:
            one_hot = jax.nn.one_hot(agent_id, self.num_agents)
            state = one_hot.reshape(1, -1)
        return super().choose_action(state)

    def choose_greedy(self, agent_id, state=None):
        if self.use_agent_id and self.use_state:
            one_hot = jax.nn.one_hot(agent_id, self.num_agents)
            state = jnp.concatenate([one_hot, jnp.array(state)]).reshape(1, -1)
        elif self.use_agent_id and not self.use_state:
            one_hot = jax.nn.one_hot(agent_id, self.num_agents)
            state = one_hot.reshape(1, -1)
        return super().choose_greedy(state)

    def _compute_gradient_conflict(self, params, states, actions, returns, agent_ids):
        """Compute cosine similarity between gradients of different agents in a vectorized way."""
        # print(f"Computing gradient conflicts for {len(states)} transitions")
        # Get unique agent IDs and count them
        unique_ids = jnp.unique(agent_ids)

        # Define loss function once
        def loss_fn(p, states, actions, returns):
            logits = self.net.apply(p, states)
            lp = distrax.Categorical(logits=logits).log_prob(actions)
            return -jnp.sum(lp * returns)

        # Compute gradients for all agents in vectorized form
        all_gradients = []

        # Vectorized computation for each agent ID
        for agent_id in unique_ids:
            # Create masks for this agent's data
            mask = agent_ids == agent_id
            if not jnp.any(mask):
                continue

            # Extract this agent's data using the mask
            agent_states = states[mask]
            agent_actions = actions[mask]
            agent_returns = returns[mask]

            # Compute gradient for this agent
            _, grad = jax.value_and_grad(loss_fn)(
                params, agent_states, agent_actions, agent_returns
            )
            flat_grad, _ = jax.flatten_util.ravel_pytree(grad)

            # Normalize the gradient for computing cosine similarity
            grad_norm = jnp.linalg.norm(flat_grad)
            normalized_grad = jnp.where(grad_norm > 0, flat_grad / grad_norm, flat_grad)
            all_gradients.append(normalized_grad)

        # Stack all gradients into a matrix
        if not all_gradients:
            return 0.0

        grad_matrix = jnp.stack(all_gradients)

        # Compute all pairwise dot products at once (these are cosine similarities since we normalized)
        similarities = jnp.matmul(grad_matrix, grad_matrix.T)

        # Extract upper triangle (excluding diagonal) to get all unique pairs
        mask = jnp.triu(jnp.ones_like(similarities), k=1)
        pairwise_sims = similarities[mask.astype(bool)]

        # Return negative mean as conflict metric
        return -jnp.mean(pairwise_sims)

    def update_policy(self):
        if self.compute_conflicts:
            if not self.states:
                return False

            batch_size = 32
            if len(self.states) < batch_size:
                return False

            S = jnp.vstack(self.states)
            A = jnp.array(self.actions)
            R = jnp.array(self.returns)

            # Extract agent IDs from the states if we're using agent IDs
            if self.use_agent_id:
                if self.use_state:
                    # Agent ID is in the first self.num_agents elements as one-hot encoding
                    agent_ids = jnp.argmax(S[:, : self.num_agents], axis=1)
                else:
                    # State is just the agent ID
                    agent_ids = jnp.argmax(S, axis=1)

                if self.num_updates + 1 in self.conflict_updates:
                    # Compute gradient conflicts between agents
                    grad_conflict = self._compute_gradient_conflict(
                        self.params, S, A, R, agent_ids
                    )
                    self.gradient_conflicts.append(grad_conflict)

        return super().update_policy()


# Override the network with our hypernetwork
class HyperPolicyNetwork(nn.Module):
    """Policy network that uses a hypernetwork to generate weights"""

    input_size: int
    hidden_size: int
    output_size: int
    num_agents: int
    embedding_dim: int
    hypernet_hidden_dims: List[int] = (16,)
    use_agent_id_embeddings: bool = True
    init_scale: float = np.sqrt(2)  # default when using relu
    use_custom_init: bool = True  # Toggle for custom initialization
    init_type: str = "orthogonal"  # Type of initialization to use
    embedding_init_scale: float = np.sqrt(
        2
    )  # Default scale for embedding initialization

    def setup(self):
        # Initialize the hypernetwork
        self.hypernet = MLPHyperNetwork(
            output_dims=[
                (self.input_size, self.hidden_size),
                (self.hidden_size, self.output_size),
            ],
            hypernet_type=HyperNetType.ACTOR,
            hidden_dims=self.hypernet_hidden_dims,
            use_custom_init=self.use_custom_init,
            init_type=self.init_type,
        )

        # print(f"Using hypernet with hidden dims {self.hypernet_hidden_dims} and init type {self.init_type} with scale {self.embedding_init_scale}")

        self.agent_embeddings = (
            self.param(
                "agent_embeddings",
                nn.initializers.orthogonal(self.embedding_init_scale),
                (self.num_agents, self.embedding_dim),
            )
            if self.use_agent_id_embeddings
            else jnp.eye(self.num_agents)
        )

        # orthogonality = check_rows_orthogonal(self.agent_embeddings)
        # print(orthogonality)

    @nn.compact
    def __call__(self, x, agent_id=None):
        # If agent_id is not provided in the input, extract it from the last part
        if agent_id is None:
            # assert agent_id_one_hot is one-hot encoded
            agent_id_one_hot = x[:, : self.num_agents]
            # assert jnp.all(jnp.sum(agent_id_one_hot, axis=1) == 1), f"Agent ID must be one-hot encoded: {agent_id_one_hot}"
            # Get agent_id from the input tensor
            agent_id = jnp.argmax(agent_id_one_hot, axis=1)

        # Get the state portion only
        state = x[:, self.num_agents :]

        # Define network structure for the policy
        output_dims = [
            (self.input_size, self.hidden_size),
            (self.hidden_size, self.output_size),
        ]

        # Use hypernetwork to generate all agent weights in parallel
        # Generate weights and biases for all agents
        all_weights, all_biases = self.hypernet(self.agent_embeddings)

        # Apply the generated weights for each agent in the batch
        batch_size = state.shape[0]
        # assert state.shape[1] == self.input_size, f"State shape {state.shape} doesn't match expected input size {self.input_size}"

        # Get indices for agent_ids
        batch_indices = jnp.arange(batch_size)

        # First layer
        w1 = all_weights[0][agent_id].reshape(
            batch_size, self.input_size, self.hidden_size
        )
        b1 = all_biases[0][agent_id]

        # Second layer
        w2 = all_weights[1][agent_id].reshape(
            batch_size, self.hidden_size, self.output_size
        )
        b2 = all_biases[1][agent_id]

        # Apply the generated weights with batched matmul
        h = jnp.matmul(state, w1) + b1
        h = jax.nn.relu(h)

        logits = jnp.matmul(h, w2) + b2

        return logits


class PGHyperMARL(PGFuPS):
    """Policy Gradient with Hypernetwork architecture for agent specialization"""

    def __init__(
        self,
        num_agents,
        input_size,
        hidden_size,
        output_size,
        optim,
        seed=0,
        gamma=0.99,
        embedding_dim=None,
        use_agent_id_embeddings=True,
        hypernet_hidden_dims=(16,),
        use_custom_init=True,
        init_type="orthogonal",
        embedding_init_scale=np.sqrt(2),
        compute_grad_conflict=False,
        target_updates=None,
    ):
        # Initialize with minimal input size (state only)
        super().__init__(
            num_agents,
            input_size,
            hidden_size,
            output_size,
            optim,
            seed,
            gamma,
            use_agent_id=True,
            use_state=True,
            compute_conflicts=compute_grad_conflict,
            target_updates=target_updates,
        )

        # Set embedding dimension based on config or default
        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else hidden_size // num_agents
        )
        self.use_agent_id_embeddings = use_agent_id_embeddings
        self.hypernet_hidden_dims = hypernet_hidden_dims

        print(
            f"Using embedding dim {self.embedding_dim} and hypernet hidden dims {self.hypernet_hidden_dims} use agent id embeddings {self.use_agent_id_embeddings} use custom init {use_custom_init} init type {init_type}"
        )

        # Create the network
        self.net = HyperPolicyNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_agents=num_agents,
            embedding_dim=self.embedding_dim,
            hypernet_hidden_dims=self.hypernet_hidden_dims,
            use_agent_id_embeddings=self.use_agent_id_embeddings,
            use_custom_init=use_custom_init,
            init_type=init_type,
            embedding_init_scale=embedding_init_scale,
        )

        # Re-initialize params with the new network
        # include one hot id in dummy state
        dummy_state = jnp.zeros((1, input_size))
        # include one hot id in dummy state
        dummy_state = jnp.concatenate(
            [jax.nn.one_hot(0, num_agents).reshape(1, -1), dummy_state], axis=1
        )
        self.params = self.net.init(self.key, dummy_state)
        self.opt_state = self.opt.init(self.params)
        self.num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))


# ------------------------------------------------------------
# Play and training loops with logging and multi-seed support
# ------------------------------------------------------------


def play_episode(env, policies, name, eval_mode=False, max_steps=100):
    state = env.reset()
    done = False
    step = 0
    rewards = []
    while not done and step < max_steps:
        actions = []
        for i in range(env.num_agents):
            if name == "PG-NoPS":
                a = (
                    policies[i].choose_greedy(state[i])
                    if eval_mode
                    else policies[i].choose_action(state[i])
                )
            else:
                a = (
                    policies.choose_greedy(i, state[i])
                    if eval_mode
                    else policies.choose_action(i, state[i])
                )
            actions.append(a)
        next_state, rs, done, info = env.step(jnp.array(actions))
        if not eval_mode:
            if name == "PG-NoPS":
                for i, p in enumerate(policies):
                    p.store_reward(rs[i], done)
            else:
                for r in rs:
                    policies.store_reward(r, done)
        rewards.append(rs)
        state = next_state
        step += 1
    converged = info.get("converged", False) if info else False
    return np.array(rewards), converged, step


def train_and_evaluate(
    game_type,
    num_agents,
    num_foods,
    max_train_steps,
    eval_episodes,
    max_steps,
    batch_size,
    seed,
    hidden_size,
    eval_interval,
    compute_grad_conflict,
    optim,
    hypernet_config=None,
):
    state_size = num_agents * num_foods
    if hypernet_config is None:
        hypernet_config = {}

    use_agent_id_embeddings = hypernet_config.get("use_agent_id_embeddings", True)
    embedding_dim = hypernet_config.get("embedding_dim", None)
    hypernet_hidden_dims = hypernet_config.get("hidden_dims", [16])
    use_custom_init = hypernet_config.get("use_custom_init", True)
    init_type = hypernet_config.get("init_type", "orthogonal")
    embedding_init_scale = hypernet_config.get("embedding_init_scale", np.sqrt(2))

    target_updates = max_train_steps // batch_size

    pg_hypernet = PGHyperMARL(
        num_agents,
        state_size,
        hidden_size,
        num_foods,
        optim,
        seed,
        embedding_dim=embedding_dim,
        use_agent_id_embeddings=use_agent_id_embeddings,
        hypernet_hidden_dims=hypernet_hidden_dims,
        use_custom_init=use_custom_init,
        init_type=init_type,
        embedding_init_scale=embedding_init_scale,
        compute_grad_conflict=compute_grad_conflict,
        target_updates=target_updates,
    )
    wandb.log(
        {
            "num_params/PG-HyperMARL": pg_hypernet.num_params,
        }
    )

    all_policies = {"PG-HyperMARL": pg_hypernet}

    target_updates = max_train_steps // batch_size
    updates_done = {name: 0 for name in all_policies}

    # per‐policy buffers & counters
    steps = dict.fromkeys(all_policies, 0)
    batch_rewards = {k: [] for k in all_policies}
    ep_conv = {k: [] for k in all_policies}
    ep_len = {k: [] for k in all_policies}

    total_steps = 0
    eval_steps = 0
    episode_count = 0

    while any(updates_done[name] < target_updates for name in all_policies):
        episode_count += 1

        for name, policies in all_policies.items():
            # skip finished policies
            if updates_done[name] >= target_updates:
                continue

            # rollout one episode (same as before)
            env = MatrixGameWithState(game_type, num_agents, num_foods, max_steps)
            rewards, conv, taken = play_episode(env, policies, name, False, max_steps)

            ep_conv[name].append(conv)
            ep_len[name].append(taken)
            for rs in rewards:
                batch_rewards[name].extend(rs)
            steps[name] += taken
            total_steps += taken
            eval_steps += taken

            # → NEW: apply as many batch‐updates as you can, but never exceed target
            while steps[name] >= batch_size and updates_done[name] < target_updates:
                # 1) perform exactly one update on full batch
                if isinstance(policies, list):
                    for p in policies:
                        p.update_policy()
                    loss_val = np.mean([p.losses[-1] for p in policies])
                else:
                    policies.update_policy()
                    loss_val = policies.losses[-1]

                updates_done[name] += 1

                # 2) log this batch's metrics
                wandb.log(
                    {
                        f"train/{seed}/{name}/batch_reward": np.mean(
                            batch_rewards[name]
                        ),
                        f"train/{seed}/{name}/loss": loss_val,
                        f"train/{seed}/{name}/convergence_rate": np.mean(ep_conv[name]),
                        f"train/{seed}/{name}/avg_episode_length": np.mean(
                            ep_len[name]
                        ),
                        "total_steps": total_steps,
                        "episode_count": episode_count,
                    }
                )

                # 3) consume that batch and reset only those buffers
                steps[name] -= batch_size
                batch_rewards[name].clear()
                ep_conv[name].clear()
                ep_len[name].clear()

            if eval_steps >= eval_interval:
                for ename, epol in all_policies.items():
                    tr, cr, el = 0, 0, 0
                    for _ in range(eval_episodes):
                        e = MatrixGameWithState(
                            game_type, num_agents, num_foods, max_steps
                        )
                        rs, cv, stp = play_episode(e, epol, ename, True, max_steps)
                        tr += rs.mean()
                        cr += cv
                        el += stp
                    wandb.log(
                        {
                            f"eval/{seed}/{ename}/avg_reward": tr / eval_episodes,
                            f"eval/{seed}/{ename}/convergence_rate": cr / eval_episodes,
                            f"eval/{seed}/{ename}/avg_steps": el / eval_episodes,
                            "total_steps": total_steps,
                            "episode_count": episode_count,
                        }
                    )
                eval_steps -= eval_interval

    # final eval & logging
    final_results = {}
    for name, policies in all_policies.items():
        reward_list, conv_list, step_list = [], [], []
        for _ in range(eval_episodes):
            env = MatrixGameWithState(game_type, num_agents, num_foods, max_steps)
            rs, cv, stp = play_episode(env, policies, name, True, max_steps)
            # rs.shape = (episode_length, num_agents)
            # average over time and agents → one scalar per episode
            avg_ep_reward = rs.mean()
            reward_list.append(avg_ep_reward)
            conv_list.append(cv)
            step_list.append(stp)

        avg_r = np.mean(reward_list)
        std_r = np.std(reward_list)
        avg_c = np.mean(conv_list)
        std_c = np.std(conv_list)
        avg_s = np.mean(step_list)
        std_s = np.std(step_list)

        final_results[name] = (avg_r, avg_c, avg_s)
        wandb.log(
            {
                f"{name}/{seed}/eval_reward_mean": avg_r,
                f"{name}/{seed}/eval_reward_std": std_r,
                f"{name}/{seed}/convergence_mean": avg_c,
                f"{name}/{seed}/convergence_std": std_c,
                f"{name}/{seed}/steps_mean": avg_s,
                f"{name}/{seed}/steps_std": std_s,
            }
        )

    # Return None for non-HyperMARL algorithms
    return final_results, None, None, None, None, pg_hypernet


def run_multiple_seeds(
    game_type,
    num_agents,
    num_foods,
    num_seeds,
    max_train_steps,
    eval_episodes,
    max_steps,
    batch_size,
    hidden_size,
    eval_interval,
    compute_grad_conflict,
    optim,
    hypernet_config=None,
):
    all_results = []
    all_nops, all_fups, all_fups_id, all_fups_id_no_state, all_hypernets = (
        [],
        [],
        [],
        [],
        [],
    )

    for s in range(num_seeds):
        res, nops, fups, fups_id, fups_id_no_state, hypernet = train_and_evaluate(
            game_type,
            num_agents,
            num_foods,
            max_train_steps,
            eval_episodes,
            max_steps,
            batch_size,
            s,
            hidden_size,
            eval_interval,
            compute_grad_conflict,
            optim,
            hypernet_config,
        )
        all_results.append(res)
        all_nops.append(nops)
        all_fups.append(fups)
        all_fups_id.append(fups_id)
        all_fups_id_no_state.append(fups_id_no_state)
        all_hypernets.append(hypernet)

    # Comment out logging for non-HyperMARL algorithms
    log_all_grad_metrics(None, None, None, None, all_hypernets)
    log_all_stability_proxy(None, None, None, None, all_hypernets)
    log_raw_gradient_norms(None, None, None, None, all_hypernets)
    log_raw_gradient_conflicts(None, None, all_hypernets)
    return (
        all_results,
        all_nops,
        all_fups,
        all_fups_id,
        all_fups_id_no_state,
        all_hypernets,
    )


def aggregate_results(all_results):
    agg = {}
    for nm in all_results[0]:
        # Extract results (reward, convergence, steps)
        rewards = [r[nm][0] for r in all_results]
        convs = [r[nm][1] for r in all_results]
        steps = [r[nm][2] for r in all_results]

        n = len(rewards)  # Number of seeds
        reward_se = np.std(rewards) / np.sqrt(n)  # Standard error of rewards

        agg[nm] = (
            np.mean(rewards),
            np.std(rewards),  # Reward mean & std
            reward_se,  # Reward standard error
            np.mean(convs),
            np.std(convs),  # Convergence mean & std
            np.mean(steps),
            np.std(steps),  # Steps mean & std
            rewards,  # Raw reward values
        )
    return agg


@hydra.main(version_base=None, config_path="", config_name="config_stateful.yaml")
def main(cfg):
    wandb.init(
        project="spec_syn_game_stateful",
        name=f"{cfg.game_type}_{cfg.num_agents}",
        config=OmegaConf.to_container(cfg),
        reinit=True,
        tags=["spec_syn_game_stateful", "hypermarl"],
        save_code=True,
        # mode="disabled"
    )
    # Extract hypernet config
    hypernet_config = OmegaConf.to_container(cfg.get("hypernet", {}))
    game_type = GameType[cfg.game_type.upper()]
    res, nops, fups, fups_id, fups_id_no_state, hypernets = run_multiple_seeds(
        game_type,
        cfg.num_agents,
        cfg.num_foods,
        cfg.num_seeds,
        cfg.max_train_steps,
        cfg.eval_episodes,
        cfg.max_steps_per_episode,
        cfg.batch_size,
        cfg.hidden_size,
        cfg.eval_step_interval,
        cfg.compute_grad_conflict,
        {
            "optim": cfg.optim,
            "learning_rate": cfg.learning_rate,
            "max_grad_norm": cfg.max_grad_norm,
        },
        hypernet_config,
    )

    # Keep logging similar to original code but only for HyperMARL
    # Log raw time-series for each seed & policy
    # names = ['PG-NoPS', 'PG-FuPS', 'PG-FuPS+ID', 'PG-FuPS+ID-No-State','PG-HyperMARL']
    names = ["PG-HyperMARL"]
    for seed in range(cfg.num_seeds):
        for nm in names:
            entry = {"seed": seed, "policy": nm}
            if nm == "PG-NoPS":
                # Skip NoPS
                pass
            elif nm == "PG-FuPS":
                # Skip FuPS
                pass
            elif nm == "PG-FuPS+ID":
                # Skip FuPS+ID
                pass
            elif nm == "PG-FuPS+ID-No-State":
                # Skip FuPS+ID-No-State
                pass
            else:  # PG-HyperMARL
                entry.update(
                    {
                        "gradient_norms": np.array(
                            hypernets[seed].gradient_norms
                        ).tolist(),
                        "gradient_variances": np.array(
                            hypernets[seed].gradient_variances
                        ).tolist(),
                        "gradient_conflicts": np.array(
                            hypernets[seed].gradient_conflicts
                        ).tolist(),
                        "returns": hypernets[seed].all_returns,
                        "losses": hypernets[seed].losses,
                    }
                )

            # print(np.array(hypernets[seed].gradient_norms).tolist())
            wandb.log(entry)

    # Log aggregated results
    agg = aggregate_results(res)
    print("\nAggregated Results:")
    for nm, (ar, sr, se_r, ac, sc, ast, sst, raw_rewards) in agg.items():
        print(
            f"{nm}: reward={ar:.3f}±{sr:.3f} (SE:{se_r:.3f}), convergence={ac:.3f}±{sc:.3f}, steps={ast:.1f}±{sst:.1f}"
        )
        wandb.log(
            {
                f"{nm}/final_eval_reward_mean": ar,
                f"{nm}/final_eval_reward_std": sr,
                f"{nm}/final_eval_reward_se": se_r,
                f"{nm}/final_convergence_mean": ac,
                f"{nm}/final_convergence_std": sc,
                f"{nm}/final_steps_mean": ast,
                f"{nm}/final_steps_std": sst,
                f"{nm}/raw_rewards": raw_rewards,
            }
        )

    # Comment out plotting functions
    base = f"stateful_policy_{cfg.num_agents}_{cfg.game_type}"
    # plot_combined_results(res, nops, fups, fups_id, cfg.num_agents, cfg.max_train_steps, cfg.batch_size, base, use_returns=True,all_pg_fu_ps_plus_id_no_state=fups_id_no_state,all_pg_hypernet=hypernets)
    plot_gradient_conflicts(
        None,
        cfg.max_train_steps,
        cfg.batch_size,
        base,
        all_pg_fu_ps_plus_id_no_state=None,
        all_pg_hypernet=hypernets,
    )

    proxies = compute_stability_proxy(None, None, None, None, all_pg_hypernet=hypernets)
    plot_stability_proxy(proxies, base)

    wandb.finish()


if __name__ == "__main__":
    main()
