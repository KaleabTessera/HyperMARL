# NoPS, FuPS and FuPS+ID baselines for the matrix game with state
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
        lr,
        seed=0,
        gamma=0.99,
        target_updates=None,
    ):
        self.net = PolicyNetwork(input_size, hidden_size, output_size)
        self.key = random.PRNGKey(seed)
        dummy = jnp.zeros((1, input_size))
        self.params = self.net.init(self.key, dummy)
        # add clip to avoid NaN gradients for all methods
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.sgd(lr))
        # self.opt = optax.sgd(lr)
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
            print("how sway" * 10)
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


# ------------------------------------------------------------
# Play and training loops with logging and multi-seed support
# ------------------------------------------------------------


def play_episode(env, policies, name, eval_mode=False, max_steps=100):
    state = env.reset()
    # print(state)
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
):

    target_updates = max_train_steps // batch_size

    state_size = num_agents * num_foods
    pg_no_ps = [
        PGNoPS(
            state_size,
            hidden_size // num_agents,
            num_foods,
            0.01,
            seed + i,
            target_updates=target_updates,
        )
        for i in range(num_agents)
    ]
    pg_fu_ps = PGFuPS(
        num_agents,
        state_size,
        hidden_size,
        num_foods,
        0.01,
        seed,
        use_state=True,
        target_updates=target_updates,
    )  # FuPS has to use state since it has no other input

    # pg fups with id and state
    pg_id = PGFuPS(
        num_agents,
        state_size + num_agents,
        hidden_size,
        num_foods,
        0.01,
        seed,
        use_agent_id=True,
        use_state=True,
        compute_conflicts=compute_grad_conflict,
        target_updates=target_updates,
    )
    # pg fups with just ids
    pg_id_no_state = PGFuPS(
        num_agents,
        num_agents,
        hidden_size,
        num_foods,
        0.01,
        seed,
        use_agent_id=True,
        use_state=False,
        compute_conflicts=compute_grad_conflict,
        target_updates=target_updates,
    )
    wandb.log(
        {
            "num_params/PG-NoPS": sum(p.num_params for p in pg_no_ps),
            "num_params/PG-FuPS": pg_fu_ps.num_params,
            "num_params/PG-FuPS+ID": pg_id.num_params,
            "num_params/PG-FuPS+ID-No-State": pg_id_no_state.num_params,
        }
    )

    all_policies = {
        "PG-NoPS": pg_no_ps,
        "PG-FuPS": pg_fu_ps,
        "PG-FuPS+ID": pg_id,
        "PG-FuPS+ID-No-State": pg_id_no_state,
    }

    updates_done = {name: 0 for name in all_policies}

    # per‐policy buffers & counters
    steps = dict.fromkeys(all_policies, 0)
    batch_rewards = {k: [] for k in all_policies}
    ep_conv = {k: [] for k in all_policies}
    ep_len = {k: [] for k in all_policies}

    total_steps = 0
    eval_steps = 0
    episode_count = 0

    # → NEW: loop until every policy has done target_updates
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

                # 2) log this batch’s metrics
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

            # → existing eval logging (unchanged)
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

    return final_results, pg_no_ps, pg_fu_ps, pg_id, pg_id_no_state


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
):
    all_results = []
    all_nops, all_fups, all_fups_id, all_fups_id_no_state = [], [], [], []

    for s in range(num_seeds):
        res, nops, fups, fups_id, fups_id_no_state = train_and_evaluate(
            game_type,
            num_agents,
            num_foods,
            max_train_steps,  # still interpreted as total_updates*batch_size
            eval_episodes,
            max_steps,
            batch_size,
            s,
            hidden_size,
            eval_interval,
            compute_grad_conflict,
        )
        all_results.append(res)
        all_nops.append(nops)
        all_fups.append(fups)
        all_fups_id.append(fups_id)
        all_fups_id_no_state.append(fups_id_no_state)

    # → external logging & plotting unchanged
    log_all_grad_metrics(all_nops, all_fups, all_fups_id, all_fups_id_no_state)
    log_all_stability_proxy(all_nops, all_fups, all_fups_id, all_fups_id_no_state)
    log_raw_gradient_norms(all_nops, all_fups, all_fups_id, all_fups_id_no_state)
    # After training is complete
    log_raw_gradient_conflicts(all_fups_id, all_fups_id_no_state)
    return all_results, all_nops, all_fups, all_fups_id, all_fups_id_no_state


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
        tags=["spec_syn_game_stateful"],
        save_code=True,
        # mode="disabled"
    )
    game_type = GameType[cfg.game_type.upper()]
    res, nops, fups, fups_id, fups_id_no_state = run_multiple_seeds(
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
    )

    # Keep logging similar to original code
    # Log raw time-series for each seed & policy
    names = ["PG-NoPS", "PG-FuPS", "PG-FuPS+ID", "PG-FuPS+ID-No-State"]
    for seed in range(cfg.num_seeds):
        for nm in names:
            entry = {"seed": seed, "policy": nm}
            if nm == "PG-NoPS":
                # average across agents
                norms = np.mean(
                    [policy.gradient_norms for policy in nops[seed]], axis=0
                )
                vars = np.mean(
                    [policy.gradient_variances for policy in nops[seed]], axis=0
                )
                rets = np.mean([policy.all_returns for policy in nops[seed]], axis=0)
                loss = np.mean([policy.losses for policy in nops[seed]], axis=0)
                entry.update(
                    {
                        "gradient_norms": norms.tolist(),
                        "gradient_variances": vars.tolist(),
                        "returns": rets.tolist(),
                        "losses": loss.tolist(),
                    }
                )
            elif nm == "PG-FuPS":
                entry.update(
                    {
                        "gradient_norms": np.array(fups[seed].gradient_norms).tolist(),
                        "gradient_variances": np.array(
                            fups[seed].gradient_variances
                        ).tolist(),
                        "returns": fups[seed].all_returns,
                        "losses": fups[seed].losses,
                    }
                )
            elif nm == "PG-FuPS+ID":
                entry.update(
                    {
                        "gradient_norms": np.array(
                            fups_id[seed].gradient_norms
                        ).tolist(),
                        "gradient_variances": np.array(
                            fups_id[seed].gradient_variances
                        ).tolist(),
                        "gradient_conflicts": np.array(
                            fups_id[seed].gradient_conflicts
                        ).tolist(),
                        "returns": fups_id[seed].all_returns,
                        "losses": fups_id[seed].losses,
                    }
                )
            else:
                entry.update(
                    {
                        "gradient_norms": np.array(
                            fups_id_no_state[seed].gradient_norms
                        ).tolist(),
                        "gradient_variances": np.array(
                            fups_id_no_state[seed].gradient_variances
                        ).tolist(),
                        "gradient_conflicts": np.array(
                            fups_id_no_state[seed].gradient_conflicts
                        ).tolist(),
                        "returns": fups_id_no_state[seed].all_returns,
                        "losses": fups_id_no_state[seed].losses,
                    }
                )
            wandb.log(entry)

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

    # Generate plots
    base = f"stateful_policy_{cfg.num_agents}_{cfg.game_type}"
    plot_combined_results(
        res,
        nops,
        fups,
        fups_id,
        cfg.num_agents,
        cfg.max_train_steps,
        cfg.batch_size,
        base,
        use_returns=True,
        all_pg_fu_ps_plus_id_no_state=fups_id_no_state,
    )
    plot_gradient_conflicts(
        fups_id,
        cfg.max_train_steps,
        cfg.batch_size,
        base,
        all_pg_fu_ps_plus_id_no_state=fups_id_no_state,
    )

    proxies = compute_stability_proxy(nops, fups, fups_id, fups_id_no_state)
    plot_stability_proxy(proxies, base)

    wandb.finish()


if __name__ == "__main__":
    main()
