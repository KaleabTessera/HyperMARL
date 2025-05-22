"""
IPPO with Recurrent Networks for SMAX

Initially built off this version https://github.com/FLAIROx/JaxMARL/blob/main/baselines/IPPO/ippo_rnn_smax.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf

from baselines.utils.smax_utils import log_experiment_results
from jaxmarl.wrappers.baselines import SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

import wandb
import functools
import matplotlib.pyplot as plt


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
    Actor-Critic network with recurrent layers for handling sequential data.
    
    Uses GRU cells to maintain memory of past observations, enabling the network
    to handle partially observable environments more effectively.
    """
    action_dim: Sequence[int]  # Dimension of action space
    config: Dict               # Configuration dictionary

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
        
        # Initial observation embedding
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # Process sequence with RNN
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Actor network head
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], 
            kernel_init=orthogonal(2), 
            bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(actor_mean)
        
        # Mask unavailable actions with large negative values
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        # Create categorical distribution for discrete actions
        pi = distrax.Categorical(logits=action_logits)

        # Critic network head
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], 
            kernel_init=orthogonal(2), 
            bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0)
        )(critic)

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

    # Apply wrapper
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
        network = ActorCriticRNN(
            env.action_space(env.agents[0]).n, 
            config=config
        )
        
        # Initialize network parameters
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
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
        # Create test environment
        test_env = SMAXLogWrapper(HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"]))
        
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
            def callback(metric, original_seed):
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
    name = f"ippo_rnn_shared_smax_org_{config['MAP_NAME']}_{now:%Y-%m-%d_%H-%M-%S}"
    tags = ["IPPO", "RNN","Baseline"] if config.get("EXP_TAGS") is None else config.get("EXP_TAGS")
    
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