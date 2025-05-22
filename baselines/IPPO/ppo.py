import distrax
import jax
import jax.numpy as jnp
import optax


def calculate_gradient_metrics(grads):
    """
    Calculate metrics to analyze gradient behavior.
    
    Args:
        grads: Gradients from optimization step
        
    Returns:
        mean_grad: Mean gradient value
        variance: Variance of gradients
        norm: Euclidean norm of gradients
    """
    # Flatten the gradients
    flat_grads = jax.tree_util.tree_leaves(grads)
    flat_grads = [g.ravel() for g in flat_grads if g is not None]
    flat_grads = jnp.concatenate(flat_grads)

    # Calculate mean
    mean_grad = jnp.mean(flat_grads)

    # Calculate variance
    variance = jnp.mean((flat_grads - mean_grad) ** 2)

    # Calculate norm
    norm = jnp.linalg.norm(flat_grads)

    return mean_grad, variance, norm


def get_update_fn_ff_ppo(config, network):
    """
    Create a PPO update function for feedforward networks.
    
    Args:
        config: Configuration dictionary with PPO parameters
        network: Network model to optimize
        
    Returns:
        Function that calculates advantages and updates policy
    """
    @jax.jit
    def calculate_advantage_and_update_ppo(traj_batch, last_obs, train_state, rng):
        """
        Calculate advantages and update policy parameters.
        
        Args:
            traj_batch: Batch of trajectory data
            last_obs: Final observations
            train_state: Current training state
            rng: Random number generator key
            
        Returns:
            Updated state and loss metrics
        """
        def _calculate_gae(traj_batch, last_val):
            """
            Calculate Generalized Advantage Estimation.
            
            Args:
                traj_batch: Batch of trajectory data
                last_val: Value estimates for final observations
                
            Returns:
                advantages: Advantage estimates
                targets: Value function targets
            """
            def _get_advantages(gae_and_next_value, transition):
                """Calculate advantages for a single timestep."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                # unroll=8,
            )
            return advantages, advantages + traj_batch.value

        # Get final value estimates
        _, last_val = network.apply(train_state.params, last_obs)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            """
            Perform a single PPO epoch.
            
            Args:
                update_state: Current update state
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated state and loss metrics
            """
            def _update_minbatch(train_state, batch_info):
                """Update policy on a single minibatch."""
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    """PPO loss function."""
                    # RERUN NETWORK
                    actor_output, value = network.apply(params, traj_batch.obs)
                    if config["ACTION_SPACE_TYPE"] == "discrete":
                        pi = distrax.Categorical(logits=actor_output)
                    else:
                        mean, log_std = actor_output
                        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS with clipping
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS with clipping
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                    # Combined loss
                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy, ratio)

                # Calculate gradients
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )

                loss_info = {}
                # Calculate additional gradient metrics if configured
                if config.get("COMPUTE_POLICY_GRADIENT_NORM"):
                    actor_grad_fn = jax.grad(
                        lambda params: _loss_fn(
                            params, traj_batch, advantages, targets
                        )[1][1]
                    )
                    critic_grad_fn = jax.grad(
                        lambda params: _loss_fn(
                            params, traj_batch, advantages, targets
                        )[1][0]
                    )
                    entropy_grad_fn = jax.grad(
                        lambda params: _loss_fn(
                            params, traj_batch, advantages, targets
                        )[1][2]
                    )

                    actor_grads = actor_grad_fn(train_state.params)
                    critic_grads = critic_grad_fn(train_state.params)
                    entropy_grads = entropy_grad_fn(train_state.params)

                    # Calculate gradient metrics for actor
                    (
                        loss_info["actor_grad_mean"],
                        loss_info["actor_grad_var"],
                        loss_info["actor_grad_norm"],
                    ) = calculate_gradient_metrics(actor_grads)
                    
                    # Calculate gradient metrics for critic
                    (
                        loss_info["critic_grad_mean"],
                        loss_info["critic_grad_var"],
                        loss_info["critic_grad_norm"],
                    ) = calculate_gradient_metrics(critic_grads)
                    
                    # Calculate gradient metrics for entropy
                    (
                        loss_info["entropy_grad_mean"],
                        loss_info["entropy_grad_var"],
                        loss_info["entropy_grad_norm"],
                    ) = calculate_gradient_metrics(entropy_grads)

                # Apply gradients to update parameters
                train_state = train_state.apply_gradients(grads=grads)

                # Calculate overall gradient metrics
                grad_mean, grad_var, grad_norm = calculate_gradient_metrics(grads)

                # Collect loss information for logging
                loss_info.update(
                    {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                        "total_grad_mean": grad_mean,
                        "total_grad_var": grad_var,
                        "total_grad_norm": grad_norm,
                    }
                )

                return train_state, loss_info

            # Unpack update state
            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            
            # Prepare data for minibatching
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            
            # Reshape trajectory data
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            
            # Shuffle data
            permutation = jax.random.permutation(_rng, batch_size)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            
            # Split into minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            
            # Update on all minibatches
            train_state, loss_info = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        # Initialize update state
        update_state = (train_state, traj_batch, advantages, targets, rng)
        
        # Run multiple epochs of updates
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        return update_state, loss_info

    return calculate_advantage_and_update_ppo


def get_update_fn_rec(config, network, hidden_state_size):
    """
    Create a PPO update function for recurrent networks.
    
    Args:
        config: Configuration dictionary with PPO parameters
        network: Recurrent network model to optimize
        hidden_state_size: Size of RNN hidden state
        
    Returns:
        Function that calculates advantages and updates policy for recurrent networks
    """
    @jax.jit
    def calculate_advantage_and_update_ppo(
        last_obs_batch,
        last_done,
        hstate,
        traj_batch,
        train_state,
        initial_hstate,
        rng,
    ):
        """
        Calculate advantages and update policy parameters for recurrent networks.
        
        Args:
            last_obs_batch: Final observations in batch form
            last_done: Final done flags
            hstate: Final RNN hidden state
            traj_batch: Batch of trajectory data
            train_state: Current training state
            initial_hstate: Initial RNN hidden state
            rng: Random number generator key
            
        Returns:
            Updated state and loss metrics
        """
        # CALCULATE ADVANTAGE
        ac_in = (
            last_obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
        )
        _, _, last_val = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze()

        def _calculate_gae(traj_batch, last_val):
            """
            Calculate Generalized Advantage Estimation.
            
            Args:
                traj_batch: Batch of trajectory data
                last_val: Value estimates for final observations
                
            Returns:
                advantages: Advantage estimates
                targets: Value function targets
            """
            def _get_advantages(gae_and_next_value, transition):
                """Calculate advantages for a single timestep."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            """
            Perform a single PPO epoch for recurrent networks.
            
            Args:
                update_state: Current update state
                unused: Unused parameter for JAX compatibility
                
            Returns:
                Updated state and loss metrics
            """
            def _update_minbatch(train_state, batch_info):
                """Update policy on a single minibatch."""
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    """PPO loss function for recurrent networks."""
                    # RERUN NETWORK
                    _, pi, value = network.apply(
                        params,
                        init_hstate.transpose(),
                        (traj_batch.obs, traj_batch.done),
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
                    ).mean(where=(1 - traj_batch.done))

                    # CALCULATE ACTOR LOSS with clipping
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                    entropy = pi.entropy().mean(where=(1 - traj_batch.done))

                    # Combined loss
                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                # Calculate gradients
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                
                # Apply gradients to update parameters
                train_state = train_state.apply_gradients(grads=grads)

                # Collect loss information for logging
                loss_info = {
                    "total_loss": total_loss[0],
                    "actor_loss": total_loss[1][1],
                    "critic_loss": total_loss[1][0],
                    "entropy": total_loss[1][2],
                }
                return train_state, loss_info

            # Unpack update state
            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)
            
            # Reshape hidden state for minibatching
            init_hstate = jnp.reshape(init_hstate, (-1, hidden_state_size))
            
            # Prepare data for minibatching
            batch = (
                init_hstate,
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            
            # Create permutation for hidden state dimension
            permutation = jax.random.permutation(_rng, hidden_state_size)

            # Shuffle data
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            # Split into minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            # Update on all minibatches
            train_state, loss_info = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, loss_info

        # Transform initial hidden state
        init_hstate = initial_hstate[None, :].squeeze().transpose()
        
        # Initialize update state
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        
        # Run multiple epochs of updates
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        
        return update_state, loss_info

    return calculate_advantage_and_update_ppo