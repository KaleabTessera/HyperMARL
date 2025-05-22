import distrax
import jax
import jax.numpy as jnp


def get_update_fn(config, network):
    """
    Create a PPO update function for the given network and configuration.
    
    This function factory returns a JAX-jitted function that calculates advantages
    and updates the policy network in a single optimized call.
    
    Args:
        config: Dictionary containing PPO hyperparameters
        network: Network architecture to be updated
        
    Returns:
        calculate_advantage_and_update_mappo: Jitted update function
    """
    def calculate_gradient_metrics(grads):
        """
        Calculate statistics about gradients for monitoring.
        
        Flattens gradient tree and computes mean, variance, and norm.
        
        Args:
            grads: PyTree of gradients
            
        Returns:
            mean_grad: Mean of all gradients
            variance: Variance of gradients
            norm: L2 norm of gradients
        """
        # Flatten the gradients
        flat_grads = jax.tree_util.tree_leaves(grads)
        flat_grads = [g.ravel() for g in flat_grads if g is not None]
        flat_grads = jnp.concatenate(flat_grads)

        # Calculate mean
        mean_grad = jnp.mean(flat_grads)

        # Calculate variance
        variance = jnp.mean((flat_grads - mean_grad) ** 2)

        # Calculate L2 norm
        norm = jnp.linalg.norm(flat_grads)

        return mean_grad, variance, norm

    @jax.jit
    def calculate_advantage_and_update_mappo(
        traj_batch, last_obs, last_global_obs, train_state, rng
    ):
        """
        Calculate advantages and update the policy network.
        
        This function handles the entire PPO update process:
        1. Calculate advantages using GAE
        2. Prepare minibatches
        3. Update policy for multiple epochs
        4. Track metrics
        
        Args:
            traj_batch: Batch of collected trajectories
            last_obs: Final observations
            last_global_obs: Final global observations 
            train_state: Current training state
            rng: JAX PRNGKey for randomness
            
        Returns:
            update_state: Updated state after training
            loss_info: Dictionary of training metrics
        """
        def _calculate_gae(traj_batch, last_val):
            """
            Calculate Generalized Advantage Estimation.
            
            Computes advantages and value targets using GAE.
            
            Args:
                traj_batch: Batch of transitions
                last_val: Value estimate for final state
                
            Returns:
                advantages: Advantage estimates
                targets: Value function targets
            """
            def _get_advantages(gae_and_next_value, transition):
                """Process a single transition in reverse order."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                # TD error with discounting
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                # GAE formula
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            # Reverse scan through trajectory to compute advantages
            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=8,  # Unroll for better XLA compilation
            )
            # Value targets are advantages plus value estimates
            return advantages, advantages + traj_batch.value

        # Get value estimate for final observation (bootstrapping)
        _, last_val = network.apply(train_state.params, last_obs, last_global_obs)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            """
            Perform a single PPO update epoch.
            
            Shuffles data and processes multiple minibatches for one epoch.
            
            Args:
                update_state: Current update state
                unused: Unused parameter for JAX scan
                
            Returns:
                update_state: Updated state
                loss_info: Loss metrics for this epoch
            """
            def _update_minbatch(train_state, batch_info):
                """
                Update network on a single minibatch.
                
                Computes losses and applies gradients.
                
                Args:
                    train_state: Current train state
                    batch_info: Tuple of (traj_batch, advantages, targets)
                    
                Returns:
                    train_state: Updated train state
                    loss_info: Dictionary of loss metrics
                """
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    """
                    Compute PPO loss function.
                    
                    Calculates actor loss (clipped surrogate objective),
                    critic loss (clipped value error), and entropy bonus.
                    
                    Args:
                        params: Network parameters
                        traj_batch: Batch of transitions
                        gae: Generalized advantage estimates
                        targets: Value function targets
                        
                    Returns:
                        total_loss: Combined loss to minimize
                        aux: Tuple of component losses
                    """
                    # Forward pass through network
                    actor_logits, value = network.apply(
                        params, traj_batch.obs, traj_batch.global_obs
                    )
                    pi = distrax.Categorical(logits=actor_logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS WITH CLIPPING
                    # Clip value function updates (PPO2 style)
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    # Compute losses for both unclipped and clipped values
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    # Take maximum as pessimistic bound
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS (PPO CLIPPED SURROGATE)
                    # Policy ratio (new policy / old policy)
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Normalize advantages for stability
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    # Standard PPO surrogate objective
                    loss_actor1 = ratio * gae
                    # Clipped surrogate objective
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    # Take minimum for pessimistic bound
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    
                    # Calculate policy entropy for regularization
                    entropy = pi.entropy().mean()

                    # Combine all losses
                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy, ratio)

                # Calculate gradients with respect to loss
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )

                loss_info = {}
                # Optionally compute gradient statistics for individual components
                if config.get("COMPUTE_POLICY_GRADIENT_NORM"):
                    # Calculate gradients for each loss component separately
                    actor_grad_fn = jax.grad(
                        lambda params: _loss_fn(
                            params, traj_batch, advantages, targets
                        )[1][1]  # Extract actor loss
                    )
                    critic_grad_fn = jax.grad(
                        lambda params: _loss_fn(
                            params, traj_batch, advantages, targets
                        )[1][0]  # Extract critic loss
                    )
                    entropy_grad_fn = jax.grad(
                        lambda params: _loss_fn(
                            params, traj_batch, advantages, targets
                        )[1][2]  # Extract entropy
                    )

                    # Apply gradient functions
                    actor_grads = actor_grad_fn(train_state.params)
                    critic_grads = critic_grad_fn(train_state.params)
                    entropy_grads = entropy_grad_fn(train_state.params)

                    # Calculate metrics for each gradient component
                    (
                        loss_info["actor_grad_mean"],
                        loss_info["actor_grad_var"],
                        loss_info["actor_grad_norm"],
                    ) = calculate_gradient_metrics(actor_grads)
                    (
                        loss_info["critic_grad_mean"],
                        loss_info["critic_grad_var"],
                        loss_info["critic_grad_norm"],
                    ) = calculate_gradient_metrics(critic_grads)
                    (
                        loss_info["entropy_grad_mean"],
                        loss_info["entropy_grad_var"],
                        loss_info["entropy_grad_norm"],
                    ) = calculate_gradient_metrics(entropy_grads)

                # Apply gradients to update parameters
                train_state = train_state.apply_gradients(grads=grads)

                # Calculate metrics for total gradients
                grad_mean, grad_var, grad_norm = calculate_gradient_metrics(grads)

                # Collect all metrics
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

            # Unpack current update state
            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            
            # Calculate batch size
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            
            # Reshape data for processing
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            
            # Shuffle data using permutation
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
            
            # Process all minibatches
            train_state, loss_info = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            
            # Return updated state
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        # Initialize update state
        update_state = (train_state, traj_batch, advantages, targets, rng)
        
        # Run multiple epochs of PPO updates
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        return update_state, loss_info

    return calculate_advantage_and_update_mappo