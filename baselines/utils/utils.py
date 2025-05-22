
from functools import partial
import os
import random
from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # do this before importing pylab to avoid display issues
from matplotlib import pyplot as plt
import numpy as np
import wandb

from baselines.utils.metrics import calculate_snd
from baselines.utils.wrappers import make_env


def random_unique_one_hot_matrix(n_samples, n_classes):
    """
    Generate a random one-hot encoded matrix of shape (n_samples, n_classes) with unique rows.
    
    Used for creating unique agent identifiers when needed by parameter sharing architectures.
    
    Args:
        n_samples (int): Number of samples (rows) in the matrix.
        n_classes (int): Number of classes (columns) in the matrix.
        
    Returns:
        numpy.ndarray: Random one-hot encoded matrix with unique rows.
    """
    # Create a list of unique one-hot encoded vectors
    one_hot_vectors = np.eye(n_classes, dtype=np.float32)
    
    # Randomly shuffle the one-hot vectors
    np.random.shuffle(one_hot_vectors)
    
    # Select the first n_samples unique one-hot vectors
    one_hot_matrix = one_hot_vectors[:n_samples]
    
    return one_hot_matrix


def log_videos(cfg, global_step):
    """
    Log evaluation videos to wandb for visualization.
    
    Args:
        cfg: Configuration dictionary with video path information
        global_step: Current training step for logging
    """
    import glob
    
    # Log videos to wandb
    video_location = f"./videos/{cfg['RUN_NAME']}/{global_step}"
    video_paths = glob.glob(os.path.join(video_location, "*.mp4"))
    for video_path in video_paths:
        wandb.log({"videos": wandb.Video(video_path)}, step=global_step)


def calculate_team_diversity(
    acting_fns,
    agent_state,
    keys,
    num_agents,
    param_sharing,
    centralised_critic=False,
    chunk_size=10000,
):
    """
    Calculate diversity metrics for a team of agents.
    
    Uses a pre-collected dataset of observations to measure how different agent policies
    are from each other, which quantifies specialization.
    
    Args:
        acting_fns: Functions to extract policy distributions
        agent_state: Agent parameters
        keys: JAX PRNGKeys
        num_agents: Number of agents in the team
        param_sharing: Whether parameters are shared across agents
        centralised_critic: Whether a centralized critic is used
        chunk_size: Size of chunks for batched processing
        
    Returns:
        None (metrics are logged to WandB)
    """
    # [num_samples, num_agents, obs_dim]
    file_path = "datasets/dispersion/combined_obs_compressed.npz"

    if os.path.exists(file_path):
        # [num_samples, num_agents, obs_dim]
        obs = jnp.load(file_path)["obs"]
    else:
        print(f"Error: The file '{file_path}' does not exist.")
        print("You can download the dataset from: [provide download link or instructions here]")
        return None  # or raise an exception, depending on how you want to handle this case
    
    # due to memory constraints, we will randomly sample 1e6 samples
    # Get the total number of observations
    total_obs = obs.shape[0]

    # Number of samples you want
    num_samples = int(1e6 // num_agents)

    # Create a random permutation of indices
    keys, subkey = jax.random.split(keys)
    indices = jax.random.permutation(subkey, total_obs)

    # Take the first 1e6 indices
    sampled_indices = indices[:num_samples]

    # Use these indices to sample from the original array
    obs_sample = obs[sampled_indices]

    # flatten agent dim
    obs_sample = obs_sample.reshape(-1, obs_sample.shape[-1])

    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import jensenshannon

    def total_variation_distance(p, q):
        """Calculate total variation distance between probability distributions."""
        return 0.5 * np.sum(np.abs(p - q))

    distance_fns = {"jensenshannon": jensenshannon, "tvd": total_variation_distance}
    
    # Calculate and log diversity metrics for each distance function
    for distance, distance_fn in distance_fns.items():
        total_snd = 0.0
        # Calculate SND in chunks to avoid memory issues
        num_chunks = obs_sample.shape[0] // chunk_size + int(
            obs_sample.shape[0] % chunk_size > 0
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, obs_sample.shape[0])
            obs_chunk = obs_sample[start_idx:end_idx]

            # Calculate SND for the chunk
            snd_chunk = calculate_snd(
                obs_chunk,
                acting_fns,
                agent_state,
                keys,
                num_agents,
                centralised_critic=centralised_critic,
                param_sharing=param_sharing,
                distance_fn=distance_fn,
            )
            total_snd += snd_chunk * (end_idx - start_idx)  # weight by the chunk size

        # Normalize the total SND by the total number of samples
        snd = total_snd / obs_sample.shape[0]

        print(distance, snd)

        wandb.log(
            {f"metrics/eval/distance_{distance}": snd},
        )


def log_eval_metrics(config, eval_metrics, run):
    """
    Log evaluation metrics to wandb.
    
    Processes evaluation metrics and creates visualizations for returns,
    episode lengths, and other performance indicators.
    
    Args:
        config: Configuration dictionary
        eval_metrics: Collection of evaluation metrics
        run: WandB run object
    """
    if not eval_metrics or config["WANDB_MODE"] != "online":
        return

    # Prepare data structures
    steps = []
    mean_returns = []
    sum_returns = []
    mean_lengths = []
    # Episode returns are the sum of rewards.
    # Similarly to https://github.com/facebookresearch/BenchMARL/blob/dc793b5dce2eeae94be03e7ca72b10b4b16c66db/benchmarl/experiment/logger.py#L111
    # we also calculate the mean/min/max reward per episode
    mean_rewards = []
    min_rewards = []
    max_rewards = []

    # Single pass through eval_metrics
    for step, step_eval_data in eval_metrics:
        episode_returns = np.array(step_eval_data["returns"])
        episode_lengths = np.array(step_eval_data["lengths"])

        steps.append(step)
        
        # Calculate mean and sum return across agents for each environment
        # if it is a common reward settings, this is just the reward returned by the env
        mean_return_across_episodes_per_env = np.mean(episode_returns, axis=1)
        sum_return_across_episodes_per_env = np.sum(episode_returns, axis=1)

        # Calculate average of mean and sum returns across all environments
        avg_mean_return_across_episodes = np.mean(mean_return_across_episodes_per_env)
        avg_sum_return_across_episodes = np.mean(sum_return_across_episodes_per_env)

        mean_returns.append(avg_mean_return_across_episodes)
        sum_returns.append(avg_sum_return_across_episodes)
        mean_lengths.append(np.mean(episode_lengths))
        
        # Calculate mean, min, and max rewards
        mean_reward = np.mean(mean_return_across_episodes_per_env / episode_lengths)
        min_reward = mean_return_across_episodes_per_env.min()
        max_reward = mean_return_across_episodes_per_env.max()

        mean_rewards.append(mean_reward)
        min_rewards.append(min_reward)
        max_rewards.append(max_reward)

        # Log at each evaluation step as requested
        run.log(
            {
                "metrics/eval/mean_episode_return_across_all_evals_mean_across_agents": avg_mean_return_across_episodes,
                "metrics/eval/mean_episode_return_across_all_evals_sum_across_agents": avg_sum_return_across_episodes,
                "metrics/eval/mean_episode_length": np.mean(episode_lengths),
                "metrics/eval/mean_reward": mean_reward,
                "metrics/eval/min_reward": min_reward,
                "metrics/eval/max_reward": max_reward,
                "timestep": step,
            },
        )

    # Create summary table
    eval_table = wandb.Table(
        columns=["Step", "Mean Return", "Sum Return", "Mean Length"]
    )
    for i in range(len(steps)):
        eval_table.add_data(steps[i], mean_returns[i], sum_returns[i], mean_lengths[i])

    # Prepare log data
    log_data = {
        "eval/metrics_summary": eval_table,
        "eval/mean_return_over_time": wandb.plot.line(
            eval_table, "Step", "Mean Return", title="Mean Return Over Time"
        ),
        "eval/sum_return_over_time": wandb.plot.line(
            eval_table, "Step", "Sum Return", title="Sum Return Over Time"
        ),
        "eval/mean_length_over_time": wandb.plot.line(
            eval_table, "Step", "Mean Length", title="Mean Episode Length Over Time"
        ),
    }

    # Log videos if captured
    if config.get("CAPTURE_VIDEO_INTERVAL", None):
        log_videos(config, steps[-1])

    # Single wandb.log call at the end for summary data
    run.log(log_data)


def log_train_metrics(config, metrics, run, agent_id=None):
    """
    Log training metrics to wandb.
    
    Creates visualizations for various training metrics including losses,
    returns, gradients, and other algorithm-specific information.
    
    Args:
        config: Configuration dictionary
        metrics: Dictionary of training metrics
        run: WandB run object
        agent_id: Optional agent ID for multi-agent settings
    """
    if metrics.get("returned_episode_returns") is not None:
        plt.plot(metrics["returned_episode_returns"])
        plt.savefig(f"ippo_ff_{config['ENV_NAME']}.png")
        plt.xlabel("Updates")
        plt.ylabel("Returns")
        plt.title(f"IPPO-FF={config['ENV_NAME']}")
        plt.close()

    # Define the list of metrics to log
    metrics_to_log = [
        "total_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "ratio",
        "returned_episode_returns",
        "total_grad_mean",
        "total_grad_var",
        "total_grad_norm",
    ]

    if config.get("COMPUTE_POLICY_GRADIENT_NORM"):
        metrics_to_log += [
            "actor_grad_mean",
            "actor_grad_var",
            "actor_grad_norm",
            "critic_grad_mean",
            "critic_grad_var",
            "critic_grad_norm",
            "entropy_grad_mean",
            "entropy_grad_var",
            "entropy_grad_norm",
        ]

    if metrics.get("total_loss") is not None:
        # Create loss table
        updates_x = jnp.arange(metrics["total_loss"].shape[0])
        loss_data = [updates_x]
        loss_columns = ["updates"]

        for metric in metrics_to_log:
            if metric in metrics and metric != "returned_episode_returns":
                loss_data.append(metrics[metric])
                loss_columns.append(metric)

        loss_table = jnp.stack(loss_data, axis=1)
        loss_table = wandb.Table(data=loss_table.tolist(), columns=loss_columns)

        # Create returns table
        if "returned_episode_returns" in metrics:
            updates_x = jnp.arange(metrics["returned_episode_returns"].shape[0])
            returns_table = jnp.stack(
                [updates_x, metrics["returned_episode_returns"]], axis=1
            )
            returns_table = wandb.Table(
                data=returns_table.tolist(), columns=["updates", "returns"]
            )

        # Log data
        log_data = {}

        def prefix_key(key):
            return f"agent_{agent_id}/{key}" if agent_id else key

        if "returned_episode_returns" in metrics:
            log_data[prefix_key("returns_plot")] = wandb.plot.line(
                returns_table, "updates", "returns", title="returns_vs_updates"
            )
            log_data[prefix_key("returns")] = metrics["returned_episode_returns"].mean()

        for metric in metrics_to_log:
            if metric in loss_columns and metric != "returned_episode_returns":
                log_data[prefix_key(f"{metric}_plot")] = wandb.plot.line(
                    loss_table, "updates", metric, title=f"{metric}_vs_updates"
                )
                log_data[prefix_key(f"{metric}_mean_across_updates")] = metrics[
                    metric
                ].mean()

        run.log(log_data)

    eval_only = config.get("EVAL_ONLY", False)

    if config["WANDB_MODE"] == "online" and not eval_only:
        print("Pushing checkpoints to WandB")
        artifact = wandb.Artifact(
            name=f'checkpoint_{config["RUN_NAME"]}', type="checkpoint"
        )
        artifact.add_dir(local_path=config["CHP_DIR"])
        run.log_artifact(artifact)  # type: ignore


def create_action_functions(low, high):
    """
    Creates simple, robust action handling functions.
    
    Useful for continuous action spaces where actions need to be scaled
    between different ranges for training and environment interactions.
    
    Args:
        low (array-like): The lower bound of the action space.
        high (array-like): The upper bound of the action space.
        
    Returns:
        tuple: (clip_actions, scale_actions, unscale_actions) jitted functions.
    """
    low = jnp.array(low)
    high = jnp.array(high)

    @jax.jit
    def clip_actions(actions):
        """Clips actions to be within the specified range."""
        return jnp.clip(actions, low, high)

    @jax.jit
    def scale_actions(actions):
        """Scales actions from [low, high] to [-1, 1]."""
        return 2.0 * (actions - low) / (high - low) - 1.0

    @jax.jit
    def unscale_actions(scaled_actions):
        """Unscales actions from [-1, 1] to [low, high]."""
        return low + (scaled_actions + 1.0) * 0.5 * (high - low)

    return clip_actions, scale_actions, unscale_actions


def extract_and_save_agent_embeddings(params, step, config, run):
    """
    Extract and save agent embeddings from trained model parameters.
    
    Useful for analyzing how agents are differentiated in hypernetwork and
    other agent-specific parameter generating architectures.
    
    Args:
        params: Model parameters containing agent embeddings
        step: Current training step
        config: Configuration dictionary
        run: WandB run object
        
    Returns:
        None
    """
    # Extract embeddings
    if "agent_embeddings" not in params['params']:
        return 
    
    agent_embeddings = params['params']['agent_embeddings']
    
    # Save locally
    local_path = f"{config['CHP_DIR']}/agent_embeddings_step_{step}.npy"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    np.save(local_path, agent_embeddings)
    
    # Log to wandb
    embeddings_table = wandb.Table(
        data=agent_embeddings.tolist(), 
        columns=[f"dim_{i}" for i in range(agent_embeddings.shape[1])]
    )
    run.log({f"agent_embeddings_step_{step}": embeddings_table})
    
    # You can also log the raw numpy array, which can be downloaded later
    embedding_artifact = wandb.Artifact(f"agent_embeddings_{step}", type="embeddings")
    embedding_artifact.add_file(local_path)
    run.log_artifact(embedding_artifact)
    
    print(f"Agent embeddings saved locally and to wandb at step {step}")