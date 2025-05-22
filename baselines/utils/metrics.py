import distrax
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import wandb
import numpy as np
import jax.numpy as jnp
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    import pandas as pd
except ImportError:
    pd = None


def jensen_shannon_distance(p, q, axis=-1, base=2):
    """
    Compute the Jensen-Shannon distance between two probability distributions.
    
    This is a custom implementation that avoids issues in SciPy's implementation
    by correctly clipping minimum divergence values to zero.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        axis: Axis along which to compute distance
        base: Logarithm base to use (default: 2)
        
    Returns:
        Jensen-Shannon distance between p and q
    """
    # Ensure the input distributions are valid probabilities
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)

    # Normalize the distributions
    p /= np.sum(p, axis=axis, keepdims=True)
    q /= np.sum(q, axis=axis, keepdims=True)

    # Calculate the midpoint distribution
    m = 0.5 * (p + q)

    # Calculate the Jensen-Shannon divergence
    divergence = 0.5 * (
        entropy(p, m, axis=axis, base=base) + entropy(q, m, axis=axis, base=base)
    )

    # Clip small negative values to zero to prevent numerical issues
    divergence = np.clip(divergence, 0, None)

    # Calculate the Jensen-Shannon distance (square root of divergence)
    distance = np.sqrt(divergence)

    return distance


def calculate_metrics_with_and_without_agent_id(
    episode_obs, acting_fns, agent_state, _keys
):
    """
    Calculate metrics comparing policy behavior with correct vs. shuffled agent IDs.
    
    This function measures how agent behavior changes when agent identities are
    randomly reassigned, to quantify the degree of specialization.
    
    Args:
        episode_obs: Observations from evaluation episodes
        acting_fns: Functions to get actions from policies
        agent_state: Trained agent parameters
        _keys: JAX random keys
        
    Returns:
        None (metrics are logged to wandb)
    """
    # shape of obs [num_episodes*max_steps, num_agents, obs_dim]
    # e.g. mpe, 32 episodes, max steps 25, obs dim 21, 3 agents - [800,3,21]

    # Get episode information
    num_episodes = len(episode_obs)
    num_agents = episode_obs[0].shape[1]
    episode_lengths = [len(episode) for episode in episode_obs]
    max_steps = max(episode_lengths)

    if len(episode_lengths) != len(set(episode_lengths)):
        print(
            "Episode lengths are not the same across episodes - per step plots will be incorrect"
        )

    # Initialize lists to store metrics for each step
    all_entropy_diffs = []
    all_js_distances = []
    all_diff_action_percentages = []

    # Initialize lists to store metrics for each step
    entropy_diffs_per_step = [[] for _ in range(max_steps)]
    js_distances_per_step = [[] for _ in range(max_steps)]
    diff_action_percentages_per_step = [[] for _ in range(max_steps)]

    for episode in range(num_episodes):
        obs_in_episode = episode_obs[episode]
        for step in range(len(obs_in_episode)):
            step_obs = obs_in_episode[step]
            shuffled_obs = step_obs.copy()
            agent_ids = step_obs[:, -num_agents:].copy()

            # Flatten the agent IDs for shuffling
            flat_agent_ids = agent_ids.reshape(-1, num_agents)

            # Shuffle the agent IDs to simulate random agent assignment
            np.random.shuffle(flat_agent_ids)

            # Reshape back to the original shape and replace in shuffled observations
            shuffled_agent_ids = flat_agent_ids.reshape(agent_ids.shape)
            shuffled_obs[:, -num_agents:] = shuffled_agent_ids

            # Get policy outputs with correct agent IDs
            _, logits, probabilities_with_agent_id = acting_fns(
                agent_state, step_obs, _keys
            )

            # Get policy outputs with shuffled agent IDs
            _, shuffled_logits, probabilities_with_random_id = acting_fns(
                agent_state, shuffled_obs, _keys
            )

            # Calculate entropy per agent with correct agent IDs
            entropies_with_agent_ids = entropy(
                probabilities_with_agent_id, axis=-1, base=2
            )
            entropies_with_agent_ids = np.clip(
                entropies_with_agent_ids, 0, None
            )  # Ensure non-negative

            # Calculate entropy per agent with shuffled agent IDs
            entropies_with_random_ids = entropy(
                probabilities_with_random_id, axis=-1, base=2
            )
            entropies_with_random_ids = np.clip(
                entropies_with_random_ids, 0, None
            )  # Ensure non-negative

            # Calculate entropy difference (indicates how much specialization affects policy certainty)
            entropy_diff = np.mean(entropies_with_random_ids - entropies_with_agent_ids)
            entropy_diffs_per_step[step].append(entropy_diff)
            all_entropy_diffs.append(entropy_diff)

            # Calculate Jensen-Shannon distance between distributions
            # Indicates how different the policies are with correct vs. shuffled IDs
            js_dist = jensen_shannon_distance(
                probabilities_with_agent_id,
                probabilities_with_random_id,
                axis=-1,
                base=2,
            )
            js_dist = np.mean(js_dist)
            js_distances_per_step[step].append(js_dist)
            all_js_distances.append(js_dist)

            # Calculate percentage of different actions taken
            # Measures how often actions would change with incorrect agent IDs
            actions_with_agent_id = np.argmax(probabilities_with_agent_id, axis=-1)
            actions_with_random_id = np.argmax(probabilities_with_random_id, axis=-1)
            different_actions = np.sum(actions_with_agent_id != actions_with_random_id)
            total_actions = actions_with_agent_id.size
            diff_action_percentage = (different_actions / total_actions) * 100

            diff_action_percentages_per_step[step].append(diff_action_percentage)
            all_diff_action_percentages.append(diff_action_percentage)

            # Log step-specific metrics
            wandb.log(
                {
                    "entropy_diff_with_agent_ids_and_random_ids": entropy_diff,
                    "js_distances_with_agent_ids_and_random_ids": js_dist,
                    "episode": episode,
                    "step": step,
                    "entropy_with_agent_ids": entropies_with_agent_ids,
                    "entropy_with_random_ids": entropies_with_random_ids,
                    "diff_action_percentage_with_agent_ids_and_random_ids": diff_action_percentage,
                }
            )
            print(
                f"Episode: {episode}, Step: {step}, JS Distance: {js_dist}, Entropy Diff: {entropy_diff}"
            )

    # Calculate average metrics for each step across all episodes
    avg_entropy_diffs_per_step = [np.mean(diffs) for diffs in entropy_diffs_per_step]
    avg_js_distances_per_step = [
        np.mean(distances) for distances in js_distances_per_step
    ]
    avg_diff_action_percentages_per_step = [
        np.mean(percentages) for percentages in diff_action_percentages_per_step
    ]

    # Calculate standard deviations for uncertainty visualization
    std_entropy_diffs_per_step = [np.std(diffs) for diffs in entropy_diffs_per_step]
    std_js_distances_per_step = [
        np.std(distances) for distances in js_distances_per_step
    ]
    std_diff_action_percentages_per_step = [
        np.std(percentages) for percentages in diff_action_percentages_per_step
    ]

    # Log average metrics per step to wandb
    for step in range(max_steps):
        wandb.log(
            {
                "average_entropy_diff_per_step_with_agent_ids_and_random_ids": avg_entropy_diffs_per_step[
                    step
                ],
                "average_js_distance_per_step_with_agent_ids_and_random_ids": avg_js_distances_per_step[
                    step
                ],
                "average_diff_action_percentage_per_step_with_agent_ids_and_random_ids": avg_diff_action_percentages_per_step[
                    step
                ],
                "step": step,
                "std_error_entropy_diff_per_step": std_entropy_diffs_per_step[step],
                "std_error_js_distance_per_step": std_js_distances_per_step[step],
                "std_error_diff_action_percentage_per_step": std_diff_action_percentages_per_step[
                    step
                ],
            }
        )

    # Calculate average metrics across all steps and episodes
    avg_entropy_diff = np.mean(all_entropy_diffs)
    avg_js_distance = np.mean(all_js_distances)
    avg_diff_action_percentage = np.mean(all_diff_action_percentages)

    # Calculate standard errors for confidence intervals
    std_error_entropy_diff = np.std(all_entropy_diffs) / np.sqrt(len(all_entropy_diffs))
    std_error_js_distance = np.std(all_js_distances) / np.sqrt(len(all_js_distances))
    std_error_diff_action_percentage = np.std(all_diff_action_percentages) / np.sqrt(
        len(all_diff_action_percentages)
    )

    # Log aggregate metrics to wandb
    wandb.log(
        {
            "average_entropy_diff_with_agent_ids_and_random_ids": avg_entropy_diff,
            "std_error_entropy_diff_with_agent_ids_and_random_ids": std_error_entropy_diff,
            "average_js_distance_with_agent_ids_and_random_ids": avg_js_distance,
            "std_error_js_distance_with_agent_ids_and_random_ids": std_error_js_distance,
            "average_diff_action_percentage_with_agent_ids_and_random_ids": avg_diff_action_percentage,
            "std_error_diff_action_percentage_with_agent_ids_and_random_ids": std_error_diff_action_percentage,
        }
    )

    # Create visualizations for metric trends over time
    
    # Plot entropy differences across steps
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(max_steps),
        np.array(avg_entropy_diffs_per_step) - np.array(std_entropy_diffs_per_step),
        np.array(avg_entropy_diffs_per_step) + np.array(std_entropy_diffs_per_step),
        alpha=0.3,
        label="Average Entropy Difference",
    )
    plt.plot(range(max_steps), avg_entropy_diffs_per_step, label=None)
    plt.xlabel("Steps")
    plt.ylabel("Average Entropy Difference")
    plt.title("Average Entropy Difference Across Steps (with agent ids and random ids)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "average_entropy_difference_across_steps_with_agent_ids_and_random_ids.png"
    )
    wandb.log(
        {
            "average_entropy_difference_across_steps_with_agent_ids_and_random_ids": wandb.Image(
                "average_entropy_difference_across_steps_with_agent_ids_and_random_ids.png"
            )
        }
    )
    plt.close()

    # Plot different action percentages across steps
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(max_steps),
        np.array(avg_diff_action_percentages_per_step)
        - np.array(std_diff_action_percentages_per_step),
        np.array(avg_diff_action_percentages_per_step)
        + np.array(std_diff_action_percentages_per_step),
        alpha=0.3,
        label="Average Diff Actions Percentage",
    )
    plt.plot(range(max_steps), avg_diff_action_percentages_per_step, label=None)
    plt.xlabel("Steps")
    plt.ylabel("Average Diff Actions Percentage")
    plt.title(
        "Average Diff Actions Percentage Across Steps (with agent ids and random ids)"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("average_diff_actions_steps_with_agent_ids_and_random_ids.png")
    wandb.log(
        {
            "average_diff_actions_steps_with_agent_ids_and_random_ids": wandb.Image(
                "average_diff_actions_steps_with_agent_ids_and_random_ids.png"
            )
        }
    )
    plt.close()

    # Plot JS distances across steps
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(max_steps),
        np.array(avg_js_distances_per_step) - np.array(std_js_distances_per_step),
        np.array(avg_js_distances_per_step) + np.array(std_js_distances_per_step),
        alpha=0.3,
        label="Average JS Distance",
    )
    plt.plot(range(max_steps), avg_js_distances_per_step, label=None)
    plt.xlabel("Steps")
    plt.ylabel("Average Jensen-Shannon Distance")
    plt.title(
        "Average Jensen-Shannon Distance Across Steps (with agent ids and random ids)"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("average_js_distance_across_steps_with_agent_ids_and_random_ids.png")
    wandb.log(
        {
            "average_js_distance_across_steps_with_agent_ids_and_random_ids": wandb.Image(
                "average_js_distance_across_steps_with_agent_ids_and_random_ids.png"
            )
        }
    )
    plt.close()


def calculate_team_diversity_metrics(episode_obs, acting_fns, agent_state, _keys):
    """
    Calculate metrics measuring diversity among team members.
    
    This function quantifies how different the policies of team members are from
    each other when facing the same observations, providing insight into agent specialization.
    
    Args:
        episode_obs: Observations from evaluation episodes
        acting_fns: Functions to get actions from policies
        agent_state: Trained agent parameters
        _keys: JAX random keys
        
    Returns:
        None (metrics are logged to wandb)
    """
    # Get episode dimensions
    num_episodes = len(episode_obs)
    num_agents = episode_obs[0].shape[1]
    episode_lengths = [len(episode) for episode in episode_obs]
    max_steps = max(episode_lengths)

    if len(episode_lengths) != len(set(episode_lengths)):
        print(
            "Episode lengths are not the same across episodes - per step plots won't be super accurate"
        )

    # Initialize metric storage
    # Overall metrics across all steps and episodes
    all_entropy_diffs = []
    all_js_distances = []
    all_diff_action_percentages = []

    # Per-step metrics
    entropy_diffs_per_step = [[] for _ in range(max_steps)]
    js_distances_per_step = [[] for _ in range(max_steps)]
    diff_action_percentages_per_step = [[] for _ in range(max_steps)]

    # Agent-pair similarity matrices
    entropy_diff_matrix_across_time = []
    js_distance_matrix_across_time = []
    similarity_matrix_across_time = []

    # Action probabilities per agent over time
    agent_probabilities_over_time = [[] for _ in range(num_agents)]
    agent_probabilities_over_time_per_step = [
        [[] for _ in range(max_steps)] for _ in range(num_agents)
    ]

    # Process each episode and step
    for episode in range(num_episodes):
        obs_in_episode = episode_obs[episode]
        for step in range(len(obs_in_episode)):
            step_obs = obs_in_episode[step]

            # Check if parameters are shared across agents or not
            param_sharing = len(agent_state) == 1

            if param_sharing:
                # For parameter sharing: each agent ID processes all observations
                
                # Expand observations for all agents
                expanded_obs = np.concatenate([step_obs] * num_agents, axis=0)

                # Create duplicated agent IDs to simulate each agent seeing every observation
                duplicated_agent_ids = np.repeat(
                    step_obs[:, -num_agents:], num_agents, axis=0
                )

                # Apply agent IDs for expanded observations
                expanded_obs[:, -num_agents:] = duplicated_agent_ids
                
                # Get action probabilities
                _, logits, probabilities_with_agent_id = acting_fns(
                    agent_state, expanded_obs, _keys
                )

                # Split probabilities by agent
                probabilities_with_agent_id_per_agent = np.split(
                    probabilities_with_agent_id, num_agents, axis=0
                )
            else:
                # For separate parameters: each agent processes all observations independently
                
                # Create expanded observation sets for each agent
                expanded_obs = []
                for obs in step_obs:
                    # Repeat each observation for all agents
                    repeated_obs = np.tile(obs, (num_agents, 1))
                    expanded_obs.append(repeated_obs)
                
                # Get action probabilities for each agent separately
                probabilities_with_agent_id_per_agent = [[] for _ in range(num_agents)]
                for obs in expanded_obs:
                    # Get probabilities from all agents for the same observation
                    _, logits, probabilities_with_agent_id = acting_fns(
                        agent_state, obs, _keys
                    )
                    # Store each agent's policy output
                    for i in range(num_agents):
                        probabilities_with_agent_id_per_agent[i].append(
                            probabilities_with_agent_id[i]
                        )

                # Convert to arrays with shape [num_obs, num_actions]
                probabilities_with_agent_id_per_agent = [
                    np.array(agent_probs)
                    for agent_probs in probabilities_with_agent_id_per_agent
                ]

            # Store average probabilities for each agent
            for i in range(num_agents):
                # Average over all observations for this step
                avg_probs_per_agent = np.mean(
                    probabilities_with_agent_id_per_agent[i], axis=0
                )
                agent_probabilities_over_time[i].append(avg_probs_per_agent)
                agent_probabilities_over_time_per_step[i][step].append(
                    avg_probs_per_agent
                )

            # Initialize matrices for team metrics
            entropy_team_diffs = []
            js_team_distances = []
            diff_action_team_percentages = []

            entropy_diff_matrix = np.zeros((num_agents, num_agents))
            js_distance_matrix = np.zeros((num_agents, num_agents))
            similarity_matrix = np.zeros((num_agents, num_agents))

            # Calculate diversity metrics between each pair of agents
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # Compare policies between agents i and j
                    probabilities_i = probabilities_with_agent_id_per_agent[i]
                    probabilities_j = probabilities_with_agent_id_per_agent[j]

                    # Calculate entropy difference
                    entropy_i = entropy(probabilities_i, axis=-1, base=2)
                    entropy_j = entropy(probabilities_j, axis=-1, base=2)
                    entropy_diff = np.mean(np.abs(entropy_i - entropy_j))
                    entropy_team_diffs.append(entropy_diff)

                    # Calculate Jensen-Shannon distance
                    js_distance = jensen_shannon_distance(
                        probabilities_i, probabilities_j, axis=-1, base=2
                    )
                    js_distance = np.mean(js_distance)
                    js_team_distances.append(js_distance)

                    # Calculate percentage of different actions
                    actions_i = np.argmax(probabilities_i, axis=-1)
                    actions_j = np.argmax(probabilities_j, axis=-1)
                    different_actions = np.sum(actions_i != actions_j)
                    total_actions = actions_i.size
                    diff_action_percentage = (different_actions / total_actions) * 100
                    diff_action_team_percentages.append(diff_action_percentage)

                    # Store in matrices for heatmap visualization
                    entropy_diff_matrix[i, j] = entropy_diff
                    entropy_diff_matrix[j, i] = entropy_diff  # Fill both sides of matrix

                    js_distance_matrix[i, j] = js_distance
                    js_distance_matrix[j, i] = js_distance  # Symmetric

                    similarity_matrix[i, j] = diff_action_percentage
                    similarity_matrix[j, i] = diff_action_percentage  # Symmetric

            # Store matrices for analysis
            entropy_diff_matrix_across_time.append(entropy_diff_matrix)
            js_distance_matrix_across_time.append(js_distance_matrix)
            similarity_matrix_across_time.append(similarity_matrix)

            # Calculate team-level averages
            team_entropy_diff = np.mean(entropy_team_diffs)
            team_js_distance = np.mean(js_team_distances)
            team_diff_action_percentage = np.mean(diff_action_team_percentages)

            # Store metrics for this step
            entropy_diffs_per_step[step].append(team_entropy_diff)
            all_entropy_diffs.append(team_entropy_diff)

            js_distances_per_step[step].append(team_js_distance)
            all_js_distances.append(team_js_distance)

            diff_action_percentages_per_step[step].append(team_diff_action_percentage)
            all_diff_action_percentages.append(team_diff_action_percentage)

            # Log step metrics
            wandb.log(
                {
                    "entropy_diff": team_entropy_diff,
                    "js_distances": team_js_distance,
                    "episode": episode,
                    "step": step,
                    "diff_action_percentage": team_diff_action_percentage,
                }
            )

            print(
                f"Episode: {episode}, Step: {step}, JS Distance: {team_js_distance}, "
                f"Entropy Diff: {team_entropy_diff} "
                f"Diff Action Percentage: {team_diff_action_percentage}"
            )

    # ----------------- Average Metrics per step -----------------#
    # Calculate average metrics for each step across all episodes
    avg_entropy_diffs_per_step = [np.mean(diffs) for diffs in entropy_diffs_per_step]
    avg_js_distances_per_step = [
        np.mean(distances) for distances in js_distances_per_step
    ]
    avg_diff_action_percentages_per_step = [
        np.mean(percentages) for percentages in diff_action_percentages_per_step
    ]

    # Calculate standard deviations for uncertainty visualization
    std_entropy_diffs_per_step = [np.std(diffs) for diffs in entropy_diffs_per_step]
    std_js_distances_per_step = [
        np.std(distances) for distances in js_distances_per_step
    ]
    std_diff_action_percentages_per_step = [
        np.std(percentages) for percentages in diff_action_percentages_per_step
    ]

    # Log average metrics per step to wandb
    for step in range(max_steps):
        wandb.log(
            {
                "average_entropy_diff_per_step": avg_entropy_diffs_per_step[step],
                "average_js_distance_per_step": avg_js_distances_per_step[step],
                "average_diff_action_percentage_per_step": avg_diff_action_percentages_per_step[
                    step
                ],
                "step": step,
                "std_error_entropy_diff_per_step": std_entropy_diffs_per_step[step],
                "std_error_js_distance_per_step": std_js_distances_per_step[step],
                "std_error_diff_action_percentage_per_step": std_diff_action_percentages_per_step[
                    step
                ],
            }
        )

    # Plot trend visualizations
    
    # Entropy differences across steps
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(max_steps),
        np.array(avg_entropy_diffs_per_step) - np.array(std_entropy_diffs_per_step),
        np.array(avg_entropy_diffs_per_step) + np.array(std_entropy_diffs_per_step),
        alpha=0.3,
        label="Average Entropy Difference Across Agent Pairs",
    )
    plt.plot(range(max_steps), avg_entropy_diffs_per_step, label=None)
    plt.xlabel("Steps")
    plt.ylabel("Average Entropy Difference Across Agent Pairs")
    plt.title("Average Entropy Difference Across Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig("average_team_entropy_difference_across_steps.png")
    wandb.log(
        {
            "average_team_entropy_difference_across_steps": wandb.Image(
                "average_team_entropy_difference_across_steps.png"
            )
        }
    )
    plt.close()

    # Different actions percentage across steps
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(max_steps),
        np.array(avg_diff_action_percentages_per_step)
        - np.array(std_diff_action_percentages_per_step),
        np.array(avg_diff_action_percentages_per_step)
        + np.array(std_diff_action_percentages_per_step),
        alpha=0.3,
        label="Average Diff Actions Percentage Across Agent Pairs",
    )
    plt.plot(range(max_steps), avg_diff_action_percentages_per_step, label=None)
    plt.xlabel("Steps")
    plt.ylabel("Average Diff Actions Percentage Across Agent Pairs")
    plt.title("Average Diff Actions Percentage Across Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig("average_team_diff_actions_steps.png")
    wandb.log(
        {
            "average_team_diff_actions_steps": wandb.Image(
                "average_team_diff_actions_steps.png"
            )
        }
    )
    plt.close()

    # JS distances across steps
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(max_steps),
        np.array(avg_js_distances_per_step) - np.array(std_js_distances_per_step),
        np.array(avg_js_distances_per_step) + np.array(std_js_distances_per_step),
        alpha=0.3,
        label="Average JS Distance Across Agent Pairs",
    )
    plt.plot(range(max_steps), avg_js_distances_per_step, label=None)
    plt.xlabel("Steps")
    plt.ylabel("Average Jensen-Shannon Distance Across Agent Pairs")
    plt.title("Average Jensen-Shannon Distance Across Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig("average_team_js_distance_across_steps.png")
    wandb.log(
        {
            "average_team_js_distance_across_steps": wandb.Image(
                "average_team_js_distance_across_steps.png"
            )
        }
    )
    plt.close()

    # ----------------- Average Metrics across time -----------------#
    # Calculate average metrics across all steps and episodes
    avg_entropy_diff = np.mean(all_entropy_diffs)
    avg_js_distance = np.mean(all_js_distances)
    avg_diff_action_percentage = np.mean(all_diff_action_percentages)

    # Calculate standard errors for confidence intervals
    std_error_entropy_diff = np.std(all_entropy_diffs) / np.sqrt(len(all_entropy_diffs))
    std_error_js_distance = np.std(all_js_distances) / np.sqrt(len(all_js_distances))
    std_error_diff_action_percentage = np.std(all_diff_action_percentages) / np.sqrt(
        len(all_diff_action_percentages)
    )

    # Log average metrics to wandb
    wandb.log(
        {
            "average_team_entropy_diff": avg_entropy_diff,
            "std_error_team_entropy_diff": std_error_entropy_diff,
            "average_team_js_distance": avg_js_distance,
            "std_error_team_js_distance": std_error_js_distance,
            "average_team_diff_action_percentage": avg_diff_action_percentage,
            "std_error_team_diff_action_percentage": std_error_diff_action_percentage,
        }
    )

    # ----------------- Agent Similarity Heatmaps -----------------#
    # Average agent similarity matrices across time
    entropy_diff_matrix_across_time_heatmap = np.mean(
        entropy_diff_matrix_across_time, axis=0
    )
    js_distance_matrix_across_time_heatmap = np.mean(
        js_distance_matrix_across_time, axis=0
    )
    similarity_matrix_across_time_heatmap = np.mean(
        similarity_matrix_across_time, axis=0
    )

    # Create a mask for redundant information in symmetric matrices
    mask = np.tril(
        np.ones_like(entropy_diff_matrix_across_time_heatmap, dtype=bool), -1
    )

    # Plot agent similarity heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Entropy Difference Heatmap
    im1 = axes[0].imshow(
        np.ma.array(entropy_diff_matrix_across_time_heatmap, mask=mask),
        cmap="coolwarm",
        aspect="auto",
        vmin=0,
        vmax=0.5,
    )
    axes[0].set_title("Entropy Differences")
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_xticks(np.arange(num_agents))
    axes[0].set_yticks(np.arange(num_agents))
    axes[0].set_xticklabels(np.arange(1, num_agents + 1))
    axes[0].set_yticklabels(np.arange(1, num_agents + 1))
    axes[0].set_xlabel("Agent")
    axes[0].set_ylabel("Agent")

    # Annotate heatmap cells with values
    for i in range(num_agents):
        for j in range(i, num_agents):
            axes[0].text(
                j,
                i,
                f"{entropy_diff_matrix_across_time_heatmap[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    # Jensen-Shannon Distance Heatmap
    im2 = axes[1].imshow(
        np.ma.array(js_distance_matrix_across_time_heatmap, mask=mask),
        cmap="coolwarm",
        aspect="auto",
        vmin=0,
        vmax=0.5,
    )
    axes[1].set_title("JS Distances")
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_xticks(np.arange(num_agents))
    axes[1].set_yticks(np.arange(num_agents))
    axes[1].set_xticklabels(np.arange(1, num_agents + 1))
    axes[1].set_yticklabels(np.arange(1, num_agents + 1))
    axes[1].set_xlabel("Agent")
    axes[1].set_ylabel("Agent")

    # Annotate heatmap cells with values
    for i in range(num_agents):
        for j in range(i, num_agents):
            axes[1].text(
                j,
                i,
                f"{js_distance_matrix_across_time_heatmap[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    # Different Actions Percentage Heatmap
    im3 = axes[2].imshow(
        np.ma.array(similarity_matrix_across_time_heatmap, mask=mask),
        cmap="coolwarm",
        aspect="auto",
        vmin=0,
        vmax=60,
    )
    axes[2].set_title("Average Percentage of Diff Actions")
    fig.colorbar(im3, ax=axes[2])
    axes[2].set_xticks(np.arange(num_agents))
    axes[2].set_yticks(np.arange(num_agents))
    axes[2].set_xticklabels(np.arange(1, num_agents + 1))
    axes[2].set_yticklabels(np.arange(1, num_agents + 1))
    axes[2].set_xlabel("Agent")
    axes[2].set_ylabel("Agent")

    # Annotate heatmap cells with values
    for i in range(num_agents):
        for j in range(i, num_agents):
            axes[2].text(
                j,
                i,
                f"{similarity_matrix_across_time_heatmap[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()

    # Save the figure
    figure_path = "agent_similarity_heatmaps.png"
    plt.savefig(figure_path)
    wandb.log({"similarity_heatmaps": wandb.Image(figure_path)})
    plt.close()

    # Log the heatmap values as tables to wandb
    def matrix_to_wandb_table(matrix, title):
        """Convert a similarity matrix to a wandb Table format."""
        columns = ["Agent {}".format(i + 1) for i in range(num_agents)]
        data = [[matrix[i, j] for j in range(num_agents)] for i in range(num_agents)]
        return wandb.Table(
            columns=[""] + columns,
            data=[[f"Agent {i+1}"] + row for i, row in enumerate(data)],
        )

    # Log the matrices as tables for detailed inspection
    entropy_diff_table = matrix_to_wandb_table(
        entropy_diff_matrix_across_time_heatmap, "Entropy Differences"
    )
    js_distance_table = matrix_to_wandb_table(
        js_distance_matrix_across_time_heatmap, "JS Distances"
    )
    similarity_table = matrix_to_wandb_table(
        similarity_matrix_across_time_heatmap, "Average Percentage of Diff Actions"
    )

    wandb.log(
        {
            "Entropy Differences Table": entropy_diff_table,
            "JS Distances Table": js_distance_table,
            "Similarity Table": similarity_table,
        }
    )

    # ----------------- Per-Agent Metrics -----------------#
    # Calculate average metrics for each agent
    average_entropy_diff_per_agent = np.mean(
        entropy_diff_matrix_across_time_heatmap, axis=1
    )
    average_js_distance_per_agent = np.mean(
        js_distance_matrix_across_time_heatmap, axis=1
    )
    average_diff_actions_per_agent = np.mean(
        similarity_matrix_across_time_heatmap, axis=1
    )

    # Log per-agent metrics
    for agent_idx in range(num_agents):
        wandb.log(
            {
                f"agent_{agent_idx + 1}_average_entropy_diff": average_entropy_diff_per_agent[
                    agent_idx
                ],
                f"agent_{agent_idx + 1}_average_js_distance": average_js_distance_per_agent[
                    agent_idx
                ],
                f"agent_{agent_idx + 1}_average_diff_actions": average_diff_actions_per_agent[
                    agent_idx
                ],
            }
        )

    # ----------------- Action Probability Visualizations -----------------#
    # Create visualization of action probabilities over time
    data = []
    for agent_idx, agent_probs in enumerate(agent_probabilities_over_time):
        for step_idx in range(0, len(agent_probs), 50):  # Skip every 50 steps for clarity
            probs = agent_probs[step_idx]
            for action_idx, prob in enumerate(probs):
                data.append([agent_idx, step_idx, action_idx, prob])

    df = pd.DataFrame(data, columns=["Agent", "Step", "Action", "Probability"])
    df["Probability"] = df["Probability"].astype(float)

    # Set visualization style
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10", len(agent_probabilities_over_time))

    # Create faceted line plot of action probabilities over time
    g = sns.FacetGrid(df, col="Action", col_wrap=3, aspect=1.5, height=4)
    g.map(sns.lineplot, "Step", "Probability", "Agent", alpha=0.7, palette=palette)
    g.add_legend(title="Agent")
    g.set_axis_labels("Step", "Probability")
    g.set_titles(col_template="Action {col_name}")
    g.despine(left=True)
    plt.gcf().set_size_inches(12, 10)
    plt.savefig("agent_probabilities_over_time_plot.png")
    wandb.log(
        {
            "agent_probabilities_over_time_plot": wandb.Image(
                "agent_probabilities_over_time_plot.png"
            )
        }
    )
    plt.close()

    # Create visualization of action probabilities per step (averaged across episodes)
    data_per_step = []
    for agent_idx, agent_probs in enumerate(agent_probabilities_over_time_per_step):
        for step_idx, step_probs in enumerate(agent_probs):
            # Average across episodes for each step
            avg_probs_across_episodes_in_this_step = np.mean(step_probs, axis=0)
            for action_idx, prob in enumerate(avg_probs_across_episodes_in_this_step):
                data_per_step.append([agent_idx, step_idx, action_idx, prob])

    df_per_step = pd.DataFrame(
        data_per_step, columns=["Agent", "Step", "Action", "Probability"]
    )
    df_per_step["Probability"] = df_per_step["Probability"].astype(float)

    # Create faceted line plot of action probabilities by step
    g = sns.FacetGrid(df_per_step, col="Action", col_wrap=3, aspect=1.5, height=4)
    g.map(sns.lineplot, "Step", "Probability", "Agent", alpha=0.7, palette=palette)
    g.add_legend(title="Agent")
    g.set_axis_labels("Step", "Probability")
    g.set_titles(col_template="Action {col_name}")
    g.despine(left=True)
    plt.savefig("agent_probabilities_per_step_plot.png")
    wandb.log(
        {
            "agent_probabilities_per_step_plot": wandb.Image(
                "agent_probabilities_per_step_plot.png"
            )
        }
    )
    plt.close()

    # ----------------- Action Probability Histograms -----------------#
    # Create histograms of action probabilities over time
    g = sns.FacetGrid(df, row="Step", hue="Agent", aspect=3, height=2.5)
    g.map(sns.barplot, "Action", "Probability", alpha=0.6)
    g.add_legend(title="Agent")
    g.set_axis_labels("Action", "Probability")
    g.set_titles(row_template="Step {row_name}")
    g.despine(left=True)
    plt.savefig("agent_probabilities_hist_over_time_plot.png")
    wandb.log(
        {
            "agent_probabilities_hist_over_time_plot": wandb.Image(
                "agent_probabilities_hist_over_time_plot.png"
            )
        }
    )
    plt.close()

    # Create histograms of action probabilities per step
    data_per_step = []
    # Sample a subset of steps for clarity
    plot_every_x_steps = max_steps // 5
    for agent_idx, agent_probs in enumerate(agent_probabilities_over_time_per_step):
        for step_idx in range(
            0, len(agent_probs), plot_every_x_steps
        ):  # Sample every plot_every_x_steps steps
            step_probs = agent_probs[step_idx]
            avg_probs_across_episodes_in_this_step = np.mean(step_probs, axis=0)
            for action_idx, prob in enumerate(avg_probs_across_episodes_in_this_step):
                data_per_step.append([agent_idx, step_idx, action_idx, prob])

    df_per_step = pd.DataFrame(
        data_per_step, columns=["Agent", "Step", "Action", "Probability"]
    )
    df_per_step["Probability"] = df_per_step["Probability"].astype(float)

    g = sns.FacetGrid(df_per_step, row="Step", hue="Agent", aspect=3, height=2.5)
    g.map(sns.barplot, "Action", "Probability", alpha=0.6)
    g.add_legend(title="Agent")
    g.set_axis_labels("Action", "Probability")
    g.set_titles(row_template="Step {row_name}")
    g.despine(left=True)
    plt.savefig("agent_probabilities_hist_per_step_plot.png")
    wandb.log(
        {
            "agent_probabilities_hist_per_step_plot": wandb.Image(
                "agent_probabilities_hist_per_step_plot.png"
            )
        }
    )
    plt.close()


def get_policies(param_sharing, centralised_critic, networks, agent_state, observations, num_agents):
    """
    Extract policy distributions from trained agents.
    
    Handles different agent architectures (parameter sharing and centralized critic variants)
    to extract action probability distributions.
    
    Args:
        param_sharing: Whether parameters are shared across agents
        centralised_critic: Whether a centralized critic is used
        networks: Network architecture(s)
        agent_state: Trained agent parameters
        observations: Input observations
        num_agents: Number of agents
        
    Returns:
        policies: Action probability distributions for each agent
    """
    if not param_sharing:
        if centralised_critic:
            # Create dummy observations for critic (not used for action selection)
            dummy_obs = jnp.zeros((observations.shape[0], observations.shape[1] * num_agents))
            policies = jnp.stack([
                distrax.Categorical(logits=networks[i].apply(agent_state[i].params, observations, dummy_obs)[0]).probs
                for i in range(num_agents)
            ])
        else:
            policies = jnp.stack([
                distrax.Categorical(logits=networks[i].apply(agent_state[i].params, observations)[0]).probs
                for i in range(num_agents)
            ])
    else:
        # Generate one-hot encoded agent IDs for parameter sharing
        agent_ids = jnp.eye(num_agents)
        if centralised_critic:
            # Create dummy observations for critic (not used for action selection)
            dummy_obs = jnp.zeros((observations.shape[0], observations.shape[1] * num_agents))
            policies = jnp.stack([
                distrax.Categorical(logits=networks.apply(
                    agent_state.params, 
                    jnp.concatenate([observations, jnp.tile(agent_ids[i], (observations.shape[0], 1))], axis=1), 
                    dummy_obs
                )[0]).probs
                for i in range(num_agents)
            ])
        else:
            policies = jnp.stack([
                distrax.Categorical(logits=networks.apply(
                    agent_state.params, 
                    jnp.concatenate([observations, jnp.tile(agent_ids[i], (observations.shape[0], 1))], axis=1)
                )[0]).probs
                for i in range(num_agents)
            ])
    
    return policies


def calculate_snd(observations, networks, agent_state, _keys, num_agents, centralised_critic, param_sharing, distance_fn=jensenshannon):
    """
    Calculate the System-level Normalized Diversity (SND) measure.
    
    SND quantifies the diversity of policies across agents, normalized by the number of
    agents and observations, providing a single metric for team diversity.
    
    Args:
        observations: Set of observations to evaluate against
        networks: Network architecture(s)
        agent_state: Trained agent parameters
        _keys: JAX random keys
        num_agents: Number of agents
        centralised_critic: Whether a centralized critic is used
        param_sharing: Whether parameters are shared across agents
        distance_fn: Distance function to use (default: Jensen-Shannon distance)
        
    Returns:
        snd: System-level Normalized Diversity value
    """
    # Get policy distributions for all agents
    policies = get_policies(param_sharing, centralised_critic, networks, agent_state, observations, num_agents)
    
    # Calculate pairwise distances between all agent policies
    def pairwise_distance(pol_i, pol_j):
        return jnp.sum(distance_fn(pol_i, pol_j))
    
    # Sum distances between all pairs of agents
    total_distance = sum(pairwise_distance(policies[i], policies[j])
                         for i in range(num_agents)
                         for j in range(i+1, num_agents))
    
    # Normalize by the number of pairs and observations
    num_observations = observations.shape[0]
    snd = (2 / (num_agents * (num_agents - 1) * num_observations)) * total_distance
    
    return snd