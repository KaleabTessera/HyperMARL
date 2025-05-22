import jax
import numpy as np
import matplotlib.pyplot as plt
import wandb

def process_metric(metric_data, episode_mask, axis):
    masked_metric = np.where(episode_mask, metric_data, np.nan)
    mean_metric = np.nanmean(masked_metric, axis=axis).squeeze()
    std_metric = np.nanstd(masked_metric, axis=axis).squeeze()
    return mean_metric, std_metric

def log_metric_to_wandb(updates, mean_metric, std_metric, metric_name):
    for update, mean_value, std_value in zip(updates, mean_metric, std_metric):
        wandb.log({
            "update": update,
            f"mean_{metric_name}": mean_value,
            f"std_{metric_name}": std_value
        })

def create_and_log_plot(updates, mean_metric, std_metric, metric_name, config):
    plt.figure(figsize=(10, 6))
    plt.plot(updates, mean_metric, label=f'Average {metric_name.capitalize()}')
    plt.fill_between(updates, mean_metric - std_metric, mean_metric + std_metric, alpha=0.3)
    plt.xlabel('Updates')
    plt.ylabel(metric_name.capitalize())
    plt.title(f"IPPO-RNN={config['ENV_NAME']} - {metric_name.capitalize()}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_filename = f"ippo_RNN_{config['ENV_NAME']}_{metric_name}.png"
    plt.savefig(plot_filename)
    wandb.log({f"{metric_name}_plot": wandb.Image(plot_filename)})
    plt.close()

def log_experiment_results(config, out, axis):
    episode_mask = out["metrics"]["returned_episode"]

    # Process and log returns
    episode_returns = out["metrics"]["returned_episode_returns"]
    mean_returns, std_returns = process_metric(episode_returns, episode_mask, axis=axis)
    
    # Create x-axis values 
    updates = np.arange(len(mean_returns))
    log_metric_to_wandb(updates, mean_returns, std_returns, "return")
    create_and_log_plot(updates, mean_returns, std_returns, "return", config)

    # Process and log win rates
    episode_win_rates = out["metrics"]["returned_won_episode"]
    mean_win_rates, std_win_rates = process_metric(episode_win_rates, episode_mask, axis=axis)
    log_metric_to_wandb(updates, mean_win_rates, std_win_rates, "win_rate")
    create_and_log_plot(updates, mean_win_rates, std_win_rates, "win_rate", config)

    # log eval metrics
    # shape [num_seeds, num_updates]
    test_episode_returns = out["metrics"]["test_returned_episode_returns"]
    mean_test_returns, std_test_returns = np.mean(test_episode_returns, axis=0), np.std(test_episode_returns, axis=0)
    log_metric_to_wandb(updates, mean_test_returns, std_test_returns, "test_return")
    create_and_log_plot(updates, mean_test_returns, std_test_returns, "test_return", config)
    
    test_episode_win_rates = out["metrics"]["test_returned_won_episode"]
    mean_test_win_rates, std_test_win_rates = np.mean(test_episode_win_rates, axis=0), np.std(test_episode_win_rates, axis=0)
    log_metric_to_wandb(updates, mean_test_win_rates, std_test_win_rates, "test_win_rate")
    create_and_log_plot(updates, mean_test_win_rates, std_test_win_rates, "test_win_rate", config)