import numpy as np
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Keep the same helper functions
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def moving_average(data, window_size):
    if len(data) < window_size:
        # If data is smaller than window, use a smaller window
        window_size = max(2, len(data) // 2)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def set_chart_style():
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1,
        'lines.markersize': 3,
        'figure.figsize': (3.5, 2.5),
        'figure.dpi': 300,
    })

def plot_returns_stateful(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, num_agents, num_trials, batch_size, name):
    policy_types = ['PG-NoPS', 'PG-FuPS', 'PG-FuPS+ID (One-Hot)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    window_size = 100  
    num_updates = num_trials // batch_size
    set_chart_style()
    fig, ax = plt.subplots()
    for i, policy_type in enumerate(policy_types):
        all_returns = []
        for seed in range(len(all_results)):
            if policy_type == 'PG-NoPS':
                all_rewards = np.mean([np.array(policy.all_rewards) for policy in all_pg_no_ps[seed]], axis=0)
            elif policy_type == 'PG-FuPS':
                all_rewards = np.array(all_pg_fu_ps[seed].all_rewards)
            else:
                all_rewards = np.array(all_pg_fu_ps_plus_id_one_hot[seed].all_rewards)
            returns = all_rewards.reshape(-1, batch_size).mean(axis=1)
            all_returns.append(returns)
        mean_returns = np.mean(all_returns, axis=0)
        std_returns  = np.std( all_returns, axis=0)
        sm_m = moving_average(mean_returns, window_size)
        sm_s = moving_average(std_returns,  window_size)
        x   = np.linspace(0, num_updates, len(sm_m))
        ax.plot(x, sm_m, color=colors[i], label=policy_type)
        ax.fill_between(x,
                        np.maximum(0, sm_m - sm_s),
                        np.minimum(1, sm_m + sm_s),
                        color=colors[i], alpha=0.2)
    ax.set_ylabel('Average Reward')
    ax.set_ylim(0,1)
    ax.set_xlabel('Update Step')
    ax.set_xlim(0,num_updates)
    ax.grid(True, linestyle='--', alpha=0.7)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fn = f'{name}_returns.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"returns_plot": wandb.Image(fig)})
    plt.close()
    

def plot_returns(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, num_agents, num_trials, batch_size, name, use_returns=False, all_pg_fu_ps_plus_id_no_state=None,all_pg_hypernet=None):
    policy_types = ['PG-NoPS', 'PG-FuPS', 'PG-FuPS+ID']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Added fourth color
    
    if all_pg_fu_ps_plus_id_no_state is not None:
        policy_types.append('PG-FuPS+ID (No State)')
    
    if all_pg_hypernet is not None:
        policy_types.append('PG-HyperMARL')
        
    window_size = 100  
    set_chart_style()
    fig, ax = plt.subplots()
    
    # Track the maximum number of updates across policies
    max_updates = 0

    num_updates = num_trials // batch_size
    print(f"Number of updates: {num_updates}")
    
    for i, policy_type in enumerate(policy_types):
        all_returns = []
        
        # Process each policy type separately
        if policy_type == 'PG-NoPS':
            for seed in range(len(all_results)):
                # For NoPS processing (unchanged)
                if use_returns:
                    agent_rewards = [np.array(policy.all_returns) for policy in all_pg_no_ps[seed]]
                else:
                    agent_rewards = [np.array(policy.all_rewards) for policy in all_pg_no_ps[seed]]
                real_batch_size = batch_size // num_agents
                
                try:
                    seed_rewards = np.mean(agent_rewards, axis=0)
                    returns = seed_rewards.reshape(-1, real_batch_size).mean(axis=1)
                except ValueError:
                    seed_rewards = np.mean(agent_rewards, axis=0)
                    returns = seed_rewards
                    
                all_returns.append(returns)
        elif policy_type == 'PG-FuPS':
            for seed in range(len(all_results)):
                # For FuPS processing (unchanged)
                if use_returns:
                    all_rewards = np.array(all_pg_fu_ps[seed].all_returns)
                else:
                    all_rewards = np.array(all_pg_fu_ps[seed].all_rewards)
                    
                try:
                    returns = all_rewards.reshape(-1, batch_size).mean(axis=1)
                except ValueError:
                    returns = all_rewards
                    
                all_returns.append(returns)
        elif policy_type == 'PG-FuPS+ID':
            for seed in range(len(all_results)):
                # For FuPS+ID processing (unchanged)
                if use_returns:
                    all_rewards = np.array(all_pg_fu_ps_plus_id_one_hot[seed].all_returns)
                else:
                    all_rewards = np.array(all_pg_fu_ps_plus_id_one_hot[seed].all_rewards)
                    
                try:
                    returns = all_rewards.reshape(-1, batch_size).mean(axis=1)
                except ValueError:
                    returns = all_rewards
                    
                all_returns.append(returns)
        elif policy_type == 'PG-FuPS+ID (No State)':
            for seed in range(len(all_results)):
                # For FuPS+ID (No State) processing
                if use_returns:
                    all_rewards = np.array(all_pg_fu_ps_plus_id_no_state[seed].all_returns)
                else:
                    all_rewards = np.array(all_pg_fu_ps_plus_id_no_state[seed].all_rewards)
                    
                try:
                    returns = all_rewards.reshape(-1, batch_size).mean(axis=1)
                except ValueError:
                    returns = all_rewards
                    
                all_returns.append(returns)
        else:  # PG-HyperMARL
            for seed in range(len(all_results)):
                if use_returns:
                    all_rewards = np.array(all_pg_hypernet[seed].all_returns)
                else:
                    all_rewards = np.array(all_pg_hypernet[seed].all_rewards)
                    
                try:
                    returns = all_rewards.reshape(-1, batch_size).mean(axis=1)
                except ValueError:
                    returns = all_rewards
                    
                all_returns.append(returns)
        
        # Find minimum length within this policy type's returns
        policy_min_length = min(len(ret) for ret in all_returns)
        
        # Update max_updates if this policy has more data
        max_updates = max(max_updates, policy_min_length)
        
        # Align data within this policy type
        all_returns = [returns[:policy_min_length] for returns in all_returns]
        
        # Calculate mean and std
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        
        # Apply smoothing
        window = min(window_size, len(mean_returns) // 2)
        window = max(2, window)  # Ensure window is at least 2
        sm_m = moving_average(mean_returns, window)
        sm_s = moving_average(std_returns, window)
        
        # Plot with proper x-axis
        x = np.linspace(0, policy_min_length, len(sm_m))
        
        ax.plot(x, sm_m, color=colors[i], label=policy_type)
        ax.fill_between(x,
                        np.maximum(0, sm_m - sm_s),
                        np.minimum(1, sm_m + sm_s),
                        color=colors[i], alpha=0.2)
    
    # Set x-axis limit to show the full range of data
    ax.set_xlim(0, max_updates)
    
    ax.set_ylabel('Average Reward')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Update Step')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fn = f'{name}_returns.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"returns_plot": wandb.Image(fig)})
    plt.close()
    
def plot_gradient_variance(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, num_agents, num_trials, batch_size, name, all_pg_fu_ps_plus_id_no_state=None,all_pg_hypernet=None):
    policy_types = ['PG-NoPS', 'PG-FuPS', 'PG-FuPS+ID']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    if all_pg_fu_ps_plus_id_no_state is not None:
        policy_types.append('PG-FuPS+ID (No State)')
    
    if all_pg_hypernet is not None:
        policy_types.append('PG-HyperMARL')
        
    window_size = 100
    set_chart_style()
    fig, ax = plt.subplots()
    max_v = 0
    
    for i, policy_type in enumerate(policy_types):
        all_gv = []
        for seed in range(len(all_results)):
            if policy_type == 'PG-NoPS':
                gv = np.mean([np.array(policy.gradient_variances) for policy in all_pg_no_ps[seed]], axis=0)
            elif policy_type == 'PG-FuPS':
                gv = np.array(all_pg_fu_ps[seed].gradient_variances)
            elif policy_type == 'PG-FuPS+ID':
                gv = np.array(all_pg_fu_ps_plus_id_one_hot[seed].gradient_variances)
            elif policy_type == 'PG-FuPS+ID (No State)':
                gv = np.array(all_pg_fu_ps_plus_id_no_state[seed].gradient_variances)
            else:  # PG-HyperMARL
                gv = np.array(all_pg_hypernet[seed].gradient_variances)
            all_gv.append(gv)
        
        # Find minimum length across seeds to align data
        min_length = min(len(gv) for gv in all_gv)
        all_gv = [gv[:min_length] for gv in all_gv]
            
        m = np.mean(all_gv, axis=0)
        s = np.std(all_gv, axis=0)
        
        max_v = max(max_v, np.max(m+s))
        
        # Apply smoothing with adaptive window size
        sm_m = moving_average(m, min(window_size, len(m) // 2))
        sm_s = moving_average(s, min(window_size, len(s) // 2))
        
        # Create x-axis based on number of points
        x = np.linspace(0, 1, len(sm_m))
        
        ax.plot(x, sm_m, color=colors[i], label=policy_type)
        ax.fill_between(x,
                        np.maximum(0, sm_m - sm_s),
                        sm_m + sm_s,
                        color=colors[i], alpha=0.2)
    
    ax.set_ylabel('Gradient Variance')
    ax.set_xlabel('Relative Training Progress')
    ax.set_ylim(0, max_v*1.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fn = f'{name}_grad_var.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"grad_var": wandb.Image(fig)})
    plt.close()


def plot_gradient_norm_std(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, num_agents, num_trials, batch_size, name, all_pg_fu_ps_plus_id_no_state=None,all_pg_hypernet=None):
    policy_types = ['PG-NoPS', 'PG-FuPS', 'PG-FuPS+ID']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    if all_pg_fu_ps_plus_id_no_state is not None:
        policy_types.append('PG-FuPS+ID (No State)')
        
    if all_pg_hypernet is not None:
        policy_types.append('PG-HyperMARL')
        
        
    window_size = 100
    set_chart_style()
    fig, ax = plt.subplots()
    max_n = 0
    
    for i, policy_type in enumerate(policy_types):
        all_n = []
        for seed in range(len(all_results)):
            if policy_type == 'PG-NoPS':
                n = np.mean([np.array(policy.gradient_norms) for policy in all_pg_no_ps[seed]], axis=0)
            elif policy_type == 'PG-FuPS':
                n = np.array(all_pg_fu_ps[seed].gradient_norms)
            elif policy_type == 'PG-FuPS+ID':
                n = np.array(all_pg_fu_ps_plus_id_one_hot[seed].gradient_norms)
            elif policy_type == 'PG-FuPS+ID (No State)':
                n = np.array(all_pg_fu_ps_plus_id_no_state[seed].gradient_norms)
            else:  # PG-HyperMARL
                n = np.array(all_pg_hypernet[seed].gradient_norms)
            all_n.append(n)
        
        # Find minimum length across seeds to align data
        min_length = min(len(n) for n in all_n)
        all_n = [n[:min_length] for n in all_n]
            
        m = np.mean(all_n, axis=0)
        s = np.std(all_n, axis=0)
        
        max_n = max(max_n, np.max(m+s))
        
        # Apply smoothing with adaptive window size
        sm_m = moving_average(m, min(window_size, len(m) // 2))
        sm_s = moving_average(s, min(window_size, len(s) // 2))
        
        # Create x-axis based on number of points
        x = np.linspace(0, 1, len(sm_m))
        
        ax.plot(x, sm_m, color=colors[i], label=policy_type)
        ax.fill_between(x,
                        np.maximum(0, sm_m - sm_s),
                        sm_m + sm_s,
                        color=colors[i], alpha=0.2)
    
    ax.set_ylabel('Gradient Norm')
    ax.set_xlabel('Relative Training Progress')
    ax.set_ylim(0, max_n*1.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fn = f'{name}_grad_norm_std.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"grad_norm_std": wandb.Image(fig)})
    plt.close()

def plot_gradient_conflicts(all_pg_fu_ps_plus_id_one_hot, num_trials, batch_size, name, all_pg_fu_ps_plus_id_no_state=None,all_pg_hypernet=None):
    window_size = 100
    set_chart_style()
    fig, ax = plt.subplots()
    
    # Process first policy (PG-FuPS+ID)
    if all_pg_fu_ps_plus_id_one_hot is not None:
        conflicts1 = [np.array(p.gradient_conflicts) for p in all_pg_fu_ps_plus_id_one_hot]
        min_length1 = min(len(c) for c in conflicts1)
        conflicts1 = [c[:min_length1] for c in conflicts1]
        m1 = np.mean(conflicts1, axis=0)
        s1 = np.std(conflicts1, axis=0)
        sm_m1 = moving_average(m1, min(window_size, len(m1) // 2))
        sm_s1 = moving_average(s1, min(window_size, len(s1) // 2))
        x1 = np.linspace(0, 1, len(sm_m1))
    
        ax.plot(x1, sm_m1, color='#2ca02c', label="PG-FuPS+ID Cosine Similarity")
        ax.fill_between(x1, sm_m1 - sm_s1, sm_m1 + sm_s1, color='#2ca02c', alpha=0.2)
    
    # Process second policy (PG-FuPS+ID No State) if provided
    if all_pg_fu_ps_plus_id_no_state is not None:
        conflicts2 = [np.array(p.gradient_conflicts) for p in all_pg_fu_ps_plus_id_no_state]
        min_length2 = min(len(c) for c in conflicts2)
        conflicts2 = [c[:min_length2] for c in conflicts2]
        m2 = np.mean(conflicts2, axis=0)
        s2 = np.std(conflicts2, axis=0)
        sm_m2 = moving_average(m2, min(window_size, len(m2) // 2))
        sm_s2 = moving_average(s2, min(window_size, len(s2) // 2))
        x2 = np.linspace(0, 1, len(sm_m2))
        
        ax.plot(x2, sm_m2, color='#d62728', label="PG-FuPS+ID (No State) Cosine Similarity")
        ax.fill_between(x2, sm_m2 - sm_s2, sm_m2 + sm_s2, color='#d62728', alpha=0.2)
    
     # Process hypernetwork policy if provided
    if all_pg_hypernet is not None:
        conflicts3 = [np.array(p.gradient_conflicts) for p in all_pg_hypernet]
        if conflicts3 and any(len(c) > 0 for c in conflicts3):  # Check if data exists
            min_length3 = min(len(c) for c in conflicts3 if len(c) > 0)
            conflicts3 = [c[:min_length3] for c in conflicts3 if len(c) > 0]
            if conflicts3:  # Make sure we still have data after filtering
                m3 = np.mean(conflicts3, axis=0)
                s3 = np.std(conflicts3, axis=0)
                sm_m3 = moving_average(m3, min(window_size, len(m3) // 2))
                sm_s3 = moving_average(s3, min(window_size, len(s3) // 2))
                x3 = np.linspace(0, 1, len(sm_m3))
                
                ax.plot(x3, sm_m3, color='#9467bd', label="PG-HyperMARL Cosine Similarity")
                ax.fill_between(x3, sm_m3 - sm_s3, sm_m3 + sm_s3, color='#9467bd', alpha=0.2)
    
    
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_xlabel('Relative Training Progress')
    ax.set_title('Gradient Conflict (Cosine Similarity)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fn = f'{name}_grad_conflict.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"grad_conflict": wandb.Image(fig)})
    plt.close()


def plot_legend(name, policy_types):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(5, 0.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, pt in enumerate(policy_types):
        ax.plot([], [], color=colors[i], label=pt)
    ax.legend(loc='center', ncol=4, frameon=False)
    ax.axis('off')
    plt.tight_layout()
    fn = f'{name}_legend.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"legend": wandb.Image(fig)})
    plt.close()


def plot_combined_results(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, num_agents, num_trials, batch_size, name, use_returns=False, all_pg_fu_ps_plus_id_no_state=None, all_pg_hypernet=None):
    plot_returns(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, 
                num_agents, num_trials, batch_size, name, use_returns=use_returns, 
                all_pg_fu_ps_plus_id_no_state=all_pg_fu_ps_plus_id_no_state,
                all_pg_hypernet=all_pg_hypernet)
    
    plot_gradient_variance(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, 
                          num_agents, num_trials, batch_size, name,
                          all_pg_fu_ps_plus_id_no_state=all_pg_fu_ps_plus_id_no_state,
                          all_pg_hypernet=all_pg_hypernet)
    
    plot_gradient_norm_std(all_results, all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, 
                          num_agents, num_trials, batch_size, name,
                          all_pg_fu_ps_plus_id_no_state=all_pg_fu_ps_plus_id_no_state,
                          all_pg_hypernet=all_pg_hypernet)
    
    policy_types = ['PG-NoPS', 'PG-FuPS', 'PG-FuPS+ID']
    if all_pg_fu_ps_plus_id_no_state is not None:
        policy_types.append('PG-FuPS+ID (No State)')
    if all_pg_hypernet is not None:
        policy_types.append('PG-HyperMARL')
    
    plot_legend(name, policy_types)
    
    plot_gradient_conflicts(all_pg_fu_ps_plus_id_one_hot, num_trials, batch_size, name, 
                           all_pg_fu_ps_plus_id_no_state=all_pg_fu_ps_plus_id_no_state,
                           all_pg_hypernet=all_pg_hypernet)

# Keep the rest of the utility functions unchanged
def compute_stability_proxy(all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_plus_id_one_hot, all_pg_fu_ps_plus_id_no_state=None, all_pg_hypernet=None):
    proxies = {}
    # PG-NoPS
    stds=[]
    if all_pg_no_ps is not None and len(all_pg_no_ps) > 0:
        for seed_policies in all_pg_no_ps:
            if seed_policies is not None: 
                norms = np.stack([policy.gradient_norms for policy in seed_policies], axis=0)
                mn = norms.mean(axis=0)
                stds.append(np.std(mn))
                
    if len(stds) > 0:
        proxies['PG-NoPS'] = (np.mean(stds), np.std(stds))
    
    # PG-FuPS
    if all_pg_fu_ps is not None:
        stds = [np.std(policy.gradient_norms) for policy in all_pg_fu_ps]
        proxies['PG-FuPS'] = (np.mean(stds), np.std(stds))
    
    # PG-FuPS+ID
    if all_pg_fu_ps_plus_id_one_hot is not None:
        stds = [np.std(policy.gradient_norms) for policy in all_pg_fu_ps_plus_id_one_hot]
        proxies['PG-FuPS+ID'] = (np.mean(stds), np.std(stds))
    
    # PG-FuPS+ID (No State)
    if all_pg_fu_ps_plus_id_no_state is not None:
        stds = [np.std(policy.gradient_norms) for policy in all_pg_fu_ps_plus_id_no_state]
        proxies['PG-FuPS+ID (No State)'] = (np.mean(stds), np.std(stds))
        
    # PG-HyperMARL
    if all_pg_hypernet is not None:
        stds = [np.std(policy.gradient_norms) for policy in all_pg_hypernet]
        proxies['PG-HyperMARL'] = (np.mean(stds), np.std(stds))
        
        
    return proxies


def plot_stability_proxy(proxies, name):
    set_chart_style()
    labels = list(proxies.keys())
    means  = [proxies[l][0] for l in labels]
    errs   = [proxies[l][1] for l in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, means, yerr=errs, capsize=3)
    ax.set_ylabel('Gradient‑norm STD')
    ax.set_title('Stability Proxy')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fn = f'{name}_stability.pdf'
    plt.savefig(fn, bbox_inches='tight')
    wandb.log({"stability": wandb.Image(fig)})
    plt.close()
    
    wandb.log({
        "stability_proxy": wandb.Table(data=[[l, m, e] for l, (m, e) in proxies.items()],
                                        columns=["Policy Type", "Mean", "STD"])
    })
    

def log_all_grad_metrics(all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_id,all_pg_fu_ps_id_no_state=None,all_pg_hypernet=None):
    """
    all_pg_no_ps:  list of length num_seeds, each entry is a list of PGNoPS agents
    all_pg_fu_ps:  list of length num_seeds, each entry is a single PGFuPS object
    all_pg_fu_ps_plus_id_one_hot: same for PGFuPSPlusIDOneHot
    """
    # PG‑NoPS: per‐agent, per‐seed
    if all_pg_no_ps is not None and len(all_pg_no_ps) > 0:
        for seed_idx, seed_policies in enumerate(all_pg_no_ps):
            if seed_policies is not None:
                for agent_idx, p in enumerate(seed_policies):
                    # gradient norms
                    for upd, norm in enumerate(p.gradient_norms):
                        wandb.log(
                            {f"grad_norm/PG-NoPS/{seed_idx}/agent_{agent_idx}": norm, "global_step": upd}
                        )
                    # gradient variances
                    for upd, var in enumerate(p.gradient_variances):
                        wandb.log(
                            {f"grad_var/PG-NoPS/{seed_idx}/agent_{agent_idx}": var, "global_step": upd},
                        )

    # PG‑FuPS: per‐seed
    if all_pg_fu_ps is not None:
        for seed_idx, p in enumerate(all_pg_fu_ps):
            if p is not None:
                for upd, norm in enumerate(p.gradient_norms):
                    wandb.log({f"grad_norm/PG-FuPS/{seed_idx}": norm, "global_step":upd})
                for upd, var in enumerate(p.gradient_variances):
                    wandb.log({f"grad_var/PG-FuPS/{seed_idx}": var , "global_step":upd})

    # PG‑FuPS+ID: per‐seed, also log conflicts
    if all_pg_fu_ps_id is not None:
        for seed_idx, p in enumerate(all_pg_fu_ps_id):
            for upd, norm in enumerate(p.gradient_norms):
                wandb.log({f"grad_norm/PG-FuPS+ID/{seed_idx}": norm , "global_step":upd})
            for upd, var in enumerate(p.gradient_variances):
                wandb.log({f"grad_var/PG-FuPS+ID/{seed_idx}": var , "global_step":upd})
            # gradient_conflicts only exists on the ID version
            for upd, conf in enumerate(p.gradient_conflicts):
                wandb.log({f"grad_conflict/PG-FuPS+ID/{seed_idx}": conf, "global_step":upd})
            
    # PG‑FuPS+ID (no state): per‐seed
    if all_pg_fu_ps_id_no_state is not None:
        for seed_idx, p in enumerate(all_pg_fu_ps_id_no_state):
            for upd, norm in enumerate(p.gradient_norms):
                wandb.log({f"grad_norm/PG-FuPS+ID-no-state/{seed_idx}": norm , "global_step":upd})
            for upd, var in enumerate(p.gradient_variances):
                wandb.log({f"grad_var/PG-FuPS+ID-no-state/{seed_idx}": var , "global_step":upd})
            for upd, conf in enumerate(p.gradient_conflicts):
                wandb.log({f"grad_conflict/PG-FuPS+ID/{seed_idx}": conf, "global_step":upd})

     # PG‑HyperMARL: per‐seed
    if all_pg_hypernet is not None:
        for seed_idx, p in enumerate(all_pg_hypernet):
            for upd, norm in enumerate(p.gradient_norms):
                wandb.log({f"grad_norm/PG-HyperMARL/{seed_idx}": norm, "global_step":upd})
            for upd, var in enumerate(p.gradient_variances):
                wandb.log({f"grad_var/PG-HyperMARL/{seed_idx}": var, "global_step":upd})
            if hasattr(p, 'gradient_conflicts'):
                for upd, conf in enumerate(p.gradient_conflicts):
                    wandb.log({f"grad_conflict/PG-HyperMARL/{seed_idx}": conf, "global_step":upd})

            
def log_all_stability_proxy(all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_id,all_pg_fu_ps_id_no_state=None,all_pg_hypernet=None):
    """
    For each seed & policy type, compute the std of its gradient_norms across updates,
    and log it once (step=0) under a per‑seed key.
    """
    if all_pg_no_ps is not None and len(all_pg_no_ps) > 0:
        # PG‑NoPS: each seed is a list of agents
        for seed_idx, seed_policies in enumerate(all_pg_no_ps):
            if seed_policies is not None:
                # stack agents' norms, average across agents to get one time‑series, then std
                norms = np.stack([policy.gradient_norms for policy in seed_policies], axis=0)
                mean_norms = norms.mean(axis=0)
                proxy = float(np.std(mean_norms))
                wandb.log({f"stability_proxy/PG-NoPS/{seed_idx}": proxy})

    
    # PG‑FuPS: one object per seed
    if all_pg_fu_ps is not None:
        for seed_idx, p in enumerate(all_pg_fu_ps):
            proxy = float(np.std(p.gradient_norms))
            wandb.log({f"stability_proxy/PG-FuPS/{seed_idx}": proxy})

    # PG‑FuPS+ID: one object per seed
    if all_pg_fu_ps_id is not None:
        for seed_idx, p in enumerate(all_pg_fu_ps_id):
            proxy = float(np.std(p.gradient_norms))
            wandb.log({f"stability_proxy/PG-FuPS+ID/{seed_idx}": proxy})
        
    if all_pg_fu_ps_id_no_state is not None:
        # PG‑FuPS+ID (no state): one object per seed
        for seed_idx, p in enumerate(all_pg_fu_ps_id_no_state):
            proxy = float(np.std(p.gradient_norms))
            wandb.log({f"stability_proxy/PG-FuPS+ID-no-state/{seed_idx}": proxy})

    if all_pg_hypernet is not None:
        # PG‑HyperMARL: one object per seed
        for seed_idx, p in enumerate(all_pg_hypernet):
            proxy = float(np.std(p.gradient_norms))
            wandb.log({f"stability_proxy/PG-HyperMARL/{seed_idx}": proxy})

def log_raw_gradient_norms(all_pg_no_ps, all_pg_fu_ps, all_pg_fu_ps_id,all_pg_fu_ps_id_no_state=None,all_pg_hypernet=None):
    """
    Log raw gradient norms as tables to wandb for later analysis.
    
    Args:
        all_pg_no_ps: List of lists of PGNoPS policies (one list per seed)
        all_pg_fu_ps: List of PGFuPS policies (one per seed)
        all_pg_fu_ps_plus_id_one_hot: List of PGFuPSPlusIDOneHot policies (one per seed)
    """
    # Create tables for each policy type
    nops_table_data = []
    fups_table_data = []
    fups_id_table_data = []
    fups_id_table_data_no_state = []
    hypernet_table_data = []
    
    # PG-NoPS: collect data for each seed and agent
    if all_pg_no_ps is not None:
        for seed_idx, seed_policies in enumerate(all_pg_no_ps):
            if seed_policies is not None:
                for agent_idx, policy in enumerate(seed_policies):
                    for step_idx, norm in enumerate(policy.gradient_norms):
                        nops_table_data.append([seed_idx, agent_idx, step_idx, float(norm)])
    
    # PG-FuPS: collect data for each seed
    if all_pg_fu_ps is not None:
        for seed_idx, policy in enumerate(all_pg_fu_ps):
            for step_idx, norm in enumerate(policy.gradient_norms):
                fups_table_data.append([seed_idx, step_idx, float(norm)])
    
    # PG-FuPS+ID: collect data for each seed
    if all_pg_fu_ps_id is not None:
        for seed_idx, policy in enumerate(all_pg_fu_ps_id):
            for step_idx, norm in enumerate(policy.gradient_norms):
                fups_id_table_data.append([seed_idx, step_idx, float(norm)])
            
    # PG-FuPS+ID (no state): collect data for each seed
    if all_pg_fu_ps_id_no_state is not None:
        for seed_idx, policy in enumerate(all_pg_fu_ps_id_no_state):
            for step_idx, norm in enumerate(policy.gradient_norms):
                fups_id_table_data_no_state.append([seed_idx, step_idx, float(norm)])
    
      # PG-HyperMARL: collect data for each seed
    if all_pg_hypernet is not None:
        for seed_idx, policy in enumerate(all_pg_hypernet):
            for step_idx, norm in enumerate(policy.gradient_norms):
                hypernet_table_data.append([seed_idx, step_idx, float(norm)])
    
    
    # Log tables to wandb
    table_dict = {
        "raw_grad_norms/PG-NoPS": wandb.Table(
            data=nops_table_data, 
            columns=["seed", "agent", "step", "gradient_norm"]
        ),
        "raw_grad_norms/PG-FuPS": wandb.Table(
            data=fups_table_data,
            columns=["seed", "step", "gradient_norm"]
        ),
        "raw_grad_norms/PG-FuPS+ID": wandb.Table(
            data=fups_id_table_data,
            columns=["seed", "step", "gradient_norm"]
        ),
    }
    
    if all_pg_fu_ps_id_no_state is not None:
        table_dict["raw_grad_norms/PG-FuPS+ID-no-state"] = wandb.Table(
            data=fups_id_table_data_no_state,
            columns=["seed", "step", "gradient_norm"]
        )
    
    if all_pg_hypernet is not None:
        table_dict["raw_grad_norms/PG-HyperMARL"] = wandb.Table(
            data=hypernet_table_data,
            columns=["seed", "step", "gradient_norm"]
        )

    # Log all tables to wandb
    wandb.log(table_dict)
    
def log_raw_gradient_conflicts(all_pg_fu_ps_id, all_pg_fu_ps_id_no_state=None, all_pg_hypernet=None):
    """
    Log raw gradient conflicts as tables to wandb for later analysis.
    
    Args:
        all_pg_fu_ps_id: List of PGFuPS+ID policies (one per seed)
        all_pg_fu_ps_id_no_state: Optional list of PGFuPS+ID (no state) policies
        all_pg_hypernet: Optional list of HyperMARL policies
    """
    # Create tables for each policy type
    fups_id_table_data = []
    fups_id_table_data_no_state = []
    hypernet_table_data = []
    table_dict = {}
    
    # PG-FuPS+ID: collect data for each seed
    if all_pg_fu_ps_id is not None:
        for seed_idx, policy in enumerate(all_pg_fu_ps_id):
            for step_idx, conflict in enumerate(policy.gradient_conflicts):
                fups_id_table_data.append([seed_idx, step_idx, float(conflict)])
        
        # Initialize the table dictionary
        table_dict = {
            "raw_grad_conflicts/PG-FuPS+ID": wandb.Table(
                data=fups_id_table_data,
                columns=["seed", "step", "gradient_conflict"]
            )
        }
    
    # PG-FuPS+ID (no state): collect data for each seed if available
    if all_pg_fu_ps_id_no_state is not None:
        for seed_idx, policy in enumerate(all_pg_fu_ps_id_no_state):
            for step_idx, conflict in enumerate(policy.gradient_conflicts):
                fups_id_table_data_no_state.append([seed_idx, step_idx, float(conflict)])
        
        table_dict["raw_grad_conflicts/PG-FuPS+ID-no-state"] = wandb.Table(
            data=fups_id_table_data_no_state,
            columns=["seed", "step", "gradient_conflict"]
        )
    
    # PG-HyperMARL: collect data for each seed if available
    if all_pg_hypernet is not None:
        for seed_idx, policy in enumerate(all_pg_hypernet):
            if hasattr(policy, 'gradient_conflicts') and policy.gradient_conflicts:
                for step_idx, conflict in enumerate(policy.gradient_conflicts):
                    hypernet_table_data.append([seed_idx, step_idx, float(conflict)])
        
        if hypernet_table_data:  # Only add if we have data
            table_dict["raw_grad_conflicts/PG-HyperMARL"] = wandb.Table(
                data=hypernet_table_data,
                columns=["seed", "step", "gradient_conflict"]
            )

    # Log all tables to wandb
    wandb.log(table_dict)