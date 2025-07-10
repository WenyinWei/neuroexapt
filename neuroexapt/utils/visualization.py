"""
Visualization utilities for Neuro Exapt.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import seaborn as sns
from datetime import datetime


def plot_evolution_history(
    evolution_history: List[Any],
    save_path: Optional[str] = None
):
    """
    Plot evolution history showing structural changes over time.
    
    Args:
        evolution_history: List of evolution steps
        save_path: Optional path to save plot
    """
    if not evolution_history:
        return
        
    # Extract data
    epochs = [step.epoch for step in evolution_history]
    params_before = [step.parameters_before for step in evolution_history]
    params_after = [step.parameters_after for step in evolution_history]
    actions = [step.action for step in evolution_history]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot parameter evolution
    ax1.plot(epochs, params_before, 'o-', label='Before', alpha=0.7)
    ax1.plot(epochs, params_after, 's-', label='After', alpha=0.7)
    
    # Add action annotations
    for i, (epoch, action) in enumerate(zip(epochs, actions)):
        if action != 'none':
            ax1.annotate(
                action,
                xy=(epoch, params_after[i]),
                xytext=(epoch, params_after[i] + 1000),
                arrowprops=dict(arrowstyle='->', alpha=0.5),
                fontsize=8,
                ha='center'
            )
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('Parameter Evolution During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot efficiency gains
    efficiency_gains = [step.efficiency_gain() for step in evolution_history]
    colors = ['green' if g > 0 else 'red' for g in efficiency_gains]
    
    ax2.bar(epochs, efficiency_gains, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Efficiency Gain')
    ax2.set_title('Structural Evolution Efficiency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_entropy_history(
    entropy_history: List[float],
    threshold_history: List[float],
    save_path: Optional[str] = None
):
    """
    Plot entropy evolution with adaptive threshold.
    
    Args:
        entropy_history: List of entropy values
        threshold_history: List of threshold values
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot entropy
    plt.plot(entropy_history, label='Entropy', alpha=0.8, linewidth=2)
    
    # Plot threshold
    if len(threshold_history) > len(entropy_history):
        threshold_history = threshold_history[:len(entropy_history)]
    elif len(threshold_history) < len(entropy_history):
        # Extend threshold history
        last_threshold = threshold_history[-1] if threshold_history else 0.5
        threshold_history.extend([last_threshold] * (len(entropy_history) - len(threshold_history)))
        
    plt.plot(threshold_history, '--', label='Threshold', color='red', alpha=0.8)
    
    # Add shaded regions
    plt.fill_between(
        range(len(entropy_history)),
        0,
        threshold_history,
        alpha=0.2,
        color='red',
        label='Pruning Zone'
    )
    
    plt.xlabel('Training Step')
    plt.ylabel('Entropy')
    plt.title('Adaptive Entropy Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_layer_importance_heatmap(
    layer_importances: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot heatmap of layer importances over time.
    
    Args:
        layer_importances: Dictionary mapping layer names to importance history
        save_path: Optional path to save plot
    """
    if not layer_importances:
        return
        
    # Prepare data
    layer_names = list(layer_importances.keys())
    importance_matrix = []
    
    max_len = max(len(hist) for hist in layer_importances.values())
    
    for name in layer_names:
        history = layer_importances[name]
        # Pad if necessary
        if len(history) < max_len:
            history = history + [history[-1]] * (max_len - len(history))
        importance_matrix.append(history)
    
    importance_matrix = np.array(importance_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        importance_matrix,
        xticklabels=False,
        yticklabels=layer_names,
        cmap='YlOrRd',
        cbar_kws={'label': 'Importance'},
        vmin=0
    )
    
    plt.xlabel('Training Step')
    plt.ylabel('Layer')
    plt.title('Layer Importance Evolution')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_information_metrics(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot information-theoretic metrics over training.
    
    Args:
        metrics_history: Dictionary of metric histories
        save_path: Optional path to save plot
    """
    n_metrics = len(metrics_history)
    if n_metrics == 0:
        return
        
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2,
        2,
        figsize=(12, 4 * ((n_metrics + 1) // 2))
    )
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[idx]
        
        ax.plot(values, alpha=0.8)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add moving average
        if len(values) > 10:
            window = min(50, len(values) // 10)
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(values)), ma, 'r--', alpha=0.8, label='MA')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def create_summary_plot(
    ne_instance,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive summary plot of Neuro Exapt training.
    
    Args:
        ne_instance: NeuroExapt instance
        save_path: Optional path to save plot
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Evolution history (top-left, 2x1)
    ax1 = fig.add_subplot(gs[0:2, 0])
    if ne_instance.evolution_history:
        epochs = [step.epoch for step in ne_instance.evolution_history]
        params = [step.parameters_after for step in ne_instance.evolution_history]
        ax1.plot(epochs, params, 'o-')
        ax1.set_title('Parameter Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Parameters')
    else:
        ax1.text(0.5, 0.5, 'No evolution history', ha='center', va='center')
    
    # 2. Entropy history (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if ne_instance.entropy_ctrl.entropy_history:
        ax2.plot(ne_instance.entropy_ctrl.entropy_history[-100:])
        ax2.set_title('Recent Entropy')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Entropy')
    else:
        ax2.text(0.5, 0.5, 'No entropy data', ha='center', va='center')
    
    # 3. Current metrics (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics_text = f"""Current Status:
    Epoch: {ne_instance.current_epoch}
    Entropy Threshold: {ne_instance.entropy_ctrl.threshold:.3f}
    Evolution Steps: {len(ne_instance.evolution_history)}
    """
    ax3.text(0.1, 0.8, metrics_text, va='top', fontsize=12)
    
    # 4. Layer importance (middle row, full width)
    ax4 = fig.add_subplot(gs[1, :])
    layer_importances = ne_instance.info_theory._activation_cache.get('layer_importances', {})
    if layer_importances:
        names = list(layer_importances.keys())[:20]  # Limit to 20 layers
        values = [layer_importances[n] for n in names]
        ax4.barh(names, values)
        ax4.set_xlabel('Importance')
        ax4.set_title('Current Layer Importances')
    else:
        ax4.text(0.5, 0.5, 'No layer importance data', ha='center', va='center')
    
    # 5. Evolution summary (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    if ne_instance.evolution_history:
        actions = {}
        for step in ne_instance.evolution_history:
            actions[step.action] = actions.get(step.action, 0) + 1
        
        ax5.bar(list(actions.keys()), list(actions.values()))
        ax5.set_title('Evolution Action Summary')
        ax5.set_xlabel('Action Type')
        ax5.set_ylabel('Count')
    else:
        ax5.text(0.5, 0.5, 'No evolution actions', ha='center', va='center')
    
    # Add title
    fig.suptitle(
        f'Neuro Exapt Training Summary - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        fontsize=16
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close() 