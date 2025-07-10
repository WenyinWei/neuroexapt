"""
Utility functions and helpers for Neuro Exapt.

This package contains:
- Configuration management
- Logging utilities
- Visualization tools
- Helper functions
"""

from .logging import setup_logger, log_metrics
from .visualization import (
    plot_evolution_history,
    plot_entropy_history,
    plot_layer_importance_heatmap,
    plot_information_metrics,
    create_summary_plot
)

__all__ = [
    # Logging
    "setup_logger",
    "log_metrics",
    # Visualization
    "plot_evolution_history",
    "plot_entropy_history",
    "plot_layer_importance_heatmap",
    "plot_information_metrics",
    "create_summary_plot",
] 