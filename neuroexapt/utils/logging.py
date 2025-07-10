"""
Logging utilities for Neuro Exapt.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    log_dir: str,
    epoch: int,
    prefix: str = "metrics"
):
    """
    Log metrics to file.
    
    Args:
        metrics: Dictionary of metrics
        log_dir: Directory to save logs
        epoch: Current epoch
        prefix: File prefix
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_epoch{epoch}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    
    # Save metrics
    with open(filepath, 'w') as f:
        json.dump({
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics
        }, f, indent=2)
        
    # Also update latest metrics
    latest_path = os.path.join(log_dir, f"{prefix}_latest.json")
    with open(latest_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics
        }, f, indent=2) 