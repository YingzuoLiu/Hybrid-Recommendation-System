import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs"):
    """
    Set up logger with custom formatting and file output
    
    Args:
        log_dir: directory to store log files
    
    Returns:
        logging.Logger: configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger('recommender_system')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_{timestamp}.log')
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_hyperparameters(logger, config):
    """
    Log model hyperparameters
    
    Args:
        logger: logging.Logger instance
        config: dictionary containing model configuration
    """
    logger.info("=== Hyperparameters ===")
    for section, params in config.items():
        logger.info(f"\n[{section}]")
        for param, value in params.items():
            logger.info(f"{param}: {value}")

def log_metrics(logger, epoch, metrics, prefix=""):
    """
    Log training/evaluation metrics
    
    Args:
        logger: logging.Logger instance
        epoch: current epoch number
        metrics: dictionary containing metric names and values
        prefix: optional prefix for metric names (e.g., "Train", "Val")
    """
    message = f"Epoch {epoch}"
    if prefix:
        message = f"{prefix} - {message}"
    
    for metric, value in metrics.items():
        message += f" | {metric}: {value:.4f}"
    
    logger.info(message)