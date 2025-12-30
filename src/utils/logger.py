import logging
import colorlog
import os
from src.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Returns a configured logger with 12h AM/PM format and file logging."""
    
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG)

    # --- 1. Console Handler (Colored) ---
    console_handler = logging.StreamHandler()
    
    # %I is 12-hour clock, %p is AM/PM
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p", 
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
            'ERROR': 'red', 'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- 2. File Handler (Plain Text for data/logs) ---
    log_dir = os.path.join(settings.BASE_DIR, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "blackbox.log")

    file_handler = logging.FileHandler(file_path)
    # Plain format for file (colors don't work well in .log files)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger