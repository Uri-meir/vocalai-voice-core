import logging
import colorlog
import sys

def setup_logging(level=logging.INFO):
    """Configures the logging system with colored output."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = colorlog.getLogger()
    
    # Clear existing handlers to prevent duplicates (uvicorn reload imports module twice)
    logger.handlers.clear()
    
    logger.addHandler(handler)
    logger.setLevel(level)
    
    # Silence some noisy libs if needed
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger
