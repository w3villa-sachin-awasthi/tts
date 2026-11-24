import logging
import colorlog


def setup_logger(name=__name__):
    # Create a logger if it doesn't already exist
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers
    if not logger.handlers:
        handler = colorlog.StreamHandler()

        formatter = colorlog.ColoredFormatter(
            '%(asctime_log_color)s%(asctime)s - %(log_color)s%(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
            secondary_log_colors={
                'asctime': {
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'WARNING': 'white',
                    'ERROR': 'white',
                    'CRITICAL': 'white',
                }
            }
        )

        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

    return logger


logger = setup_logger()

__all__ = ['logger', 'setup_logger']
