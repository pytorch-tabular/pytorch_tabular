import logging
import os
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    # ch = logging.StreamHandler()
    logger.setLevel(level=os.environ.get("PT_LOGLEVEL", "INFO"))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)

        fmt = "%(asctime)s - {%(name)s:%(lineno)d} - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
