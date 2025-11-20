
"""logger.py: simple logger helper"""
import logging, os, sys
def get_logger(name=__name__, level=logging.INFO, logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if logfile:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            fh = logging.FileHandler(logfile)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
