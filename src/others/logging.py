"""
    logging
"""
from __future__ import absolute_import
import logging

# Used in distributed training
logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
        Initial logger for console and file
    '''
    # Initial a logger
    logger = logging.getLogger()  # pylint: disable=redefined-outer-name

    # Set log level (NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL)
    logger.setLevel(logging.INFO)

    # Set log format
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")

    # logging in console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    # logging in file
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)  # record less info in file
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
