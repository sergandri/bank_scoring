import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_LEVEL = logging.INFO
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app_logger')
logger.setLevel(LOG_LEVEL)

