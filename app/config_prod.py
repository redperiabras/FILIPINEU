import logging

from app.config_common import *

PORT = 80

# DEBUG has to be to False in a production environment for security reasons
DEBUG = False

LOG_LEVEL = logging.INFO
LOG_FILENAME = 'activity.log'
