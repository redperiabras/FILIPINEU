'''
logger_setup.py customizes the app's logging module. Each time an event is
logged the logger checks the level of the event (eg. debug, warning, info...).
If the event is above the approved threshold then it goes through. The handlers
do the same thing; they output to a file/shell if the event level is above their
threshold.
:Example:
		>>> from website import logger
		>>> logger.info('event', foo='bar')
**Levels**:
		- logger.debug('For debugging purposes')
		- logger.info('An event occured, for example a database update')
		- logger.warning('Rare situation')
		- logger.error('Something went wrong')
		- logger.critical('Very very bad')
You can build a log incrementally as so:
		>>> log = logger.new(date='now')
		>>> log = log.bind(weather='rainy')
		>>> log.info('user logged in', user='John')
'''

import datetime as dt
import logging
from logging.handlers import RotatingFileHandler
from termcolor import colored

import pytz

from app import app

class Logger:
	def __init__(self):

		# Set the logging level
		app.logger.setLevel(app.config['LOG_LEVEL'])

		TZ = pytz.timezone(app.config['TIMEZONE'])

		log_formatter = logging.Formatter('FILIPINEU [%(asctime)s]: %(message)s')

		app.logger.handlers[0].setFormatter(log_formatter)
		app.logger.handlers[0].setLevel(logging.DEBUG)

		# Add a handler to write log messages to a file
		if app.config.get('LOG_FILENAME'):
			file_handler = RotatingFileHandler(filename=app.config['LOG_FILENAME'],
											   mode='a',
											   encoding='utf-8')
			file_handler.setLevel(logging.DEBUG)
			file_handler.setFormatter(log_formatter)
			app.logger.addHandler(file_handler)

	def info(self, message, color=None):
		for i in range(0, len(message), 52):
			text = message[0+i:52+i]
			app.logger.info(colored(text, color))

	def warning(self, message, color='red'):
		for i in range(0, len(message, 52)):
			text = message[0+i:52+i]
			app.logger.warning(colored(text, color))




