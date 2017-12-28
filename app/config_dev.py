import logging

from app.config_common import *

PORT = 5000

# DEBUGcan only be set to True in a development environment for security reasons
DEBUG = True

LOG_LEVEL = logging.DEBUG

SWAGGER = {
	"title": "FILIPINEU API Docs",
	"specs_route": "/api/docs",
	"swagger": "2.0",
		"info": {
			"title": "FILIPINEU",
			"description": "Bidirectional Filipino - English Neural Machine Translation",
			"contact": {
				"responsibleOrganization": "DCS, CCIS - PUP, Manila",
				"responsibleDeveloper": "Red Periabras"
			},
		"version": "0.0.1"
	},
	"basePath": "/api",  # base bash for blueprint registration
	"schemes": [
		"http",
		"https"
	],
	"operationId": "getmyData"
}
