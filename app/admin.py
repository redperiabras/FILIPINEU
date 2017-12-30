import os.path as op

from flask import request, Response
from werkzeug.exceptions import HTTPException
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_admin.contrib.fileadmin import FileAdmin

from app import app, db
from app.models import User, Translation, Input, Output


admin = Admin(app, name='Admin', template_mode='bootstrap3')

class ModelView(ModelView):

	def is_accessible(self):
		auth = request.authorization or request.environ.get('REMOTE_USER')  # workaround for Apache
		if not auth or (auth.username, auth.password) != app.config['ADMIN_CREDENTIALS']:
			raise HTTPException('', Response('You have to an administrator.', 401,
				{'WWW-Authenticate': 'Basic realm="Login Required"'}
			))
		return True

class UserModelView(ModelView):
	column_display_pk = True
	form_columns = ['first_name', 'last_name', 'key', 'confirmation']
	column_searchable_list = ('first_name', 'last_name', 'email')
	column_filters = ('first_name', 'last_name', 'email')

class TranslationModelView(ModelView):
	column_display_pk = True
	form_columns = ['id', 'user','timestamp', 'input', 'output', 'elapsed_time']

# Users
admin.add_view(UserModelView(User, db.session))
admin.add_view(TranslationModelView(Translation, db.session))

# Static files
path = op.join(op.dirname(__file__), 'static')
admin.add_view(FileAdmin(path, '/static/', name='Static'))