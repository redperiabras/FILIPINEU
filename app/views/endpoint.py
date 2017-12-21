from flask import (Blueprint, render_template, redirect, url_for,
                   abort, flash, jsonify)
from flask.ext.login import login_user, logout_user, login_required, current_user
from itsdangerous import URLSafeTimedSerializer
from app import app, models, db
from app.forms import user as user_forms
from app.toolbox import email
from flask import request
import requests
import pprint

# Serializer for generating random tokens
ts = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Create a user blueprint
endpoint = Blueprint('endpoint', __name__, url_prefix='/endpoint')


@endpoint.route('/request_token', methods=['GET'])
def store_token():

    auth_code = requests.get('https://api-uat.unionbankph.com/partners/sb/convergent/v1/oauth2/authorize?client_id=606f2098-9803-4d70-878d-0a08ab3c1f15&response_type=code&scope=balances transfers&redirect_uri=http://localhost:5000/endpoint')

    if 'user' in request.args:
        user = request.args.get('user')
    else:
        user = None

    print(data)

    return redirect(url_for('index', user=user))
    
