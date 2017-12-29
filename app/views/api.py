from flask import (Blueprint, render_template, redirect, url_for,
				   abort, flash, jsonify, request)
from flask.ext.login import login_user, logout_user, login_required
from itsdangerous import URLSafeTimedSerializer
from app import app, models, db
from app.forms import user as user_forms
from app.toolbox import email

from models import en_fl, fl_en
from modules.sentence import tokenizer, detokenize
from modules.text import Encoded
from time import time
from datetime import datetime
from langdetect import detect

# Serializer for generating random tokens
ts = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Create an api blueprint
api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/')
def index():
	return render_template('index.html', title='API')

@api.route('/signup', methods=['GET', 'POST'])
def signup():
	form = user_forms.SignUp()
	if form.validate_on_submit():
		# Create a user who hasn't validated his email address
		user = models.User(
			first_name=form.first_name.data,
			last_name=form.last_name.data,
			phone=form.phone.data,
			email=form.email.data,
			confirmation=False,
			password=form.password.data,
		)
		# Insert the user in the database
		db.session.add(user)
		db.session.commit()
		# Subject of the confirmation email
		subject = 'Please confirm your email address.'
		# Generate a random token
		token = ts.dumps(user.email, salt='email-confirm-key')
		# Build a confirm link with token
		confirmUrl = url_for('api.confirm', token=token, _external=True)
		# Render an HTML template to send by email
		html = render_template('email/confirm.html',
							   confirm_url=confirmUrl)
		# Send the email to user
		email.send(user.email, subject, html)
		# Send back to the home page
		flash('Check your emails to confirm your email address.', 'positive')
		return redirect(url_for('index'))
	return render_template('user/signup.html', form=form, title='Sign up')


@api.route('/confirm/<token>', methods=['GET', 'POST'])
def confirm(token):
	try:
		email = ts.loads(token, salt='email-confirm-key', max_age=86400)
	# The token can either expire or be invalid
	except:
		abort(404)
	# Get the user from the database
	user = models.User.query.filter_by(email=email).first()
	# The user has confirmed his or her email address
	user.confirmation = True
	# Update the database with the user
	db.session.commit()
	# Send to the signin page
	flash(
		'Your email address has been confirmed, you can sign in.', 'positive')
	return redirect(url_for('api.signin'))


@api.route('/signin', methods=['GET', 'POST'])
def signin():
	form = user_forms.Login()
	if form.validate_on_submit():
		user = models.User.query.filter_by(email=form.email.data).first()
		# Check the user exists
		if user is not None:
			# Check the password is correct
			if user.check_password(form.password.data):
				login_user(user)
				# Send back to the home page
				flash('Succesfully signed in.', 'positive')
				return redirect(url_for('index'))
			else:
				flash('The password you have entered is wrong.', 'negative')
				return redirect(url_for('api.signin'))
		else:
			flash('Unknown email address.', 'negative')
			return redirect(url_for('api.signin'))
	return render_template('user/signin.html', form=form, title='Sign in')


@api.route('/signout')
def signout():
	logout_user()
	flash('Succesfully signed out.', 'positive')
	return redirect(url_for('index'))


@api.route('/account')
@login_required
def account():
	return render_template('user/account.html', title='Account')


@api.route('/forgot', methods=['GET', 'POST'])
def forgot():
	form = user_forms.Forgot()
	if form.validate_on_submit():
		user = models.User.query.filter_by(email=form.email.data).first()
		# Check the user exists
		if user is not None:
			# Subject of the confirmation email
			subject = 'Reset your password.'
			# Generate a random token
			token = ts.dumps(user.email, salt='password-reset-key')
			# Build a reset link with token
			resetUrl = url_for('api.reset', token=token, _external=True)
			# Render an HTML template to send by email
			html = render_template('email/reset.html', reset_url=resetUrl)
			# Send the email to user
			email.send(user.email, subject, html)
			# Send back to the home page
			flash('Check your emails to reset your password.', 'positive')
			return redirect(url_for('index'))
		else:
			flash('Unknown email address.', 'negative')
			return redirect(url_for('api.forgot'))
	return render_template('user/forgot.html', form=form)


@api.route('/reset/<token>', methods=['GET', 'POST'])
def reset(token):
	try:
		email = ts.loads(token, salt='password-reset-key', max_age=86400)
	# The token can either expire or be invalid
	except:
		abort(404)
	form = user_forms.Reset()
	if form.validate_on_submit():
		user = models.User.query.filter_by(email=email).first()
		# Check the user exists
		if user is not None:
			user.password = form.password.data
			# Update the database with the user
			db.session.commit()
			# Send to the signin page
			flash('Your password has been reset, you can sign in.', 'positive')
			return redirect(url_for('api.signin'))
		else:
			flash('Unknown email address.', 'negative')
			return redirect(url_for('api.forgot'))
	return render_template('user/reset.html', form=form, token=token)


@api.route('/translate', methods=['POST'])
def translate():
	"""
	Endpoint for translating texts
	---

	tags:
	- translate

	parameters:

	  - name: text
	    description: Text input to be translated 
	    in: formData
	    type: string
	    required: true

	  - name: lang
	    description: Language ID of the text input
	    in: formData
	    type: string
	    enum: ['fl', 'en']
	    default: 'en'
	    required: false

	  - name: key
	    description: Client API key
	    in: query
	    type: string
	    example: asdf1234ghjk5678
	    default: asdf1234ghjk5678

	definitions:
		Text:
			type: object
			properties:
			    lang:
			        type: string
			        description: Language ID
			        required: true
			        example: en
			    text:
			        type: string
			        description: Input Language Text
			        required: true
			        example: Hello there!
		Translation:
			type: object
			properties:
			    input:
			        type: object
			        $ref: '#/definitions/Text'
			        required: true
			    output:
			        type: object
			        $ref: '#/definitions/Text'
			        required: true
			    elapsed time:
			    	type: float
			    	descrption: Time elapsed in translating (seconds)
			    	required: true
			    timestamp:
			        type: string
			        description: Timestamp
			        required: true
			        form: timestamp
		Success:
			type: object
			properties:
			    status:
			        type: integer
			        description: HTTP Response Code
			        required: true
			    description:
			        type: string
			        description: HTTP Response Description
			        required: true
			    message:
			        type: string
			        description: System Message
			        required: true
			    data:
			        type: object
			        $ref: '#/definitions/Translation'
			        required: true
		Error:
			type: object
			properties:
			    status:
			        type: integer
			        description: HTTP Response Code
			        required: true
			    description:
			        type: string
			        description: HTTP Response Description
			        required: true
			    message:
			        type: string
			        description: System Message
			        required: true
	responses:
		200:
			description: Success
			schema:
				$ref: '#/definitions/Success'
		400:
			description: Bad Request
			schema:
				$ref: '#/definitions/Error'
		403:
			description: Fobidden
			schema:
				$ref: '#/definitions/Error'
	"""

	#Get provided api key
	if 'key' not in request.args:
		return jsonify(
			{
				'status': 400,
				'description': 'Bad Request',
				'message': 'Client Key required'
			}), 400
	else:
		api_key = request.args.get('key')

	    #TODO Verify the api key

	if request.form.get('lang') not in ('en', 'fl', None):

		return jsonify(
			{
				'status': 400,
				'description': 'Bad Request',
				'message': 'Invalid Language ID'
			}), 400
	elif request.form.get('lang') is not None:

		lang_id = request.form.get('lang')
	else:

		lang_id = 'en'
		if detect(request.form.get('text')) != 'en':
			lang_id = 'fl'

	model = en_fl if lang_id == 'en' else fl_en

	if 'text' not in request.form:
		return jsonify(
			{
				'status': 400,
				'description': 'Bad Request',
				'message': 'Text to be translated required'
			}), 400
	else:
		input_text = request.form.get('text')

	# Translator

	MAX_TARGET_LEN = 1000
	BEAM_SIZE = 8

	def translate(sent, encode=False, nbest=0, backwards=False):
		
		if encode:
			sent = [model.config['source_encoder'].encode_sequence(sent)]

		x = model.config['source_encoder'].pad_sequences(
				sent, fake_hybrid=True)
		
		beams = model.search(
				*(x + (MAX_TARGET_LEN,)),
				beam_size=BEAM_SIZE,
				prune=(nbest == 0))

		nbest = min(nbest, BEAM_SIZE)

		for batch_sent_idx, (_, beam) in enumerate(beams):
			lines = []
			for best in list(beam)[:max(1, nbest)]:
				encoded = Encoded(best.history + (best.last_sym,), None)
				decoded = model.config['target_encoder'].decode_sentence(encoded)
				hypothesis = detokenize(
					decoded[::-1] if backwards else decoded,
					model.config['target_tokenizer'])
				if nbest > 0:
					lines.append(' ||| '.join((str(i+batch_sent_idx), hypothesis, str(best.norm_score))))
				else:
					yield hypothesis
			if lines:
				yield '\n'.join(lines)

	t0 = time()
	
	sent_tokenizer = tokenizer(model.config['source_tokenizer'],
		lowercase=model.config['source_lowercase'])

	input_tokens = sent_tokenizer(input_text)

	result = translate(
                input_tokens, encode=True, nbest=0, backwards=model.config['backwards'])
	
	elapsed_time = time() - t0

	return jsonify({
		'data' : {
			'input' : {
				'lang' : lang_id,
				'text' : input_text
			},
			'output' : {
				'lang' : 'en' if lang_id == 'fl' else 'fl',
				'text' : result.__next__()
			},
			'timestamp' : datetime.now(),
			'elapsed_time' : elapsed_time
		},
		'description' : 'Success',
		'message' : 'Translation Response',
		'status' : 200
		}), 200
