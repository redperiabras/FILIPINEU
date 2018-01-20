from flask import render_template, jsonify
from app import app
import random

@app.route('/')
def index():
	return render_template('index.html', title='Home')

@app.route('/about')
def about():
	return render_template('about.html', title='About')
