import os
import uuid
import codecs

from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy_utils import UUIDType
from flask_login import UserMixin
from datetime import datetime

from app import db, bcrypt


class User(db.Model, UserMixin):

	''' A user who has an account on the website. '''

	__tablename__ = 'users'

	first_name = db.Column(db.String)
	last_name = db.Column(db.String)
	phone = db.Column(db.String)
	email = db.Column(db.String, primary_key=True)
	key = db.Column(db.String, default=codecs.encode(os.urandom(24), 'hex').decode())
	confirmation = db.Column(db.Boolean)
	_password = db.Column(db.String)

	def __unicode__(self):
		return self.key

	def __repr__(self):
		return self.key
		# return '<User %r>' % (self.key)

	@property
	def full_name(self):
		return '{} {}'.format(self.first_name, self.last_name)

	@hybrid_property
	def password(self):
		return self._password

	@password.setter
	def _set_password(self, plaintext):
		self._password = bcrypt.generate_password_hash(plaintext)

	def check_password(self, plaintext):
		return bcrypt.check_password_hash(self.password, plaintext)

	def get_id(self):
		return self.email

	def get_key(self):
		return self.key

class Input(db.Model):
	__tablename__ = 'input'

	id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
	lang_id = db.Column(db.String, nullable=False)
	text = db.Column(db.Text(4294967295), nullable=False)

	def __unicode__(self):
		return self.text

	def __repr__(self):
		return self.text
		# return '<Input %r>' % (self.text)

	def get_id(self):
		return self.id

class Output(db.Model):
	__tablename__ = 'output'

	id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
	lang_id = db.Column(db.String, nullable=False)
	text = db.Column(db.Text(4294967295), nullable=False)

	def __unicode__(self):
		return self.text

	def __repr__(self):
		return self.text
		# return '<Output %r>' % (self.text)

	def get_id(self):
		return self.id

class Translation(db.Model):
	__tablename__ = 'translations'

	id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
	key = db.Column(db.String, db.ForeignKey('users.key', ondelete="CASCADE", onupdate="CASCADE"))
	input_id = db.Column(UUIDType(binary=False), db.ForeignKey('input.id', ondelete="CASCADE", onupdate="CASCADE"))
	output_id = db.Column(UUIDType(binary=False), db.ForeignKey('output.id', ondelete="CASCADE", onupdate="CASCADE"))
	timestamp = db.Column(db.DateTime, default=datetime.now())
	elapsed_time = db.Column(db.Float)

	user = db.relationship('User', backref='translations')
	input = db.relationship('Input', backref='translations')
	output = db.relationship('Output', backref='translations')

	def get_id(self):
		return self.id
