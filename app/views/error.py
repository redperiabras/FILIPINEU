from flask import render_template
from app import app


@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', message='Forbidden', status="403"), 403


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message='Not Found', status="404"), 404


@app.errorhandler(410)
def gone(e):
    return render_template('error.html', message='Gone', status="410"), 410


@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', message='Internal Error', status="500"), 500


@app.errorhandler(503)
def service_unavailable(e):
	return render_template('error.html', message='Service Unavailable', status="503"), 503
