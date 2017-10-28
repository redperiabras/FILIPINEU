from flask import Flask
import socket


app = Flask(__name__)
# Insert Here other Initializations


@app.route("/")
def index():
    html = "<h3>FILIPINEU DevOps on Load!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(hostname=socket.gethostname()), 200
