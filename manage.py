from app import app
from flask_script import Manager, Shell, Server


def context():
    return dict(app=app)


manager = Manager(app)
manager.add_command("run", Server(host="0.0.0.0", port="5000", debug=True))
manager.add_command("shell", Shell(make_context=context))


if __name__ == "__main__":
    manager.run()
