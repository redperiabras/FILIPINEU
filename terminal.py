# from app import parser

# import codecs


# if __name__ == "__main__":
#     # Check what Major Command
#     opt = parser.parse_args()

from app import app
from arghandler import ArgumentHandler, subcmd


@subcmd('runserver', help="Run the API Server")
def runserver(parser, context, args):
    parser.add_argument('--host', default="0.0.0.0", help="Host")
    parser.add_argument('--port', type=int, default=5000, help="Port")
    parser.add_argument('-d', action='store_true',
                        help="Debugging Mode")

    args = parser.parse_args(args)

    app.run(host=args.host, port=args.port, debug=args.d)


if __name__ == '__main__':
    handler = ArgumentHandler(enable_autocompletion=True)
    handler.run()
