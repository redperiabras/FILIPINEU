from app import app
from app.modules import preprocess, model, train, translate
from arghandler import ArgumentHandler, subcmd


@subcmd('runserver', help="Run the API Server")
def runserver(parser, context, args):
    parser.add_argument('--host', default="localhost", help="Host")
    parser.add_argument('--port', type=int, default=5000, help="Port")
    parser.add_argument('-d', action='store_true',
                        help="Debugging Mode")

    args = parser.parse_args(args)

    app.run(host=args.host, port=args.port, debug=args.d)


@subcmd('preprocess', help="Raw data initializer")
def preprocessor(parser, context, args):
    preprocess.opts(parser)

    args = parser.parse_args(args)


@subcmd('translate', help="Raw data initializer")
def translator(parser, context, args):
    translate.opts(parser)

    args = parser.parse_args(args)


@subcmd('train', help="Training commands")
def trainer(parser, context, args):
    args = parser.parse_args(args)


@subcmd('model', help="Model Commads")
def modeling(parser, context, args):
    args = parser.parse_args(args)


if __name__ == '__main__':
    handler = ArgumentHandler(enable_autocompletion=True)
    handler.run()
