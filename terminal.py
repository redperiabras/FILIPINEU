from app import app
from app.modules import preprocess, model, train, translate
from app.modules.IO import Data, export_data, dist_data
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

    # assert that the sum of train, test, and eval is 100
    assert sum([args.train_size, args.test_size, args.eval_size]
               ) == 100, 'Suggested sizes not equal to 100%'

    src_lang = Data(args.src_loc, "r")
    tgt_lang = Data(args.tgt_loc, "r")

    # assert that the number of sentences in the two corpus are equal
    assert tgt_lang.total_data == src_lang.total_data, 'Corpus sizes does not match\
        : \n\t Source Language has %d \n\t Target Language has %d' % (src_lang.total_data, tgt_lang.total_data)

    # retrieve data
    src_data = src_lang.tokenized_data()
    tgt_data = tgt_lang.tokenized_data()
    data = list(zip(src_data, tgt_data))

    # initialize list of data
    train_data, test_data, eval_data = dist_data(data, train_size=args.train_size,
                                                 test_size=args.test_size, eval_size=args.eval_size)

    # export data
    export_data("./data/train/", train_data)
    export_data("./data/test/", test_data)
    export_data("./data/evaluate/", eval_data)


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
