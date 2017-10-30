from app import app
from app.modules import preprocess, model, train, translate
from app.modules.IO import File_Handler, Data_Export
from arghandler import ArgumentHandler, subcmd


import random


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

    # suggested sizes
    sizes = [args.train_size, args.test_size, args.eval_size]

    # check the sum of train, test, and eval size
    if sum(sizes) > 100:
        raise ValueError('Exceeded 100%')
    elif sum(sizes) < 100:
        raise ValueError('Less than 100%')
    else:
        pass

    src_lang = File_Handler(args.src_loc, "r")
    tgt_lang = File_Handler(args.tgt_loc, "r")

    # check the number of sentences in the two corpus
    if tgt_lang.get_number_of_lines() != src_lang.get_number_of_lines():
        raise ValueError('Corpus sizes does not match: \n\t Source Language \
            has %d \n\t Target Language has %d' %
                         (src_lang.get_number_of_lines(), tgt_lang.get_number_of_lines()))
    else:
        pass

    # retrieve data
    src_data = src_lang.get_lines()
    tgt_data = tgt_lang.get_lines()

    # initial number of data
    data_len = src_lang.get_number_of_lines()

    # initialize list of data
    train_data = []
    test_data = []
    eval_data = []

    for i in range(1, 4):

        if i is 1:
            # training
            n = round(src_lang.get_number_of_lines()
                      * (sizes[i - 1] / 100))
        elif i is 2:
            # testing
            n = round(src_lang.get_number_of_lines()
                      * (sizes[i - 1] / 100))
        else:
            # evaluation
            n = round(src_lang.get_number_of_lines()
                      * (sizes[i - 1] / 100))

        for j in range(0, n):
            data_len = len(src_data) - 1
            pick = random.randint(0, data_len)

            if i is 1:
                # training
                train_data.append([src_data.pop(pick), tgt_data.pop(pick)])
            elif i is 2:
                # testing
                test_data.append([src_data.pop(pick), tgt_data.pop(pick)])
            else:
                # evaluation
                eval_data.append([src_data.pop(pick), tgt_data.pop(pick)])

    # export data
    Data_Export("./data/train/", train_data)
    Data_Export("./data/test/", test_data)
    Data_Export("./data/evaluate/", eval_data)


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
