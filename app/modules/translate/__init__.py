def opts(subparser):
    parser = subparser.add_parser(
        'translate', help="Translation Options")

    parser.add_argument('--src', required=True)

    parser.set_defaults(all="translate")
