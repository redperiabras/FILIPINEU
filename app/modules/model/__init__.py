def opts(subparser):
    parser = subparser.add_parser(
        'model', help="Modeling Options")

    parser.set_defaults(all="model")
