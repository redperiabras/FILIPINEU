def opts(subparser):
    parser = subparser.add_parser(
        'train', help="Training Options")

    parser.set_defaults(all="train")
