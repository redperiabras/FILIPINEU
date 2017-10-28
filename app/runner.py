def opts(subparser):
    parser = subparser.add_parser(
        'runserver', help="Runs API Server")

    parser.add_argument('--host', default="0.0.0.0", help="Host")
    parser.add_argument('--port', default=5000, help="Port Number")
