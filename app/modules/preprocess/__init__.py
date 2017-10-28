def opts(subparser):
    parser = subparser.add_parser(
        'init_data', help="Preprocessing options")
    parser.add_argument('-src_loc', required=True,
                        help="Path to the training source data")
    parser.add_argument('-tgt_loc', required=True,
                        help="Path to the training target data")
    parser.add_argument('-train_size', required=True,
                        help="Tell the system how many percent of 100 will \
                        be allotted for training")
    parser.add_argument('-test_size', required=True,
                        help='Tell the system how many percent of 100 will \
                        be alloted for testing')
    parser.add_argument('-eval_size', required=True,
                        help='Tell the system how many percent of 100 will \
                        be alloted for evaluation')
    parser.set_defaults(all="init_data")
