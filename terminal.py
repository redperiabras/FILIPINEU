from app.modules import parser

import codecs


if __name__ == "__main__":
    # Check what Major Command
    opt = parser.parse_args()

    print(opt)
