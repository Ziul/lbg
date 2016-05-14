from lbgargs import _parser

(_options, _args) = _parser.parse_args()


def main():
    if (not _options.filename) and (not _options.text):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return


if __name__ == '__main__':
    main()
