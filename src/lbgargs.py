import optparse


_parser = optparse.OptionParser(
    usage="""%prog [OPTIONS]
Examples:

Encode the file abc.txt:
    ~ $ %prog -f abc.txt -e
    ~ $ %prog abc.txt

Decode the file abc.txt into `teste.txt`:
    ~ $ %prog -f abc.txt -d
    ~ $ %prog -f abc.txt --decode
        """,
    description="Encode/Decode a file using Linde-Buzo-Gray's code",
)

# quiet options
_parser.add_option("-q", "--quiet",
                   dest="verbose",
                   action="store_false",
                   help="suppress non error messages",
                   default=True
                   )

_parser.add_option("-f", "--filename",
                   dest="filename",
                   type='string',
                   help="Name of the file",
                   )

_parser.add_option("-d", "--decode",
                   dest="encode",
                   action="store_false",
                   help="decode a file",
                   default=True
                   )

_parser.add_option("-e", "--encode",
                   dest="encode",
                   action="store_true",
                   help="encode a file. Is the default value",
                   default=True
                   )

_parser.add_option("-t", "--text",
                   dest="text",
                   type='string',
                   help="text to be encoded",
                   )

_parser.add_option("--error",
                   dest="error",
                   type='long',
                   default=1e-5,
                   help="Error",
                   )
