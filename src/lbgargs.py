import optparse


_parser = optparse.OptionParser(
    usage="""%prog [OPTIONS]
Examples:

Build codebooks with some pack of images:
    lbg-learn <PATH> [FLAGS]

Quantize a file using the codebook from the deep learning:
    lbg <file>

Test a file with a custom quantization:
    lbg-test <file> [FLAGS]
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

_parser.add_option("--error",
                   dest="error",
                   type='long',
                   default=1e-5,
                   help="Error",
                   )

_parser.add_option("-c", "--compress",
                   dest="compress",
                   type='float',
                   default=10,
                   help="Compress value",
                   )
