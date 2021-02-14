
import argparse
import sys
import os

from .server import start
from .server import stop
from .server import wait
from .server import serve
from .__version__ import __version__

def main():
    parser = argparse.ArgumentParser(description='Viewer for neural network, deep learning and machine learning models.')
    parser.add_argument('file', metavar='MODEL_FILE', help='model file to serve', nargs='?', default=None)
    parser.add_argument('-v', '--version', help="print version", action='store_true')
    parser.add_argument('-b', '--browse', help='launch web browser', action='store_true')
    parser.add_argument('-p', '--port', help='port to serve', type=int)
    parser.add_argument('--host', help="host to serve")
    parser.add_argument('--log', help='log details to console', action='store_true')
    args = parser.parse_args()
    if args.file and not os.path.exists(args.file):
        print("Model file '" + args.file + "' does not exist.")
        sys.exit(2)
    if args.version:
        print(__version__)
        sys.exit(0)
    address = (args.host, args.port) if args.host else args.port if args.port else None
    start(args.file, address=address, browse=args.browse, log=args.log)
    wait()
    sys.exit(0)

if __name__ == '__main__':
    main()
