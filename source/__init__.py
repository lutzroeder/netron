""" Python Server entry point """

import argparse
import logging
import os
import sys

from .server import __version__, serve, start, status, stop, wait, widget

__all__ = ["start", "stop", "status", "wait", "serve", "widget", "__version__"]

def main():
    """ main entry point """
    parser = argparse.ArgumentParser(description=
        "Viewer for neural network, deep learning and machine learning models.")
    parser.add_argument("file",
        metavar="MODEL_FILE", help="model file to serve", nargs="?", default=None)
    parser.add_argument("-b", "--browse",
        help="launch web browser", action="store_true")
    parser.add_argument("-p", "--port", help="port to serve", type=int)
    parser.add_argument("--host",
        metavar="ADDR", help="host to serve", default="localhost")
    parser.add_argument("--verbosity",
        metavar="LEVEL", help="log verbosity (quiet, default, debug)",
        choices=[ "quiet", "debug", "default" ], default="default")
    parser.add_argument("--version", help="print version", action="store_true")
    args = parser.parse_args()
    levels = {
        "quiet": logging.CRITICAL,
        "default": logging.INFO,
        "debug": logging.DEBUG,
    }
    logging.basicConfig(level=levels[args.verbosity], format="%(message)s")
    logger = logging.getLogger(__name__)
    if args.file and not os.path.exists(args.file):
        logger.error(f"Model file '{args.file}' does not exist.")
        sys.exit(2)
    if args.version:
        logger.info(__version__)
        sys.exit(0)
    address = (args.host, args.port) if args.host else args.port if args.port else None
    start(args.file, address=address, browse=args.browse)
    wait()
    sys.exit(0)

if __name__ == "__main__":
    main()
