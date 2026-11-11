"""Command-line interface: ``blcnn predict AUDIO_DIR -o results.csv``."""

import argparse
import sys

from . import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="blcnn",
        description="Binaural sound localization with the pretrained networks of "
                    "Francl & McDermott (2022).")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = parser.add_subparsers(dest="command")

    predict = sub.add_parser(
        "predict",
        help="predict source locations for a folder of binaural .wav files",
        description="Predict the source location of each binaural .wav file in AUDIO_DIR. "
                    "Each file should be 2 s long; the model judges its middle 1 s.")
    predict.add_argument("audio_dir", help="folder with binaural (stereo) .wav files")
    predict.add_argument("-o", "--output", default="predictions.csv",
                         help="output CSV (default: %(default)s)")
    predict.add_argument("--weights-dir", default="weights",
                         help="folder containing the net1 ... net10 folders "
                              "(default: %(default)s)")
    predict.add_argument("--nets", default="all",
                         help="networks to use, e.g. '1,3,8' or 'all' (default: %(default)s)")
    predict.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                         help="device to run on (default: %(default)s)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "predict":
        print("blcnn predict is not implemented yet.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
