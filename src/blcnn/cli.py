"""Command-line interface: ``python -m blcnn predict AUDIO_DIR -o results.csv``."""

import argparse
import logging
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
        description="Predict the source location of each binaural .wav file in AUDIO_DIR "
                    "(searched recursively). Each file should be a 2-s stereo recording "
                    "(left, right ear) at 48 kHz; each network judges its middle 1 s. Other "
                    "sample rates are resampled, longer files are cropped to their first "
                    "2 s and shorter ones zero-padded, with a warning in the output.")
    predict.add_argument("audio_dir", help="folder with binaural (stereo) .wav files")
    predict.add_argument("-o", "--output", default="predictions.csv",
                         help="output CSV (default: %(default)s); skipped files are listed "
                              "in <output>_errors.csv")
    predict.add_argument("--weights-dir", default="weights",
                         help="folder containing the net1 ... net10 folders "
                              "(default: %(default)s)")
    predict.add_argument("--nets", default="all",
                         help="networks to use, e.g. '1,3,8' or 'all' (default: %(default)s); "
                              "the output has one row per file and network")
    predict.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                         help="device to run on (default: %(default)s)")
    predict.add_argument("--batch-size", type=int, default=None,
                         help="files per forward pass (default: 1 on CPU, 16 on GPU)")
    predict.add_argument("-q", "--quiet", action="store_true",
                         help="only print warnings and errors")

    example = sub.add_parser(
        "example",
        help="write example stimuli (noise at known locations) to try 'predict' on",
        description="Write binaural example stimuli to OUTPUT_DIR: 2-s pink noise rendered "
                    "with the KEMAR HRTFs at known locations, named azAAA_elEE.wav "
                    "(azimuth counterclockwise, 90 = left). Requires the optional "
                    "dependency slab: pip install \".[examples]\".")
    example.add_argument("output_dir", nargs="?", default="example_stimuli",
                         help="folder for the .wav files (default: %(default)s)")
    example.add_argument("-q", "--quiet", action="store_true",
                         help="only print warnings and errors")
    return parser


def _example(args):
    from .examples import ExamplesError, make_examples

    try:
        paths = make_examples(args.output_dir)
    except ExamplesError as err:
        logging.error("%s", err)
        return 1
    logging.info("%d example stimuli written; try: blcnn predict %s -o example_predictions.csv",
                 len(paths), args.output_dir)
    return 0


def _predict(args):
    from .audio import AudioError
    from .predict import predict_folder
    from .weights import WeightsError

    if args.device == "gpu":
        from .model import gpu_available
        if not gpu_available():
            logging.error("--device gpu requested, but TensorFlow sees no GPU")
            return 1
    batch_size = args.batch_size or (16 if args.device == "gpu" else 1)
    if batch_size < 1:
        logging.error("--batch-size must be at least 1")
        return 1
    try:
        predict_folder(args.audio_dir, args.output, weights_dir=args.weights_dir,
                       nets=args.nets, device=args.device, batch_size=batch_size)
    except (WeightsError, FileNotFoundError, AudioError) as err:
        logging.error("%s", err)
        return 1
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")
    if args.command == "predict":
        return _predict(args)
    if args.command == "example":
        return _example(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
