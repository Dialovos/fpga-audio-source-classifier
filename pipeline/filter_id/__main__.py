"""
Command line for the filter ID model.

    python -m pipeline.filter_id predict --model model.joblib clip.wav [more.wav ...]

predict prints one JSON line per WAV file with the predicted filter. Training
isn't in the repo yet.

This file has two groups:
    * Predicting
    * The command line
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import soundfile as sf

from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import load_model

# ********************************
# Predicting
# ********************************


# Reads a WAV file. It stops with a message if the file isn't mono or isn't at
# the model's sample rate.
def _read_mono_wav(path: str, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float64")

    if audio.ndim != 1:
        sys.exit(f"{path}: expected a mono WAV file")

    if rate != sample_rate:
        sys.exit(f"{path}: {rate} Hz, but the model was trained at {sample_rate} Hz")

    return audio


# Runs the predict command: it prints one JSON line with the predicted filter for
# each WAV file.
def predict(args) -> None:
    model, settings = load_model(args.model)

    for path in args.wavs:
        audio = _read_mono_wav(path, settings["sample_rate"])

        # The features have to be made the same way as in training.
        features = spectrum_features(audio, settings["sample_rate"], **settings["features"])
        result = model.predict(features)[0]

        print(json.dumps({"file": str(path), **result}))


# ********************************
# The command line
# ********************************


# Returns the parser that reads the predict command and its options.
def _build_parser() -> argparse.ArgumentParser:
    description = __doc__.strip().split("\n\n")[0]
    parser = argparse.ArgumentParser(prog="python -m pipeline.filter_id", description=description)
    commands = parser.add_subparsers(dest="command", required=True)

    predict_parser = commands.add_parser("predict", help="predict the filter on WAV files")
    predict_parser.add_argument("--model", required=True, help="a trained model.joblib")
    predict_parser.add_argument("wavs", nargs="+", help="mono WAV files at the model's sample rate")
    predict_parser.set_defaults(run=predict)

    return parser


# Reads the command line and runs the chosen command.
def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
