"""
Command line for the filter ID model.

    python -m pipeline.filter_id train --out results/<YYYY-MM-DD>-<your-name>-<short-name>
    python -m pipeline.filter_id predict --model results/<run>/model.joblib clip.wav [more.wav ...]

train builds the examples from the WAV files in the settings' "source_dir", holds
out whole recordings for testing, trains, and writes model.joblib, metrics.json,
and settings.json to --out. predict prints one JSON line per WAV file with the
predicted filter.

This file has three groups:
    * Training
    * Predicting
    * The command line
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.filter_id.dataset import build_dataset
from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import (
    FilterIdModel,
    _pick_rows,
    evaluate,
    load_model,
    save_model,
    split_by_recording,
)
from pipeline.records import load_settings

# The settings train uses when --settings isn't given.
DEFAULT_SETTINGS = Path(__file__).with_name("settings.json")

# ********************************
# Training
# ********************************


# Prints how many examples have been built so far.
def _print_progress(done: int, total: int) -> None:
    print(f"built {done}/{total} examples", flush=True)


# Runs the train command: it builds the examples, trains on most of the
# recordings, scores on the ones it held out, and saves the model, the settings,
# and the scores to a new folder. It stops with a message if the folder already
# exists, so an earlier run never gets overwritten, or if the settings or the WAV
# files are bad.
def train(args) -> None:
    settings = load_settings(args.settings)
    out = Path(args.out)

    if out.exists():
        sys.exit(f"{out} already exists; pick a new folder for --out")

    started = time.perf_counter()

    try:
        data = build_dataset(settings, progress=_print_progress)
    except ValueError as error:
        sys.exit(f"can't build the training examples: {error}")

    # Every recording goes entirely to the train side or the test side.
    train_rows, test_rows = split_by_recording(data.groups, settings["test_fraction"], settings["seed"])

    model = FilterIdModel(seed=settings["seed"])
    model.fit(data.features[train_rows], _pick_rows(data.labels, train_rows))

    test_labels = _pick_rows(data.labels, test_rows)
    test_records = _pick_rows(data.records, test_rows)
    metrics = evaluate(model, data.features[test_rows], test_labels, test_records)

    # Add a few counts so the scores are easy to put in context, and the held-out
    # recordings so two machines can check they tested on the same ones.
    test_recordings = sorted(set(data.groups[test_rows].tolist()))
    metrics["train_examples"] = len(train_rows)
    metrics["train_recordings"] = len(set(data.groups[train_rows].tolist()))
    metrics["test_recordings"] = len(test_recordings)
    metrics["test_recording_ids"] = test_recordings
    metrics["seconds"] = round(time.perf_counter() - started, 1)

    out.mkdir(parents=True)
    save_model(out / "model.joblib", model, settings)
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"type accuracy {metrics['type_accuracy']:.3f}, type + order {metrics['type_and_order_accuracy']:.3f}")
    print(f"saved to {out}")


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


# Returns the parser that reads the train and predict commands and their options.
def _build_parser() -> argparse.ArgumentParser:
    description = __doc__.strip().split("\n\n")[0]
    parser = argparse.ArgumentParser(prog="python -m pipeline.filter_id", description=description)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train a model on the WAV files in the settings' source_dir")
    train_parser.add_argument("--settings", default=DEFAULT_SETTINGS, help="a settings JSON file")
    train_parser.add_argument("--out", required=True, help="a new folder for the model, settings, and scores")
    train_parser.set_defaults(run=train)

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
