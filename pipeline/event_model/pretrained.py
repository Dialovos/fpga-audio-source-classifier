"""
Frame features from a pretrained sound model (PANNs CNN14), lined up with ours.

PANNs CNN14 was trained on AudioSet (~2 million clips, 527 sound classes). We use
the version that gives a probability for each of the 527 classes every 10 ms
(Cnn14_DecisionLevelMax), frozen. It takes 32 kHz audio and hears 50 Hz - 14 kHz,
so our own features still cover below 50 Hz, 14-24 kHz, and sharp timing.

The model only loads when the settings turn it on, so the rest of the event model
and the normal test run never need PyTorch. Get its two files once with:

    python -m pipeline.event_model.pretrained download

That puts the checkpoint (~300 MB) where the settings point (data/models/ by
default, which is gitignored) and AudioSet's label list in ~/panns_data/, which is
where the panns_inference package reads it from when it's imported.

This file has four groups:
    * Getting the files
    * Loading the model
    * Frame features
    * The command line
"""

from __future__ import annotations

import argparse
import urllib.request
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal

CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
LABELS_PATH = Path.home() / "panns_data" / "class_labels_indices.csv"
DEFAULT_CHECKPOINT = "data/models/Cnn14_DecisionLevelMax_mAP=0.385.pth"

# What CNN14 was trained on: 32 kHz audio, one output row every 10 ms, 527 classes.
PANNS_RATE = 32000
PANNS_STEP_S = 0.01
PANNS_CLASSES = 527

# Models already loaded in this run, by checkpoint path, so each loads only once.
_loaded = {}

# ********************************
# Getting the files
# ********************************


# Returns after downloading url to path, unless path is already there. It stops
# with an error from urllib if the download fails.
def _download(url, path) -> None:
    path = Path(path)

    if path.is_file():
        print(f"already have {path.name}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    print(f"downloading {path.name} ...", flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    print(f"saved {path}")


# Returns after making sure the checkpoint and the label list are on this
# computer.
def download(checkpoint_path=DEFAULT_CHECKPOINT) -> None:
    _download(LABELS_URL, LABELS_PATH)
    _download(CHECKPOINT_URL, checkpoint_path)


# ********************************
# Loading the model
# ********************************


# Returns the frozen CNN14 model from a checkpoint, loading it only once per run.
# It stops with a FileNotFoundError that says how to get the files if either one
# is missing.
def load_panns(checkpoint_path):
    checkpoint_path = str(checkpoint_path)

    if checkpoint_path in _loaded:
        return _loaded[checkpoint_path]

    if not Path(checkpoint_path).is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"PANNs files are missing ({checkpoint_path} or {LABELS_PATH}). "
            "Get them with: python -m pipeline.event_model.pretrained download"
        )

    # Imported here, so nothing needs PyTorch unless PANNs is turned on.
    import torch
    from panns_inference.models import Cnn14_DecisionLevelMax

    model = Cnn14_DecisionLevelMax(
        sample_rate=PANNS_RATE, window_size=1024, hop_size=320, mel_bins=64,
        fmin=50, fmax=14000, classes_num=PANNS_CLASSES,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _loaded[checkpoint_path] = model

    return model


# ********************************
# Frame features
# ********************************


# Returns mono audio resampled to 32 kHz, the rate CNN14 was trained at, using
# whole-number up and down factors (192 kHz is 1 up and 6 down).
def to_32k(audio, sample_rate) -> np.ndarray:
    common = gcd(PANNS_RATE, int(sample_rate))

    return signal.resample_poly(audio, PANNS_RATE // common, int(sample_rate) // common)


# Returns the model's 527 class probabilities every 10 ms for 32 kHz audio, as
# a (rows, 527) array.
def run_panns(model, audio_32k) -> np.ndarray:
    import torch

    batch = torch.from_numpy(np.asarray(audio_32k, dtype=np.float32)[None, :])

    with torch.no_grad():
        output = model(batch, None)

    return output["framewise_output"][0].numpy().astype(np.float64)


# Returns PANNs rows lined up with our frames: our frame k, at k x hop /
# sample_rate seconds, gets the PANNs row nearest that time, so the result has
# one row per frame of ours.
def line_up(rows, n_frames, hop, sample_rate) -> np.ndarray:
    features = np.empty((n_frames, rows.shape[1]))

    for frame in range(n_frames):
        time_s = frame * hop / sample_rate
        row = round(time_s / PANNS_STEP_S)
        row = min(max(row, 0), len(rows) - 1)
        features[frame] = rows[row]

    return features


# Returns (n_frames x 527) PANNs probabilities for the audio, lined up with our
# frames. It stops with a FileNotFoundError if the PANNs files are missing.
def panns_features(audio, sample_rate, n_frames, hop, checkpoint_path) -> np.ndarray:
    model = load_panns(checkpoint_path)
    rows = run_panns(model, to_32k(audio, sample_rate))

    return line_up(rows, n_frames, hop, sample_rate)


# ********************************
# The command line
# ********************************


# Returns after reading the command line and running the chosen command.
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pipeline.event_model.pretrained")
    commands = parser.add_subparsers(dest="command", required=True)
    getting = commands.add_parser("download", help="get the PANNs checkpoint and AudioSet's label list")
    getting.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)

    download(args.checkpoint)


if __name__ == "__main__":
    main()
