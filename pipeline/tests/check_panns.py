"""
Checks for the real PANNs model (CNN14) that the event model can use.

These load the real checkpoint, so they need PyTorch, panns_inference, and the two downloaded files, and they use
about 1 GB of memory. That's why the file isn't named test_*.py and the normal test run leaves it out (the normal
run checks the same code with a stand-in, in test_pretrained.py). Get the files, then run it on its own from the
repo root:

    python -m pipeline.event_model.pretrained download
    python -m pytest pipeline/tests/check_panns.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.event_model import events, pretrained
from pipeline.event_model.model import clip_features

FS = 192_000
REPO = Path(__file__).parents[2]
SETTINGS_PATH = REPO / "pipeline" / "event_model" / "settings.json"


# Returns the default event model settings with only the PANNs features on.
def panns_settings():
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["pretrained"]["use_ours"] = False
    settings["pretrained"]["use_panns"] = True

    return settings


# Returns the real CNN14 model, loaded once for this file. If the files are missing, every check fails with how to
# get them.
@pytest.fixture(scope="module")
def model():
    checkpoint = REPO / panns_settings()["pretrained"]["checkpoint"]

    try:
        return pretrained.load_panns(checkpoint)
    except FileNotFoundError as error:
        pytest.fail(str(error))


# Returns AudioSet's 527 class names in the model's column order.
def class_names():
    with pretrained.LABELS_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    names = []

    for row in rows:
        names.append(row["display_name"])

    return names


# Returns the names of the classes with the highest average probability over all rows, highest first.
def top_classes(rows, count):
    average = np.mean(rows, axis=0)
    names = class_names()
    top = []

    for column in np.argsort(average)[::-1][:count]:
        top.append(names[column])

    return top


# Tests: 10 s of audio gives 1001 rows (one every 10 ms, counting both ends) of 527 probabilities.
# Why: line_up assumes one row every 10 ms, and the model reads exactly 527 PANNs columns.
def test_output_has_527_probabilities_every_10_ms(model):
    noise = np.random.default_rng(7).standard_normal(10 * pretrained.PANNS_RATE) * 0.05
    rows = pretrained.run_panns(model, noise)

    assert rows.shape == (1001, pretrained.PANNS_CLASSES)
    assert np.all((rows >= 0) & (rows <= 1))


# Tests: clip_features with only PANNs on gives 300 rows of 527 for 3 s at 192 kHz, one per frame of ours.
# Why: training and detect use clip_features, so the real model has to line up there too, not only in the stand-in.
def test_clip_features_line_up_with_our_frames(model):
    settings = panns_settings()
    settings["pretrained"]["checkpoint"] = str(REPO / settings["pretrained"]["checkpoint"])
    audio = np.random.default_rng(7).standard_normal(3 * FS) * 0.05
    features, n_frames = clip_features(audio, settings)

    assert n_frames == 300
    assert features.shape == (300, pretrained.PANNS_CLASSES)


# Tests: 10 s of a team voiceover has Speech among its top 3 classes.
# Why: if the model can't hear speech in our own recordings, its other answers on them can't be trusted either.
def test_voiceover_scores_high_on_speech(model):
    path = REPO / "data" / "test-data-prelim" / "VO1.wav"

    if not path.is_file():
        pytest.fail(f"Put the team's VO1.wav in {path.parent.relative_to(REPO).as_posix()}/ first.")

    audio, sample_rate = sf.read(path, frames=10 * FS, dtype="float64")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    rows = pretrained.run_panns(model, pretrained.to_32k(audio, sample_rate))
    top = top_classes(rows, 3)
    print("voiceover top classes:", top)

    assert "Speech" in top


# Tests: our generated hum (a 120 Hz tone with 4 harmonics) has Hum or Mains hum among its top 5 classes.
# Why: it shows the pretrained model hears our made-up hum as a hum, so its features can help that class.
def test_generated_hum_scores_high_on_hum(model):
    settings = {"duration_s": [3, 3], "freq_hz": [120, 120], "fade_s": [0.1, 0.1]}
    hum, _ = events.hum(settings, FS, np.random.default_rng(7))
    hum *= 0.1 / np.sqrt(np.mean(hum**2))

    rows = pretrained.run_panns(model, pretrained.to_32k(hum, FS))
    top = top_classes(rows, 5)
    print("hum top classes:", top)

    assert "Hum" in top or "Mains hum" in top
