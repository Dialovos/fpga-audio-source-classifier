"""
Check the PANNs frame features without PyTorch or the real checkpoint.

A small stand-in takes CNN14's place and gives fixed numbers, so these run in
the normal test run with nothing downloaded. pipeline/tests/check_panns.py
checks the real model on its own.

This file has two groups:
    * Stand-in model
    * Resampling, lining up, and switches
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.event_model import pretrained
from pipeline.event_model.features import frame_count
from pipeline.event_model.labels import EVENT_TYPES
from pipeline.event_model.model import EventModel, clip_features, load_model, prepare_clip, save_model

# ********************************
# Stand-in model
# ********************************

FS = 192000
SETTINGS_PATH = Path(__file__).parents[1] / "event_model" / "settings.json"


# Takes CNN14's place. It's only here so the tests can check it never ends up
# inside a saved event model.
class StandInPanns:
    pass


# Returns one row every 10 ms for 32 kHz audio, like CNN14 (one row per 320
# samples, plus one). Row r, column c holds r + c / 1000, so a test can tell
# which row landed on which frame.
def stand_in_rows(model, audio_32k):
    rows = len(audio_32k) // 320 + 1
    row_numbers = np.arange(rows)[:, None]
    column_numbers = np.arange(pretrained.PANNS_CLASSES)[None, :]

    return row_numbers + column_numbers / 1000


# Returns the default settings with PANNs turned on or off as asked.
def settings_with(use_ours, use_panns):
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["pretrained"]["use_ours"] = use_ours
    settings["pretrained"]["use_panns"] = use_panns

    return settings


# Puts the stand-in in place of loading and running CNN14 for one test.
@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(pretrained, "load_panns", lambda checkpoint_path: StandInPanns())
    monkeypatch.setattr(pretrained, "run_panns", stand_in_rows)


# ********************************
# Resampling, lining up, and switches
# ********************************


# Tests: 192 kHz audio comes out at 32 kHz with one sixth of the samples, and a
# 1 kHz tone stays at 1 kHz.
# Why: CNN14 was trained at 32 kHz, so audio at any other rate would sound
# wrong to it.
def test_to_32k_keeps_length_and_pitch():
    time = np.arange(FS) / FS
    tone = np.sin(2 * np.pi * 1000 * time)
    resampled = pretrained.to_32k(tone, FS)
    spectrum = np.abs(np.fft.rfft(resampled))
    peak_hz = np.argmax(spectrum) * pretrained.PANNS_RATE / len(resampled)

    assert len(resampled) == FS // 6
    assert peak_hz == pytest.approx(1000, abs=1)


# Tests: our frame k gets the PANNs row nearest k x hop / sample_rate, and
# frames past the last row repeat it.
# Why: a PANNs row lined up with the wrong frame would mark events at the wrong
# time.
def test_line_up_matches_frame_times_and_repeats_the_end():
    rows = np.arange(5)[:, None] * np.ones((1, 3))

    same_step = pretrained.line_up(rows, 8, 1920, FS)
    double_step = pretrained.line_up(rows, 4, 3840, FS)

    assert same_step.shape == (8, 3)
    assert list(same_step[:, 0]) == [0, 1, 2, 3, 4, 4, 4, 4]
    assert list(double_step[:, 0]) == [0, 2, 4, 4]


# Tests: the switches pick our features, the PANNs columns, or both with ours
# first, and every choice has one row per frame.
# Why: the model has to read the same columns in training and in detect, or
# its trees would read the wrong numbers.
def test_clip_features_switches_and_column_order(stand_in):
    audio = np.random.default_rng(7).standard_normal(FS // 2)
    n_frames = frame_count(len(audio), 1920)

    ours, ours_frames = clip_features(audio, settings_with(True, False))
    panns, panns_frames = clip_features(audio, settings_with(False, True))
    both, both_frames = clip_features(audio, settings_with(True, True))

    assert ours_frames == panns_frames == both_frames == n_frames
    assert len(ours) == n_frames
    assert panns.shape == (n_frames, pretrained.PANNS_CLASSES)
    assert both.shape == (n_frames, ours.shape[1] + pretrained.PANNS_CLASSES)
    np.testing.assert_array_equal(both[:, :ours.shape[1]], ours)
    np.testing.assert_array_equal(both[:, ours.shape[1]:], panns)

    # The hop is 10 ms like PANNs' step, so frame k gets row k.
    assert panns[3, 0] == 3
    assert panns[3, 2] == pytest.approx(3.002)


# Tests: a missing checkpoint stops with a message that says how to get it.
# Why: a teammate who turns PANNs on before downloading it should know what to
# run instead of seeing a PyTorch error.
def test_missing_checkpoint_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="pretrained download"):
        pretrained.load_panns(tmp_path / "missing.pth")


# Tests: a model trained on both feature sets saves without the PANNs model,
# keeps the checkpoint path in its settings, and detects the same events after
# loading.
# Why: the network is ~300 MB, so the saved model points to it instead of
# carrying a copy.
def test_saved_model_keeps_the_path_not_the_network(stand_in, tmp_path):
    settings = settings_with(True, True)
    settings["model"]["max_iter"] = 2
    clips = []

    for index in range(2):
        audio = np.random.default_rng([7, index]).standard_normal(FS)
        events = []

        for number, kind in enumerate(EVENT_TYPES):
            start_s = 0.1 + 0.1 * number
            events.append({"type": kind, "start_s": start_s, "end_s": start_s + 0.05})

        record = {"example_id": f"clip-{index}", "sources": [{"recording_id": "r"}], "events": events}
        clips.append(prepare_clip(audio, record, settings))

    model = EventModel(EVENT_TYPES, settings, 7).fit(clips)
    path = tmp_path / "model.joblib"
    save_model(path, model, settings)
    loaded, saved_settings = load_model(path)
    clip = clips[0]

    assert b"StandInPanns" not in path.read_bytes()
    assert saved_settings["pretrained"]["checkpoint"] == settings["pretrained"]["checkpoint"]
    assert loaded.detect(clip.features, clip.n_frames, clip.duration_s) == model.detect(
        clip.features, clip.n_frames, clip.duration_s
    )
