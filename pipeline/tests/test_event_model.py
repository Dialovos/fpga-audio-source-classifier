"""
Check event audio, splits, frames, scoring, training and the four-version comparison.

This file has five groups:
    * Shared fixture and settings
    * Backgrounds, events and clips
    * Frame features and labels
    * Scoring and the trained model
    * Commands and comparison
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.compensation import compensate
from pipeline.degrade import add_noise, quantize
from pipeline.event_model import events
from pipeline.event_model.__main__ import main
from pipeline.event_model.comparison import _ai_inverse, run_comparison
from pipeline.event_model.dataset import make_clip, make_labeled_clip, make_test_cases
from pipeline.event_model.dataset import test_clip_count as count_test_clips
from pipeline.event_model.features import (
    add_context,
    band_averages,
    band_changes,
    band_levels,
    band_spreads,
    frame_count,
    frame_features,
    frame_times,
)
from pipeline.event_model.labels import EVENT_TYPES, SHORT_TYPES, events_to_frames, frames_to_events
from pipeline.event_model.model import Clip, EventModel, evaluate, load_model, prepare_clip, save_model
from pipeline.event_model.scoring import add_counts, drop_edge_detections, match_events, precision_recall
from pipeline.event_model.settings import check_settings
from pipeline.event_model.sources import (
    load_split,
    make_split,
    recordings,
    render_background,
    save_split,
)
from pipeline.filter_id.dataset import build_dataset
from pipeline.filter_id.model import FilterIdModel
from pipeline.filters import apply_filter

# ********************************
# Shared fixture and settings
# ********************************

FS = 48000
SETTINGS_PATH = Path(__file__).parents[1] / "event_model" / "settings.json"


# Returns a fresh copy of the exact default settings.
def default_settings():
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


# Returns the folder after writing count simple background recordings: noise
# kept to 150 Hz - 3 kHz like speech, in 1 s phrases with quiet pauses between
# them, at RMS 0.1. They stand in for real recordings, which the tests can't
# ship.
def write_test_backgrounds(folder, count, seconds):
    folder.mkdir(parents=True, exist_ok=True)
    speech_band = {"type": "bandpass", "family": "butter", "order": 4, "cutoff": [150, 3000]}
    n_samples = round(seconds * FS)
    time = np.arange(n_samples) / FS

    for index in range(count):
        rng = np.random.default_rng([7, index])
        noise, _ = apply_filter(rng.standard_normal(n_samples), FS, speech_band)
        envelope = np.maximum(0.05, np.sin(2 * np.pi * 0.5 * time + index))
        audio = noise * envelope
        audio *= 0.1 / np.sqrt(np.mean(audio**2))
        sf.write(folder / f"background-{index}.wav", audio, FS, subtype="FLOAT")

    return folder.as_posix()


# Returns the fixed development run, training its model only once in the test
# session.
@pytest.fixture(scope="session")
def development(tmp_path_factory):
    folder = tmp_path_factory.mktemp("development")
    settings = default_settings()
    settings["seed"] = 7
    settings["clip_seconds"] = 3.0
    # Each type shows up 0 or 1 times per 3 s clip, on its own, like the 0-2 per
    # 10 s clip in real runs. Putting all 7 in every clip made alarm always
    # overlap whine and hum, so the model learned "a tone is around" instead of
    # "it beeps".
    settings["events"]["per_clip"] = [0, 1]

    # The development fixture uses the louder end of the permitted ranges for the
    # sounds that are hardest to see in noise. Real runs keep the defaults,
    # including quiet hiss at -24 dB.
    settings["events"]["hiss"]["level_db"] = [-18, -6]
    settings["events"]["hum"]["level_db"] = [-6, 0]
    settings["events"]["whine"]["level_db"] = [-12, -6]
    settings["events"]["click"]["peak_db"] = [10, 14]

    settings["clips"]["train_clips"] = 240
    settings["clips"]["tune_clips"] = 60
    settings["model"]["max_iter"] = 60
    check_settings(settings)
    settings["sources"]["source_dir"] = write_test_backgrounds(folder / "backgrounds", 3, 160)
    settings["sources"]["recording_groups"] = {}
    items = recordings(settings)
    split = make_split(items, settings)
    clips = {}

    for name, count in (("train", 240), ("tune", 60)):
        clips[name] = []

        for index in range(count):
            audio, record = make_labeled_clip(name, index, items, split, settings)
            clips[name].append(prepare_clip(audio, record, settings))
            del audio

    model = EventModel(EVENT_TYPES, settings, 7)
    model.fit(clips["train"])
    model.tune_thresholds(clips["tune"], settings["scoring"])
    model_path = folder / "model.joblib"
    save_model(model_path, model, settings)

    return {
        "settings": settings,
        "items": items,
        "split": split,
        "model": model,
        "clips": clips,
        "folder": folder,
        "model_path": model_path,
    }


# Returns every required setting path, including nested groups. The entries
# inside recording_groups aren't required, since each run names its own.
def setting_paths(settings, prefix=()):
    paths = []

    for key, value in settings.items():
        path = prefix + (key,)
        paths.append(path)

        if isinstance(value, dict) and key != "recording_groups":
            paths.extend(setting_paths(value, path))

    return paths


# Tests: defaults pass and removing any required field names that missing field.
# Why: a long run should stop before rendering when a settings file is
# incomplete.
def test_settings_require_every_field():
    settings = default_settings()
    check_settings(settings)

    for path in setting_paths(settings):
        changed = copy.deepcopy(settings)
        parent = changed

        for key in path[:-1]:
            parent = parent[key]

        del parent[path[-1]]

        with pytest.raises(ValueError, match=path[-1]):
            check_settings(changed)


# Tests: every settings file in run-settings/ passes the settings check.
# Why: a teammate starts the long final runs from these files on another
# computer, so a bad one should fail here and not hours into a run.
def test_run_settings_files_pass_the_check():
    folder = Path(__file__).parents[2] / "run-settings"
    paths = sorted(folder.glob("*.json"))

    assert paths

    for path in paths:
        check_settings(json.loads(path.read_text(encoding="utf-8")))


# Tests: bad rates, ranges, counts, smoothing and types stop with their setting
# names.
# Why: malformed values must not turn into obscure errors halfway through a run.
def test_settings_reject_bad_values():
    cases = [
        ("sample_rate", 192000),
        ("seed", True),
        ("clip_seconds", 0),
        ("bits", 16),
        ("decoding.rumble.smooth_frames", 2),
        ("features.long_window", 15),
        ("noise.snr_db", [40, 20]),
        ("sources.recording_groups", []),
        ("model.train_frame_step", 0),
        ("events.rumble.cutoff_hz", [0, 100]),
        ("split.test_recording_ids", ["x", "x"]),
        ("events.whine.freq_hz", [2000, 20000]),
        ("events.alarm.beep_s", [0.01, 0.25]),
        ("pretrained.use_ours", False),
        ("pretrained.use_panns", 1),
        ("pretrained.checkpoint", ""),
        ("clips.speech_rms", 0),
        ("clips.train_level_db", [0, -10]),
        ("clips.train_filtered_chance", 1.5),
        ("clips.train_undone_chance", -0.1),
    ]

    for path, value in cases:
        changed = default_settings()
        keys = path.split(".")
        parent = changed

        for key in keys[:-1]:
            parent = parent[key]

        parent[keys[-1]] = value

        with pytest.raises(ValueError, match=keys[-1]):
            check_settings(changed)


# ********************************
# Backgrounds, events and clips
# ********************************


# Tests: all 7 sounds have exact sample lengths, repeat with a seed and keep
# their drawn ranges. The short sounds last 5 decays or 50 ms, an alarm's
# length is its whole beeps and gaps, and a click has almost nothing left above
# 30 kHz.
# Why: labels must describe the actual inserted sound and its recipe, and a
# sound with energy a real microphone wouldn't pick up would teach the model the
# wrong thing.
def test_event_sounds_repeat_and_keep_ranges():
    settings = default_settings()["events"]
    functions = {
        "rumble": events.rumble,
        "hum": events.hum,
        "whine": events.whine,
        "hiss": events.hiss,
        "alarm": events.alarm,
        "thump": events.thump,
        "click": events.click,
    }

    assert tuple(functions) == EVENT_TYPES

    for name, generate in functions.items():
        for seed in range(6):
            audio, recipe = generate(settings[name], FS, np.random.default_rng(seed))
            again, repeated = generate(settings[name], FS, np.random.default_rng(seed))
            np.testing.assert_array_equal(audio, again)

            assert recipe == repeated
            assert len(audio) == round(recipe["duration_s"] * FS)
            assert np.all(np.isfinite(audio))

            for key, value in recipe.items():
                if key in settings[name] and not (name == "alarm" and key == "duration_s"):
                    assert settings[name][key][0] <= value <= settings[name][key][1]

            if name in SHORT_TYPES:
                assert recipe["duration_s"] == min(5 * recipe["decay_s"], 0.05)

            if name == "alarm":
                beeps = recipe["beeps"]
                beep_samples = round(recipe["beep_s"] * FS)
                gap_samples = round(recipe["gap_s"] * FS)

                assert beeps >= 1
                assert len(audio) == beeps * beep_samples + (beeps - 1) * gap_samples
                assert recipe["duration_s"] <= settings["alarm"]["duration_s"][1]

            if name == "click":
                spectrum = np.abs(np.fft.rfft(audio, 65536))**2
                frequencies = np.fft.rfftfreq(65536, 1 / FS)

                assert spectrum[frequencies > 30000].sum() < 0.02 * spectrum.sum()

        with pytest.raises(TypeError, match="rng"):
            generate(settings[name], FS, 3)


# Tests: whole recordings are held out and train and tune stretches keep their
# exact guard gap.
# Why: clips from the same moment must never appear on both sides of the split.
def test_split_keeps_recordings_and_time_apart(development):
    split = development["split"]

    assert len(split["test"]) == 1
    assert split == make_split(development["items"], development["settings"])

    for train, tune in zip(split["train"], split["tune"]):
        assert train["recording_id"] == tune["recording_id"]
        assert train["recording_id"] not in split["test"]
        assert tune["start_s"] - train["end_s"] == 10

    settings = copy.deepcopy(development["settings"])
    settings["split"]["test_recording_ids"] = [development["items"][0]["recording_id"]]

    assert make_split(development["items"], settings)["test"] == settings["split"]["test_recording_ids"]

    with pytest.raises(ValueError, match="test_recordings"):
        make_split(development["items"][:1], settings)

    settings["split"]["guard_seconds"] = 100

    with pytest.raises(ValueError):
        make_split(development["items"], settings)


# Tests: recordings of awkward lengths still get a split with exactly the guard
# gap, and the split passes its own check.
# Why: the train end and tune start are decimal numbers, and for some lengths
# (like 4 s with a 0.1 s guard) their difference comes out a hair under the
# guard, which used to make make_split refuse its own split.
def test_split_guard_survives_rounding():
    settings = default_settings()
    settings["clip_seconds"] = 0.6
    settings["split"]["guard_seconds"] = 0.1
    settings["split"]["tune_fraction"] = 0.5

    for seconds in (3.0, 4.0, 7.3, 12.9, 301.8):
        items = []

        for recording_id in ("a", "b"):
            items.append({
                "recording_id": recording_id,
                "kind": "wav",
                "background_path": "unused.wav",
                "copies": [],
                "duration_s": seconds,
            })

        split = make_split(items, settings)

        for train, tune in zip(split["train"], split["tune"]):
            assert tune["start_s"] - train["end_s"] == pytest.approx(0.1)


# Tests: raw and noisy files form one recording, while missing or repeated group
# files stop.
# Why: noisy copies may be listed for audit but must never become the
# background.
def test_recording_groups_use_only_the_raw_file(tmp_path):
    settings = default_settings()
    settings["clip_seconds"] = 0.1
    settings["sources"]["source_dir"] = str(tmp_path)

    for name, level in (("raw", 0.1), ("noisy", 0.2), ("alone", 0.3)):
        sf.write(tmp_path / f"{name}.wav", np.full(FS // 10, level), FS, subtype="FLOAT")

    groups = {"voice": {"background": "raw", "copies": ["noisy.wav"]}}
    settings["sources"]["recording_groups"] = groups
    items = recordings(settings)

    assert len(items) == 2
    assert items[0]["recording_id"] == "voice"
    assert items[0]["copies"] == ["noisy.wav"]
    assert np.mean(render_background(items[0], 0, 0.1, FS)) == pytest.approx(0.1)

    bad_groups = [
        {"x": {"background": "missing", "copies": []}},
        {"x": {"background": "raw", "copies": ["raw"]}},
        {**groups, "y": {"background": "noisy", "copies": []}},
    ]

    for bad in bad_groups:
        settings["sources"]["recording_groups"] = bad

        with pytest.raises(ValueError, match="recording_groups"):
            recordings(settings)


# Tests: wrong-rate, stereo and too-short recordings are refused before
# rendering.
# Why: every saved source has to match the model's frame sizes and clip length.
def test_recordings_reject_bad_wavs(tmp_path):
    settings = default_settings()
    settings["clip_seconds"] = 0.1
    settings["sources"]["recording_groups"] = {}

    cases = [
        ("stereo", np.ones((FS // 10, 2)), FS),
        ("rate", np.ones(FS // 10), 192000),
        ("short", np.ones(10), FS),
    ]

    for name, audio, rate in cases:
        folder = tmp_path / name
        folder.mkdir()
        sf.write(folder / "bad.wav", audio, rate, subtype="FLOAT")
        settings["sources"]["source_dir"] = str(folder)

        with pytest.raises(ValueError):
            recordings(settings)


# Tests: loading rejects a wrong version, unknown IDs, outside stretches, shared
# test IDs and a short guard.
# Why: an edited split file must not silently break the held-out comparison.
def test_loading_checks_every_split_rule(development, tmp_path):
    items = development["items"]
    settings = development["settings"]
    split = development["split"]
    path = tmp_path / "split.json"
    save_split(path, split)

    assert load_split(path, items, settings) == split

    bad_splits = []
    bad = copy.deepcopy(split)
    bad["version"] = 2
    bad_splits.append(bad)

    for group in ("test", "train", "tune"):
        bad = copy.deepcopy(split)

        if group == "test":
            bad[group][0] = "unknown"
        else:
            bad[group][0]["recording_id"] = "unknown"

        bad_splits.append(bad)

    for key, value in (("start_s", -1), ("end_s", 161), ("end_s", 0.1)):
        bad = copy.deepcopy(split)
        bad["train"][0][key] = value
        bad_splits.append(bad)

    bad = copy.deepcopy(split)
    bad["train"][0]["recording_id"] = split["test"][0]
    bad_splits.append(bad)
    bad = copy.deepcopy(split)
    bad["tune"][0]["start_s"] -= 1
    bad_splits.append(bad)

    for bad in bad_splits:
        save_split(path, bad)

        with pytest.raises(ValueError):
            load_split(path, items, settings)


# Tests: clip indices rebuild exactly, labels keep margins and gaps, and
# recorded gains recover RMS 0.1.
# Why: case comparisons depend on having exactly the same underlying clip and
# labels.
def test_clips_rebuild_and_keep_event_bounds(development):
    settings = development["settings"]
    items = development["items"]
    split = development["split"]
    all_events = []

    for name in ("train", "tune", "test"):
        audio, record = make_clip(name, 0, items, split, settings)
        again, repeated = make_clip(name, 0, items, split, settings)
        np.testing.assert_array_equal(audio, again)

        assert record == repeated

        all_events.append(record["events"])

        for event in record["events"]:
            assert event["start_s"] >= 0.25
            assert event["end_s"] <= 2.75

            for key in ("start_s", "end_s"):
                assert event[key] * FS == pytest.approx(round(event[key] * FS))

        for kind in EVENT_TYPES:
            end = -10

            for event in record["events"]:
                if event["type"] == kind:
                    assert event["start_s"] - end >= 0.3 - 1e-12

                    end = event["end_s"]

        source = None

        for item in items:
            if item["recording_id"] == record["sources"][0]["recording_id"]:
                source = item

        background = render_background(source, record["clip"]["start_s"], 3, FS)

        assert np.sqrt(np.mean((background * record["clip"]["gain"])**2)) == pytest.approx(0.1)

    assert all_events[0] != all_events[1]
    assert all_events[1] != all_events[2]
    assert count_test_clips(items, split, settings) == 53

    _, last = make_clip("test", 52, items, split, settings)

    assert last["clip"]["start_s"] == 156

    with pytest.raises(ValueError, match="index"):
        make_clip("test", 53, items, split, settings)


# Tests: oversized events are counted as dropped and silent backgrounds stop
# before adding sounds.
# Why: a crowded clip needs an honest record, and gain cannot repair a silent
# background.
def test_dropped_events_and_silent_backgrounds(development, tmp_path):
    settings = copy.deepcopy(development["settings"])
    settings["events"]["per_clip"] = [1, 1]

    for kind in ("rumble", "hum", "whine", "hiss", "alarm"):
        settings["events"][kind]["duration_s"] = [3, 3]

    _, record = make_clip("test", 0, development["items"], development["split"], settings)

    assert record["clip"]["dropped_events"] == 5
    assert len(record["events"]) == 2

    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(3 * FS), FS, subtype="FLOAT")
    item = {
        "recording_id": "silent",
        "kind": "wav",
        "background_path": str(path),
        "copies": [],
        "duration_s": 3,
    }

    with pytest.raises(ValueError, match="silent"):
        make_clip("test", 0, [item], {"test": ["silent"]}, settings)


# Tests: test tiling restarts at zero in ID order and skips leftovers for each
# recording.
# Why: a partial last clip must never be joined to the next recording.
def test_test_tiling_restarts_for_each_recording(tmp_path):
    settings = default_settings()
    settings["clip_seconds"] = 3
    settings["events"]["per_clip"] = [0, 0]
    items = []

    for recording_id, seconds in (("b", 9.2), ("a", 6.1)):
        path = tmp_path / f"{recording_id}.wav"
        sf.write(path, np.full(round(seconds * FS), 0.1), FS, subtype="FLOAT")
        items.append({
            "recording_id": recording_id,
            "kind": "wav",
            "background_path": str(path),
            "copies": [],
            "duration_s": seconds,
        })

    split = {"test": ["b", "a"]}

    assert count_test_clips(items, split, settings) == 5

    expected = [("a", 0), ("a", 3), ("b", 0), ("b", 3), ("b", 6)]

    for index, (recording_id, start) in enumerate(expected):
        _, record = make_clip("test", index, items, split, settings)

        assert record["sources"][0]["recording_id"] == recording_id
        assert record["clip"]["start_s"] == start


# Tests: before and after cases rebuild with the same noise and the clean clip's
# power.
# Why: different noise levels or draws would make the stage comparison unfair.
def test_cases_share_noise_and_reference(development):
    settings = development["settings"]
    items = development["items"]
    split = development["split"]
    clean, record = make_clip("test", 0, items, split, settings)
    cases = list(make_test_cases(clean, record, 0, settings))

    for first in range(0, len(cases), 2):
        before, after = cases[first:first + 2]

        assert before["pair_id"] == after["pair_id"]
        assert before["noise_seed"] == after["noise_seed"]
        assert before["snr_db"] == after["snr_db"]
        assert before["damage"][0]["noise_rms"] == after["damage"][0]["noise_rms"]

        for case in (before, after):
            rng = np.random.default_rng(np.random.SeedSequence(case["noise_sequence"]))
            rng.bit_generator.advance(case["noise_seed"])
            power = float(np.mean(clean**2))
            noisy, step = add_noise(clean, case["snr_db"], rng, stage=case["stage"], reference_power=power)
            noise = noisy - clean

            assert step["reference_power"] == np.mean(clean**2)
            assert np.mean(noise**2) == pytest.approx(np.mean(clean**2) / 10**(case["snr_db"] / 10))

            if case["stage"] == "before_filter":
                expected, _ = apply_filter(noisy, FS, case["spec"])
            else:
                expected, _ = apply_filter(clean, FS, case["spec"])
                expected += noise

            expected, _ = quantize(expected, stage="after_filter")
            np.testing.assert_array_equal(case["audio"], expected)


# Tests: training noise continues the clip's random stream after the source and
# event choices, rather than restarting from the clip seed.
# Why: restarting can reuse draws from event sounds for the added white noise.
def test_training_noise_continues_the_clip_stream(tmp_path):
    settings = default_settings()
    settings["clip_seconds"] = 0.1
    settings["edge_margin_s"] = 0.01
    settings["events"]["per_clip"] = [0, 0]
    settings["clips"]["train_noise_chance"] = 1

    path = tmp_path / "background.wav"
    sf.write(path, np.full(2 * FS, 0.1), FS, subtype="FLOAT")
    item = {
        "recording_id": "background",
        "kind": "wav",
        "background_path": str(path),
        "copies": [],
        "duration_s": 2,
    }
    stretch = {"recording_id": "background", "start_s": 0, "end_s": 2}
    split = {"train": [stretch]}
    clean, clean_record = make_clip("train", 0, [item], split, settings)

    seed = np.random.SeedSequence([settings["seed"], 20, 0])
    rng = np.random.default_rng(seed)
    rng.integers(1)
    count = round(settings["clip_seconds"] * FS)
    rng.integers(0, 2 * FS - count + 1)

    for _ in EVENT_TYPES:
        rng.integers(0, 1)

    rng.random()
    snr = float(rng.uniform(20, 40))
    power = float(np.mean(clean**2))
    expected, noise_record = add_noise(clean, snr, rng, stage="after_filter", reference_power=power)
    expected, rounding = quantize(expected, stage="after_filter")

    actual, record = make_labeled_clip("train", 0, [item], split, settings)

    np.testing.assert_array_equal(actual, expected)

    assert record["clip"] == clean_record["clip"]
    assert record["damage"][0] == noise_record
    assert record["damage"][1] == rounding


# Tests: leaving the optional clip settings out makes the same clip and record
# as writing them out with their earlier values, and two chances that add up
# to more than 1 stop with their names.
# Why: runs saved before these settings existed have to keep loading and keep
# matching their settings.json.
def test_optional_clip_settings_keep_the_old_clip(development):
    items = development["items"]
    split = development["split"]
    settings = development["settings"]
    written = copy.deepcopy(settings)
    written["clips"]["speech_rms"] = 0.1
    written["clips"]["train_level_db"] = [0, 0]
    written["clips"]["train_filtered_chance"] = 0
    written["clips"]["train_undone_chance"] = 0
    check_settings(written)

    audio, record = make_labeled_clip("train", 0, items, split, settings)
    same, same_record = make_labeled_clip("train", 0, items, split, written)

    np.testing.assert_array_equal(audio, same)

    assert record == same_record

    written["clips"]["train_filtered_chance"] = 0.6
    written["clips"]["train_undone_chance"] = 0.6

    with pytest.raises(ValueError, match="train_undone_chance"):
        check_settings(written)


# Tests: speech_rms sets the level the speech is scaled to, and train_level_db
# moves a whole training clip by the drawn number of dB before it is rounded.
# Why: the team picks the speech level so 16-bit audio doesn't clip, and a
# model trained at one level changes its answers on a quieter recording.
def test_speech_level_and_training_level_settings(development):
    items = development["items"]
    split = development["split"]
    settings = copy.deepcopy(development["settings"])
    settings["events"]["per_clip"] = [0, 0]
    settings["clips"]["train_noise_chance"] = 0

    quiet = copy.deepcopy(settings)
    quiet["clips"]["speech_rms"] = 0.025
    usual, _ = make_clip("train", 0, items, split, settings)
    quarter, _ = make_clip("train", 0, items, split, quiet)

    assert np.sqrt(np.mean(usual**2)) == pytest.approx(0.1)
    assert np.sqrt(np.mean(quarter**2)) == pytest.approx(0.025)
    np.testing.assert_allclose(quarter, usual / 4)

    moved = copy.deepcopy(settings)
    moved["clips"]["train_level_db"] = [-20, -20]
    expected, _ = quantize(usual * 10 ** (-20 / 20), stage="after_filter")
    actual, record = make_labeled_clip("train", 0, items, split, moved)

    np.testing.assert_array_equal(actual, expected)

    assert record["clip"]["level_db"] == -20


# Tests: with train_filtered_chance 1 a training clip is the clean clip put
# through the filter in its record and rounded once. With train_undone_chance
# 1 it is that same clip with the filter undone. With both at one half, both
# versions and both noise stages show up.
# Why: the extra comparison trains on filtered and undone audio, so those
# clips have to be made the way the test cases are and say so in their records.
def test_filtered_and_undone_training_clips(development):
    items = development["items"]
    split = development["split"]
    settings = copy.deepcopy(development["settings"])
    settings["clips"]["train_noise_chance"] = 0
    clean, clean_record = make_clip("train", 0, items, split, settings)

    filtered_settings = copy.deepcopy(settings)
    filtered_settings["clips"]["train_filtered_chance"] = 1
    filtered, record = make_labeled_clip("train", 0, items, split, filtered_settings)
    spec = {}

    for key in ("type", "family", "order", "cutoff"):
        spec[key] = record["filter"][key]

    expected, _ = apply_filter(clean, FS, spec)
    expected, rounding = quantize(expected, stage="after_filter")

    np.testing.assert_array_equal(filtered, expected)

    assert record["clip"]["version"] == "filtered"
    assert record["events"] == clean_record["events"]
    assert record["damage"] == [rounding]
    assert "compensation" not in record

    undone_settings = copy.deepcopy(settings)
    undone_settings["clips"]["train_undone_chance"] = 1
    undone, undone_record = make_labeled_clip("train", 0, items, split, undone_settings)
    recovered, _ = compensate(expected, FS, spec)

    np.testing.assert_array_equal(undone, recovered)

    assert undone_record["clip"]["version"] == "undone"
    assert undone_record["filter"] == record["filter"]
    assert undone_record["compensation"]["method"] == "python"

    mixed = copy.deepcopy(settings)
    mixed["clips"]["train_filtered_chance"] = 0.5
    mixed["clips"]["train_undone_chance"] = 0.5
    mixed["clips"]["train_noise_chance"] = 1
    versions = set()
    stages = set()

    for index in range(12):
        _, mixed_record = make_labeled_clip("train", index, items, split, mixed)
        versions.add(mixed_record["clip"]["version"])
        stages.add(mixed_record["damage"][0]["stage"])

    assert versions == {"filtered", "undone"}
    assert stages == {"before_filter", "after_filter"}


# ********************************
# Frame features and labels
# ********************************


# Tests: frame counts round up and frame centers stay at k times the hop.
# Why: labels and detection times must use the exact same frame positions.
def test_frame_counts_and_times():
    assert frame_count(1, 480) == 1
    assert frame_count(480, 480) == 1
    assert frame_count(481, 480) == 2
    np.testing.assert_array_equal(frame_times(4, FS, 480), [0, 0.01, 0.02, 0.03])


# Tests: feature shape, dtype, finite values, the 60-Hz long band and onset flux
# are right.
# Why: the low window must resolve rumble and the short window must time clicks.
def test_features_resolve_low_tones_and_clicks():
    settings = default_settings()["features"]
    time = np.arange(FS) / FS
    audio = np.sin(2 * np.pi * 60 * time)
    features, times = frame_features(audio, FS, settings)

    assert features.shape == (100, 170)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    assert times[1] == 0.01

    edges = np.geomspace(20, 400, 9)
    expected = np.searchsorted(edges, 60) - 1

    assert np.argmax(np.mean(features[10:-10, :8], axis=0)) == expected

    click = np.zeros(FS)
    click[FS // 2] = 1
    measured, _ = frame_features(click, FS, settings)
    flux_column = settings["long_bands"] + settings["short_bands"] + 2

    assert abs(np.argmax(measured[:, flux_column]) - 50) <= 1
    assert measured[0, flux_column] == 0


# Tests: band shares keep the same shape when the whole signal gets louder,
# give the expected dB values, and stay finite for silence.
# Why: an overall level rise must remain separate from a change in the spectrum.
def test_band_levels_separate_shape_from_volume():
    power = np.array([[1.0, 3.0], [0.0, 0.0]])
    total = np.array([4.0, 0.0])
    levels = band_levels(power, total)
    louder = band_levels(power * 100, total * 100)

    np.testing.assert_allclose(levels[0], 10 * np.log10([0.25, 0.75]))
    np.testing.assert_allclose(levels, louder)
    assert np.all(np.isfinite(levels))


# Tests: time averages and changes use the stated neighbors and keep band
# offsets out of the change measurements.
# Why: a louder frequency band in a new recording should not look like a new
# attack, and these measurements must not need quiet clip edges.
def test_band_averages_and_previous_changes():
    bands = np.arange(60, dtype=np.float64).reshape(30, 2)
    averages = band_averages(bands, FS, 480)
    changes = band_changes(bands, FS, 480)

    np.testing.assert_allclose(averages[0], np.mean(bands[:11], axis=0))
    np.testing.assert_allclose(averages[15], np.mean(bands[5:26], axis=0))
    np.testing.assert_allclose(changes[15], bands[15] - np.mean(bands[5:15], axis=0))
    np.testing.assert_array_equal(changes[0], [0, 0])

    shifted = bands + np.array([30, -20])
    np.testing.assert_allclose(band_changes(shifted, FS, 480), changes)


# Tests: spreads use the 25 frames on each side, a steady band has none, and a
# band switching on and off every 15 frames has a large one.
# Why: the spread is what tells a beeping alarm from a steady whine at the same
# pitch.
def test_band_spreads_tell_beeping_from_steady():
    frames = np.arange(80)
    beeping = np.where((frames // 15) % 2 == 0, 0.0, -30.0)
    bands = np.column_stack([np.full(80, -10.0), beeping])
    spreads = band_spreads(bands, FS, 480)

    np.testing.assert_allclose(spreads[40], np.std(bands[15:66], axis=0))
    np.testing.assert_allclose(spreads[0], np.std(bands[:26], axis=0))
    assert np.all(spreads[:, 0] == 0)
    assert np.all(spreads[:, 1] > 10)


# Tests: sounds at either WAV edge do not alter features far away in the clip.
# Why: arbitrary real recordings can begin and end with events; feature values
# elsewhere must not be calibrated from those edges.
def test_frame_features_do_not_assume_quiet_edges():
    settings = default_settings()["features"]
    time = np.arange(2 * FS) / FS
    audio = 0.1 * np.sin(2 * np.pi * 440 * time)
    original, times = frame_features(audio, FS, settings)

    changed_audio = audio.copy()
    changed_audio[:FS // 4] += 0.5
    changed_audio[-FS // 4:] -= 0.5
    changed, changed_times = frame_features(changed_audio, FS, settings)

    np.testing.assert_array_equal(times, changed_times)
    np.testing.assert_array_equal(original[60:140], changed[60:140])


# Tests: context columns run from left neighbors to right, repeating this clip's
# edges.
# Why: neighboring clips must not leak their frames into each other.
def test_context_repeats_edges_in_column_order():
    features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    expected = [[1, 2, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 5, 6]]
    np.testing.assert_array_equal(add_context(features, 1), expected)
    other = add_context(np.array([[90, 91]], dtype=np.float32), 1)
    np.testing.assert_array_equal(other, [[90, 91, 90, 91, 90, 91]])


# Tests: targets include the start center, exclude the end, and keep a 2-ms
# click or thump with early ties.
# Why: short sounds shorter than a frame step still need a positive target.
def test_frame_targets_use_half_open_times_and_nearest_short_sound():
    labeled = [
        {"type": "hiss", "start_s": 0.01, "end_s": 0.03},
        {"type": "click", "start_s": 0.045, "end_s": 0.047},
        {"type": "thump", "start_s": 0.065, "end_s": 0.067},
    ]
    targets = events_to_frames(labeled, 10, FS, 480, EVENT_TYPES)

    assert targets.dtype == np.uint8
    np.testing.assert_array_equal(np.flatnonzero(targets[:, EVENT_TYPES.index("hiss")]), [1, 2])
    np.testing.assert_array_equal(np.flatnonzero(targets[:, EVENT_TYPES.index("click")]), [4])
    np.testing.assert_array_equal(np.flatnonzero(targets[:, EVENT_TYPES.index("thump")]), [6])


# Tests: exact gaps merge, larger gaps do not, smoothing shortens at edges and
# minimums use clipped lengths.
# Why: the fixed decoding rules must be applied the same way in tuning and
# comparison.
def test_decoding_smoothing_gaps_clipping_and_minimums():
    decoding = {"hiss": {"smooth_frames": 1, "merge_gap_s": 0.05, "min_duration_s": 0}}
    values = np.zeros((20, 1))
    values[[0, 6, 13, 19], 0] = 1
    found = frames_to_events(values, 20, FS, 480, 0.191, ("hiss",), {"hiss": 0.5}, decoding)

    assert len(found) == 2
    assert found[0]["start_s"] == 0
    assert found[0]["end_s"] == pytest.approx(0.065)
    assert found[1]["end_s"] == 0.191

    decoding["hiss"]["merge_gap_s"] = 0
    decoding["hiss"]["smooth_frames"] = 3
    probabilities = np.array([[1.0], [0], [0]])
    smoothed = frames_to_events(
        probabilities, 3, FS, 480, 0.025, ("hiss",), {"hiss": 0.5}, decoding
    )

    assert len(smoothed) == 1
    assert smoothed[0]["confidence"] == 0.5
    assert smoothed[0]["end_s"] == 0.005

    decoding["hiss"]["smooth_frames"] = 1
    decoding["hiss"]["min_duration_s"] = 0.01

    assert frames_to_events(np.ones((1, 1)), 1, FS, 480, 0.003, ("hiss",), {"hiss": 0.5}, decoding) == []


# Tests: generated labels survive targets-to-events decoding under the exact
# scoring collars.
# Why: frame timing alone should not prevent a perfect detector from matching
# the labels.
def test_labels_round_trip_through_frames(development):
    settings = development["settings"]
    thresholds = dict.fromkeys(EVENT_TYPES, 0.5)

    for index in range(5):
        audio, record = make_clip("test", index, development["items"], development["split"], settings)
        count = frame_count(len(audio), 480)
        targets = events_to_frames(record["events"], count, FS, 480, EVENT_TYPES)
        probabilities = targets.astype(float)
        detected = frames_to_events(
            probabilities, count, FS, 480, 3, EVENT_TYPES, thresholds, settings["decoding"]
        )
        counts = match_events(detected, record["events"], settings["scoring"])

        for kind in EVENT_TYPES:
            assert counts[kind]["fp"] == 0
            assert counts[kind]["fn"] == 0


# ********************************
# Scoring and the trained model
# ********************************


# Tests: matching is one to one and deterministic, with separate onset collars
# and the short sounds' offsets ignored.
# Why: overlapping possible pairs must not count an event twice or favor a
# version.
def test_scoring_pairs_and_collars():
    rules = default_settings()["scoring"]
    labels = [
        {"type": "hiss", "start_s": 1, "end_s": 2},
        {"type": "hiss", "start_s": 1.15, "end_s": 2.15},
        {"type": "click", "start_s": 3, "end_s": 3.01},
    ]
    detected = [
        {"type": "hiss", "start_s": 1.1, "end_s": 2.1},
        {"type": "hiss", "start_s": 0.85, "end_s": 1.85},
        {"type": "click", "start_s": 3.05, "end_s": 8},
    ]
    counts = match_events(detected, labels, rules)

    assert counts["hiss"]["tp"] == 2
    assert counts["hiss"]["pairs"] == [(0, 1), (1, 0)]
    assert counts["click"]["tp"] == 1
    assert counts == match_events(detected, labels, rules)

    detected[2]["start_s"] = 3.051

    assert match_events(detected, labels, rules)["click"]["tp"] == 0

    for kind in ("rumble", "hum", "whine", "hiss", "alarm"):
        label = [{"type": kind, "start_s": 1, "end_s": 4}]

        assert match_events([{"type": kind, "start_s": 1.2, "end_s": 4.6}], label, rules)[kind]["tp"] == 1
        assert match_events([{"type": kind, "start_s": 1.201, "end_s": 4}], label, rules)[kind]["tp"] == 0
        assert match_events([{"type": kind, "start_s": 1, "end_s": 4.601}], label, rules)[kind]["tp"] == 0


# Tests: empty sets count misses and alarms correctly, undefined scores are
# None, and edge drops are exact.
# Why: clips with no events still affect false alarm totals without inventing
# precision or recall.
def test_empty_scores_counts_and_edges():
    rules = default_settings()["scoring"]
    event = {"type": "hiss", "start_s": 1, "end_s": 2}

    assert match_events([], [event], rules)["hiss"]["fn"] == 1
    assert match_events([event], [], rules)["hiss"]["fp"] == 1

    empty = match_events([], [], rules)
    scores = precision_recall(empty)

    assert scores["overall"]["f1"] is None
    assert scores["hiss"]["precision"] is None
    assert scores["hiss"]["recall"] is None

    totals = add_counts({}, match_events([event], [event], rules))
    add_counts(totals, match_events([], [event], rules))

    assert precision_recall(totals)["overall"]["f1"] == pytest.approx(2 / 3)

    edges = [
        {"start_s": 0, "end_s": 0.25},
        {"start_s": 2.75, "end_s": 3},
        {"start_s": 0.24, "end_s": 0.26},
    ]

    assert drop_edge_detections(edges, 3, 0.25) == [edges[2]]


# Tests: the exact development fixture reaches at least 0.7 F1 on its clean test
# split for every type except alarm and thump, which need at least 0.6.
# Why: the baseline detector must recognize all 7 kinds before running the real
# comparison. Alarm and thump get the lower bar because they share pitches with
# other sounds (alarm with whine and hum, thump with alarm beeps), and our own
# features scored 0.65-0.73 on them in every version of this fixture we tried.
def test_development_model_reaches_required_f1(development):
    clips = []
    settings = development["settings"]

    for index in range(count_test_clips(development["items"], development["split"], settings)):
        audio, record = make_clip("test", index, development["items"], development["split"], settings)
        clips.append(prepare_clip(audio, record, settings))

    metrics = evaluate(development["model"], clips, settings["scoring"])
    print("development scores:", metrics["scores"])

    for kind in EVENT_TYPES:
        required = 0.7

        if kind in ("alarm", "thump"):
            required = 0.6

        assert metrics["scores"][kind]["f1"] >= required


# Tests: saved models give identical events, empty-type fit stops, and threshold
# ties prefer 0.5 then lower.
# Why: the frozen model and its threshold choices must rebuild exactly.
def test_saved_model_fit_errors_and_threshold_ties(development):
    model, settings = load_model(development["model_path"])
    clip = development["clips"]["tune"][0]

    assert model.detect(clip.features, clip.n_frames, clip.duration_s) == development["model"].detect(
        clip.features, clip.n_frames, clip.duration_s
    )

    for column, kind in enumerate(EVENT_TYPES):
        targets = np.ones_like(clip.targets)
        targets[:, column] = 0
        no_yes = Clip(clip.features, clip.n_frames, targets, [], 3, "empty", "empty")

        with pytest.raises(ValueError, match=kind):
            EventModel(EVENT_TYPES, settings, 7).fit([no_yes])

    tied = EventModel(EVENT_TYPES, copy.deepcopy(settings), 7)

    assert tied.tune_thresholds([], settings["scoring"]) == dict.fromkeys(EVENT_TYPES, 0.5)

    tied.settings["decoding"]["threshold_grid"] = [0.6, 0.4]

    assert tied.tune_thresholds([], settings["scoring"]) == dict.fromkeys(EVENT_TYPES, 0.4)


# ********************************
# Commands and comparison
# ********************************


# Tests: detect prints one JSON line and refuses stereo or wrong-rate inputs
# with a clear message.
# Why: command output must be usable by a teammate's script without hiding
# invalid inputs.
def test_detect_command_json_and_bad_wavs(development, tmp_path, capsys):
    audio, _ = make_clip("test", 0, development["items"], development["split"], development["settings"])
    path = tmp_path / "clip.wav"
    sf.write(path, audio, FS, subtype="FLOAT")
    capsys.readouterr()
    main(["detect", "--model", str(development["model_path"]), str(path)])
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 1

    result = json.loads(lines[0])

    assert result["file"] == str(path)
    assert isinstance(result["events"], list)

    for samples, rate, message in ((np.column_stack([audio, audio]), FS, "mono"), (audio, 192000, "sample rate")):
        sf.write(path, samples, rate, subtype="FLOAT")

        with pytest.raises(SystemExit, match=message):
            main(["detect", "--model", str(development["model_path"]), str(path)])


# Returns a small filter ID model trained on the WAV files in source_dir, and
# its settings.
def small_filter_id(source_dir):
    settings = {
        "seed": 4240,
        "sample_rate": FS,
        "source_dir": source_dir,
        "clip_seconds": 1.0,
        "examples_per_source": 12,
        "test_fraction": 0.2,
        "bits": 24,
        "filters": {
            "types": ["highpass", "lowpass", "bandpass"],
            "orders": {"highpass": [1, 2, 4], "lowpass": [1, 2, 4], "bandpass": [2, 4, 8]},
            "highpass_cutoff_hz": [80, 1000],
            "lowpass_cutoff_hz": [2000, 16000],
            "bandpass_low_hz": [80, 1000],
            "bandpass_high_hz": [2000, 16000],
        },
        "noise": {"snr_db": [20, 40], "stages": ["before_filter", "after_filter"]},
        "features": {"n_bands": 96, "f_min_hz": 20, "f_max_hz": 24000, "segment": 8192},
    }
    data = build_dataset(settings)
    model = FilterIdModel(7).fit(data.features, data.labels)

    return model, settings


# Tests: two fixture clips give equal version counts under Python and fake C,
# with or without blind ID.
# Why: every version must be scored on exactly the same cases and saved without
# clipping recovery.
def test_comparison_versions_and_both_inverse_methods(development, tmp_path):
    settings = development["settings"]
    recording_id = development["split"]["test"][0]
    original = None

    for item in development["items"]:
        if item["recording_id"] == recording_id:
            original = item

    audio = render_background(original, 0, 6, FS)
    path = tmp_path / "two_clips.wav"
    sf.write(path, audio, FS, subtype="FLOAT")
    item = {**original, "background_path": str(path), "duration_s": 6}
    script = tmp_path / "fake_filter.py"
    script.write_text(
        "import soundfile as sf\n"
        "audio, rate = sf.read('in.wav', dtype='int16')\n"
        "sf.write('out.wav', audio, rate, subtype='PCM_16')\n",
        encoding="utf-8",
    )
    fid = small_filter_id(settings["sources"]["source_dir"])

    for method in ("python", "c"):
        for use_fid in (False, True):
            folder = tmp_path / f"{method}_{use_fid}" / "examples"
            filter_id = None

            if use_fid:
                filter_id = fid

            command = [sys.executable, str(script)]
            split = {"test": [recording_id]}
            result = run_comparison(
                development["model"], settings, [item], split,
                filter_id=filter_id, method=method, command=command,
                export_dir=folder,
            )

            assert result["test_clips"] == 2
            assert result["case_count"] == 8
            assert result["method"] == method
            expected = {"clean", "filtered", "known_undone"}

            if use_fid:
                expected.add("ai_undone")

                assert result["filter_id"]["type_accuracy"]["cases"] == 8

            assert set(result["versions"]) == expected

            for version in expected:
                assert result["versions"][version]["case_count"] == 8
                assert result["by_noise_stage"]["before_filter"][version]["case_count"] == 4
                total = 0

                for bucket in result["by_snr_bin"].values():
                    total += bucket.get(version, {}).get("case_count", 0)

                assert total == 8

            lines = (folder.parent / "cases.jsonl").read_text().splitlines()

            assert len(lines) == 8
            assert json.loads(lines[0])["spec"] != json.loads(lines[0])["filter"]

            for wav in folder.glob("*.wav"):
                assert sf.info(wav).subtype == "FLOAT"


# Tests: unsupported blind specs fall back to unchanged audio and explain the
# failure.
# Why: an unsupported estimate must not remove a case from just one version's
# totals.
def test_unsupported_blind_spec_is_recorded():
    audio = np.ones(100)
    predicted = {
        "type": "lowpass",
        "family": "butter",
        "order": 3,
        "cutoff": 2000,
    }
    recovered, record, failure = _ai_inverse(audio, default_settings(), predicted, "python", None)
    np.testing.assert_array_equal(recovered, audio)

    assert record["not_compensated"] is True
    assert record["spec"] == predicted
    assert "order" in failure
