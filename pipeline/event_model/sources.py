"""
Read recordings and keep the three splits apart.

This file has two groups:
    * Reading recordings
    * Saving and checking the split
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

# ********************************
# Reading recordings
# ********************************


# Returns checked WAV paths by stem. It stops with a ValueError if the folder
# has no WAVs, or a file is short, stereo, or at the wrong rate.
def _wav_paths(settings) -> dict:
    folder = settings["sources"]["source_dir"]

    if not folder:
        raise ValueError("sources.source_dir must name a folder of WAV recordings")

    paths = {}

    for path in sorted(Path(folder).glob("*.wav")):
        info = sf.info(path)

        if info.channels != 1:
            raise ValueError(f"{path.name} must be mono")

        if info.samplerate != settings["sample_rate"]:
            raise ValueError(f"{path.name} sample_rate must be {settings['sample_rate']}")

        if info.frames < round(settings["clip_seconds"] * settings["sample_rate"]):
            raise ValueError(f"{path.name} is shorter than clip_seconds")

        paths[path.stem] = path.resolve()

    if not paths:
        raise ValueError("sources.source_dir has no WAV files")

    return paths


# Returns one recording per group or ungrouped file, using only each group's raw
# background. It stops with a ValueError if a group names a missing file or a
# file belongs to two groups.
def recordings(settings) -> list[dict]:
    paths = _wav_paths(settings)
    used = set()
    items = []

    for recording_id, group in settings["sources"]["recording_groups"].items():
        if not isinstance(recording_id, str) or not recording_id or not isinstance(group, dict):
            raise ValueError("recording_groups needs a recording ID and group dict")

        background = group.get("background")
        copies = group.get("copies")

        if not isinstance(background, str) or not background or not isinstance(copies, list):
            raise ValueError("recording_groups needs background and copies")

        for name in [background] + copies:
            if not isinstance(name, str) or not name:
                raise ValueError("recording_groups file names must be strings")

            stem = _file_stem(name)

            if stem not in paths:
                raise ValueError(f"recording_groups file {name} is missing")

            if stem in used:
                raise ValueError(f"recording_groups file {name} is in two groups")

            used.add(stem)

        background_stem = _file_stem(background)
        items.append(_recording(recording_id, paths[background_stem], copies))

    known_ids = set()

    for item in items:
        known_ids.add(item["recording_id"])

    for name, path in paths.items():
        if name not in used:
            if name in known_ids:
                raise ValueError(f"recording_groups recording_id {name} is also an ungrouped file")

            items.append(_recording(name, path, []))

    return items


# Returns the public details and absolute path for one background file.
def _recording(recording_id, path, copies) -> dict:
    info = sf.info(path)
    kind = "wav"

    if path.stem.startswith("synthetic-bg-"):
        kind = "synthetic"

    return {
        "recording_id": recording_id,
        "kind": kind,
        "background_path": str(path),
        "copies": copies,
        "duration_s": info.frames / info.samplerate,
    }


# Returns a WAV name without its extension, or the name as given if it already
# has no extension.
def _file_stem(name) -> str:
    if name.lower().endswith(".wav"):
        return name[:-4]

    return name


# Returns exactly one background stretch. It stops with a ValueError if its
# times are out of range or the audio is not finite mono at the requested rate.
def render_background(recording, start_s, seconds, sample_rate) -> np.ndarray:
    start = round(start_s * sample_rate)
    count = round(seconds * sample_rate)

    if start < 0 or count < 1:
        raise ValueError("background start_s or seconds is out of range")

    audio, rate = sf.read(recording["background_path"], start=start, frames=count, dtype="float64")

    if rate != sample_rate or audio.ndim != 1 or len(audio) != count or not np.all(np.isfinite(audio)):
        raise ValueError("background stretch must be finite mono audio with the requested rate and length")

    return audio


# ********************************
# Saving and checking the split
# ********************************


# Returns a whole-recording test split and separated train and tune stretches.
# It stops with a ValueError if there are too few recordings or a stretch is
# shorter than one clip after leaving the guard gap.
def make_split(recordings, settings) -> dict:
    rules = settings["split"]

    if len(recordings) < rules["test_recordings"] + 1:
        raise ValueError("split.test_recordings needs at least one more recording")

    ids = []

    for recording in recordings:
        ids.append(recording["recording_id"])

    ids.sort()
    test = list(rules["test_recording_ids"])

    if not test:
        rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], 22, 0]))
        test = sorted(rng.choice(ids, rules["test_recordings"], replace=False).tolist())

    split = {"version": 1, "test": test, "train": [], "tune": []}

    for recording in recordings:
        recording_id = recording["recording_id"]

        if recording_id in test:
            continue

        duration = recording["duration_s"]
        cut = (1 - rules["tune_fraction"]) * duration
        half_guard = rules["guard_seconds"] / 2
        split["train"].append({
            "recording_id": recording_id,
            "start_s": 0,
            "end_s": cut - half_guard,
        })
        split["tune"].append({
            "recording_id": recording_id,
            "start_s": cut + half_guard,
            "end_s": duration,
        })

    _check_split(split, recordings, settings)

    return split


# Returns after checking a loaded split. It stops with a ValueError for unknown
# IDs, invalid stretches, shared test recordings, or a missing guard gap.
def _check_split(split, recordings, settings) -> None:
    if not isinstance(split, dict) or split.get("version") != 1:
        raise ValueError("split version must be 1")

    durations = {}

    for recording in recordings:
        durations[recording["recording_id"]] = recording["duration_s"]

    for name in ("test", "train", "tune"):
        if not isinstance(split.get(name), list) or not split[name]:
            raise ValueError(f"split {name} must be a non-empty list")

    test = split["test"]

    for recording_id in test:
        if not isinstance(recording_id, str) or recording_id not in durations:
            raise ValueError("split test has an unknown recording_id")

    if len(set(test)) != len(test):
        raise ValueError("split test has duplicate recording_id")

    for name in ("train", "tune"):
        for stretch in split[name]:
            if not isinstance(stretch, dict) or not stretch:
                raise ValueError(f"split {name} stretch must be a dict")

            recording_id = stretch.get("recording_id")

            if not isinstance(recording_id, str) or recording_id not in durations:
                raise ValueError(f"split {name} has an unknown recording_id")

            if recording_id in test:
                raise ValueError("split test recording is in train or tune")

            for key in ("start_s", "end_s"):
                value = stretch.get(key)

                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"split {name} {key} must be a finite number")

            start = stretch["start_s"]
            end = stretch["end_s"]

            if not 0 <= start < end <= durations[recording_id]:
                raise ValueError(f"split {name} stretch is outside recording")

            stretch_samples = round((end - start) * settings["sample_rate"])
            clip_samples = round(settings["clip_seconds"] * settings["sample_rate"])

            if stretch_samples < clip_samples:
                raise ValueError(f"split {name} stretch is shorter than clip_seconds")

    for train in split["train"]:
        for tune in split["tune"]:
            if train["recording_id"] == tune["recording_id"]:
                gap = max(tune["start_s"] - train["end_s"], train["start_s"] - tune["end_s"])

                if gap < settings["split"]["guard_seconds"]:
                    raise ValueError("split train and tune must keep guard_seconds apart")


# Returns after saving a split as readable JSON.
def save_split(path, split) -> None:
    Path(path).write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


# Returns a saved split after checking the current recordings and settings. It
# stops with a ValueError if a recording, stretch, or guard gap is invalid.
def load_split(path, recordings, settings) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_split(split, recordings, settings)

    return split
