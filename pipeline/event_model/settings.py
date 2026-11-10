"""
Check the settings for the event model before making audio.

This file has two groups:
    * Small setting checks
    * Checking a run
"""

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from pipeline.event_model.labels import EVENT_TYPES, SHORT_TYPES

# ********************************
# Small setting checks
# ********************************


# Returns after checking required keys and groups. It stops with a ValueError
# naming the full setting path if a key or group is missing.
def _required(settings, defaults, name="settings") -> None:
    is_dictionary = isinstance(settings, dict)

    if not is_dictionary:
        raise ValueError(f"{name} must be a dict")

    for key, default in defaults.items():
        path = f"{name}.{key}"

        if key not in settings:
            raise ValueError(f"missing {path}")

        if isinstance(default, dict):
            _required(settings[key], default, path)


# Returns after checking a number. It stops with a ValueError if the setting is
# not finite or is outside its allowed range.
def _number(value, name, low, high=np.inf, whole=False) -> None:
    kind = Real
    description = "number"

    if whole:
        kind = Integral
        description = "integer"

    if isinstance(value, bool) or not isinstance(value, kind) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite {description}")

    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


# Returns after checking a pair. It stops with a ValueError if the setting does
# not contain two ordered numbers inside its allowed range.
def _range(value, name, low=-np.inf, high=np.inf, whole=False) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a pair")

    for item in value:
        _number(item, name, low, high, whole)

    if value[0] > value[1]:
        raise ValueError(f"{name} must go from low to high")


# ********************************
# Checking a run
# ********************************


# Returns after checking every required setting. It stops with a ValueError
# naming the first missing or invalid setting.
def check_settings(settings) -> None:
    path = Path(__file__).with_name("settings.json")
    defaults = json.loads(path.read_text(encoding="utf-8"))

    # Each run names its own recording groups, so only the groups dict itself is
    # required, not the default's entries.
    defaults["sources"]["recording_groups"] = {}
    _required(settings, defaults)

    _number(settings["seed"], "seed", 0, whole=True)
    _number(settings["sample_rate"], "sample_rate", 48000, 48000, whole=True)
    _number(settings["clip_seconds"], "clip_seconds", 1 / 48000)
    _number(settings["edge_margin_s"], "edge_margin_s", 0)

    if 2 * settings["edge_margin_s"] >= settings["clip_seconds"]:
        raise ValueError("edge_margin_s must leave room inside clip_seconds")

    _number(settings["bits"], "bits", 24, 24, whole=True)
    _check_sources_split(settings)
    _check_features_model(settings)
    _check_events(settings)
    _check_filters_noise(settings)
    _check_pretrained(settings)

    for kind in EVENT_TYPES:
        decoding = settings["decoding"][kind]
        _number(decoding["smooth_frames"], f"decoding.{kind}.smooth_frames", 1, whole=True)

        if decoding["smooth_frames"] % 2 != 1:
            raise ValueError(f"decoding.{kind}.smooth_frames must be odd")

        for key in ("merge_gap_s", "min_duration_s"):
            _number(decoding[key], f"decoding.{kind}.{key}", 0)

        _number(settings["scoring"]["onset_collar_s"][kind], f"scoring.onset_collar_s.{kind}", 0)

    grid = settings["decoding"]["threshold_grid"]

    if not isinstance(grid, list) or not grid:
        raise ValueError("decoding.threshold_grid must be a non-empty list")

    for threshold in grid:
        _number(threshold, "decoding.threshold_grid", 0, 1)

    for key in ("offset_collar_s", "offset_fraction"):
        _number(settings["scoring"][key], f"scoring.{key}", 0)

    offset_types = settings["scoring"]["offset_types"]
    is_list = isinstance(offset_types, list)

    if not is_list:
        raise ValueError("scoring.offset_types must list event types")

    for kind in offset_types:
        if kind not in EVENT_TYPES:
            raise ValueError("scoring.offset_types must list event types")


# Returns after checking sources, splits, and clip counts. It stops with a
# ValueError if any of these settings cannot describe a run.
def _check_sources_split(settings) -> None:
    sources = settings["sources"]

    if not isinstance(sources["source_dir"], str) or not sources["source_dir"]:
        raise ValueError("sources.source_dir must name a folder of WAV recordings")

    split = settings["split"]
    _number(split["test_recordings"], "split.test_recordings", 1, whole=True)
    _number(split["tune_fraction"], "split.tune_fraction", 0, 1)

    if split["tune_fraction"] in (0, 1):
        raise ValueError("split.tune_fraction must be between 0 and 1")

    _number(split["guard_seconds"], "split.guard_seconds", 0)

    ids = split["test_recording_ids"]
    is_list = isinstance(ids, list)

    if not is_list:
        raise ValueError("split.test_recording_ids must be a list")

    for recording_id in ids:
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError("split.test_recording_ids must contain recording IDs")

    if len(set(ids)) != len(ids):
        raise ValueError("split.test_recording_ids must not repeat")

    for key in ("train_clips", "tune_clips", "conditions_per_test_clip", "export_examples"):
        minimum = 1

        if key == "export_examples":
            minimum = 0

        _number(settings["clips"][key], f"clips.{key}", minimum, whole=True)

    _number(settings["clips"]["train_noise_chance"], "clips.train_noise_chance", 0, 1)


# Returns after checking frame sizes, bands, and tree settings. It stops with a
# ValueError if one of these settings is out of range.
def _check_features_model(settings) -> None:
    features = settings["features"]

    for key in ("hop", "short_window", "long_window", "short_bands", "long_bands", "context"):
        minimum = 1

        if key == "context":
            minimum = 0

        _number(features[key], f"features.{key}", minimum, whole=True)

    for size in ("short_window", "long_window"):
        if features[size] < 2 or features[size] % 2:
            raise ValueError(f"features.{size} must be even and at least 2")

    if features["short_window"] > features["long_window"]:
        raise ValueError("features.short_window must not exceed long_window")

    for prefix in ("short", "long"):
        _number(features[f"{prefix}_min_hz"], f"features.{prefix}_min_hz", 1e-9, 24000)
        _number(features[f"{prefix}_max_hz"], f"features.{prefix}_max_hz", 1e-9, 24000)

        if features[f"{prefix}_min_hz"] >= features[f"{prefix}_max_hz"]:
            raise ValueError(f"features.{prefix}_min_hz must be below {prefix}_max_hz")

    model = settings["model"]

    for key in ("max_iter", "max_leaf_nodes", "train_frame_step"):
        minimum = 1

        if key == "max_leaf_nodes":
            minimum = 2

        _number(model[key], f"model.{key}", minimum, whole=True)

    _number(model["learning_rate"], "model.learning_rate", 1e-12, 1)


# Returns after checking the event ranges. It stops with a ValueError if a range
# is invalid or the fades cannot fit inside an event's duration.
def _check_events(settings) -> None:
    events = settings["events"]
    _range(events["per_clip"], "events.per_clip", 0, whole=True)
    _number(events["same_type_gap_s"], "events.same_type_gap_s", 0)

    for kind in EVENT_TYPES:
        for key, value in events[kind].items():
            low = 1 / 48000

            if key.endswith("_db"):
                low = -300

            high = np.inf

            if key.endswith("_hz"):
                high = 23999

            _range(value, f"events.{kind}.{key}", low, high)

        if kind not in SHORT_TYPES and 2 * events[kind]["fade_s"][1] > events[kind]["duration_s"][0]:
            raise ValueError(f"events.{kind}.fade_s must fit inside duration_s")

    if events["rumble"]["cutoff_hz"][0] <= 15:
        raise ValueError("events.rumble.cutoff_hz must be above 15")

    if events["hiss"]["cutoff_hz"][1] >= 20000:
        raise ValueError("events.hiss.cutoff_hz must be below 20000")

    if events["whine"]["freq_hz"][1] >= 20000:
        raise ValueError("events.whine.freq_hz must be below 20000")

    # Each beep gets its own fades, so they have to fit inside the shortest beep.
    if 2 * events["alarm"]["fade_s"][1] > events["alarm"]["beep_s"][0]:
        raise ValueError("events.alarm.fade_s must fit inside beep_s")


# Returns after checking which features the model uses. It stops with a
# ValueError if both are off or the PANNs checkpoint isn't a path.
def _check_pretrained(settings) -> None:
    pretrained = settings["pretrained"]

    for key in ("use_ours", "use_panns"):
        is_switch = isinstance(pretrained[key], bool)

        if not is_switch:
            raise ValueError(f"pretrained.{key} must be true or false")

    if not pretrained["use_ours"] and not pretrained["use_panns"]:
        raise ValueError("pretrained.use_ours and pretrained.use_panns can't both be off")

    if not isinstance(pretrained["checkpoint"], str) or not pretrained["checkpoint"]:
        raise ValueError("pretrained.checkpoint must be a path")


# Returns after checking channel filters and noise. It stops with a ValueError
# if a setting cannot be used by the comparison.
def _check_filters_noise(settings) -> None:
    filters = settings["filters"]

    if not isinstance(filters["types"], list) or not filters["types"]:
        raise ValueError("filters.types must be a non-empty list")

    for kind in filters["types"]:
        if kind not in ("highpass", "lowpass", "bandpass"):
            raise ValueError("filters.types has an unknown type")

    for kind in ("highpass", "lowpass", "bandpass"):
        orders = filters["orders"][kind]

        if not isinstance(orders, list) or not orders:
            raise ValueError(f"filters.orders.{kind} must be a non-empty list")

        allowed = (1, 2, 4)

        if kind == "bandpass":
            allowed = (2, 4, 8)

        for order in orders:
            _number(order, f"filters.orders.{kind}", 1, 8, whole=True)

            if order not in allowed:
                raise ValueError(f"filters.orders.{kind} is not supported")

    for key in ("highpass_cutoff_hz", "lowpass_cutoff_hz", "bandpass_low_hz", "bandpass_high_hz"):
        _range(filters[key], f"filters.{key}", 1e-9, 23999)

    if filters["bandpass_low_hz"][1] >= filters["bandpass_high_hz"][0]:
        raise ValueError("filters.bandpass_low_hz must end below bandpass_high_hz")

    _range(settings["noise"]["snr_db"], "noise.snr_db", 20, 40)

    if settings["noise"]["stages"] != ["before_filter", "after_filter"]:
        raise ValueError("noise.stages must be [before_filter, after_filter]")
