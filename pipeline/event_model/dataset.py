"""
Build labeled clips and paired filter and noise cases from saved backgrounds.

This file has three groups:
    * Picking background stretches
    * Adding labeled sounds
    * Training clips and test cases
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.degrade import add_noise, quantize
from pipeline.event_model import events
from pipeline.event_model.labels import event_order
from pipeline.event_model.sources import render_background
from pipeline.filter_id.dataset import sample_filter
from pipeline.filters import apply_filter
from pipeline.records import build_record

# ********************************
# Picking background stretches
# ********************************


# Returns test recordings in ID order. It stops with a ValueError if the split
# names a recording that is not available.
def _test_recordings(recordings, split) -> list[dict]:
    by_id = {}

    for recording in recordings:
        by_id[recording["recording_id"]] = recording

    ordered = []

    for recording_id in sorted(split["test"]):
        if recording_id not in by_id:
            raise ValueError("split test has an unknown recording_id")

        ordered.append(by_id[recording_id])

    return ordered


# Returns the number of complete test clips. Each recording's leftover samples
# are skipped, so a clip never crosses from one recording into another.
def test_clip_count(recordings, split, settings) -> int:
    count = 0
    clip_samples = round(settings["clip_seconds"] * settings["sample_rate"])

    for recording in _test_recordings(recordings, split):
        count += sf.info(recording["background_path"]).frames // clip_samples

    return count


# Returns a recording and a clip start on a whole sample. It stops with a
# ValueError if the split is unknown or the test index is outside the split.
def _pick_background(split_name, index, recordings, split, settings, rng) -> tuple[dict, float]:
    sample_rate = settings["sample_rate"]
    count = round(settings["clip_seconds"] * sample_rate)

    if split_name == "test":
        remaining = index

        for recording in _test_recordings(recordings, split):
            clips = sf.info(recording["background_path"]).frames // count

            if remaining < clips:
                return recording, remaining * count / sample_rate

            remaining -= clips

        raise ValueError("test clip index is out of range")

    if split_name not in ("train", "tune"):
        raise ValueError("split_name must be train, tune or test")

    stretch_index = int(rng.integers(len(split[split_name])))
    stretch = split[split_name][stretch_index]
    first = int(np.ceil(stretch["start_s"] * sample_rate))
    last = int(np.floor(stretch["end_s"] * sample_rate)) - count
    start = int(rng.integers(first, last + 1))

    for recording in recordings:
        if recording["recording_id"] == stretch["recording_id"]:
            return recording, start / sample_rate

    raise ValueError("split stretch has an unknown recording_id")


# ********************************
# Adding labeled sounds
# ********************************


# Returns a start sample after at most 50 tries. It returns None if the sound
# cannot fit inside the margins with the required gap from its own type.
def _place_event(count, kind, placed, settings, rng):
    sample_rate = settings["sample_rate"]
    first = int(np.ceil(settings["edge_margin_s"] * sample_rate))
    last = round(settings["clip_seconds"] * sample_rate) - first - count
    gap = settings["events"]["same_type_gap_s"]

    if last < first:
        return None

    for _ in range(50):
        start = int(rng.integers(first, last + 1))
        end_s = (start + count) / sample_rate
        start_s = start / sample_rate
        fits = True

        for other in placed:
            if other["type"] != kind:
                continue

            if start_s < other["end_s"] + gap and end_s + gap > other["start_s"]:
                fits = False
                break

        if fits:
            return start

    return None


# Returns the inserted event labels and the number that could not fit. Event
# levels use the background's RMS before any events are added.
def _mix_events(audio, settings, rng) -> tuple[list[dict], int]:
    placed = []
    dropped = 0
    generators = {
        "rumble": events.rumble,
        "hiss": events.hiss,
        "transient": events.transient,
    }
    sample_rate = settings["sample_rate"]
    background_rms = float(np.sqrt(np.mean(audio**2)))
    low, high = settings["events"]["per_clip"]

    for kind, generate in generators.items():
        count = int(rng.integers(low, high + 1))

        for _ in range(count):
            sound, recipe = generate(settings["events"][kind], sample_rate, rng)
            start = _place_event(len(sound), kind, placed, settings, rng)

            if start is None:
                dropped += 1
                continue

            level_key = "level_db"
            measured = np.sqrt(np.mean(sound**2))

            if kind == "transient":
                level_key = "peak_db"
                measured = np.max(np.abs(sound))

            level = float(rng.uniform(*settings["events"][kind][level_key]))
            gain = background_rms * 10 ** (level / 20) / measured
            audio[start:start + len(sound)] += sound * gain

            event = {
                "type": kind,
                "start_s": start / sample_rate,
                "end_s": (start + len(sound)) / sample_rate,
                "recipe": recipe,
                level_key: level,
                "gain": float(gain),
            }
            placed.append(event)

    placed.sort(key=event_order)

    return placed, dropped


# Returns one clip's random generator. It stops with a ValueError if the split
# is unknown or the clip index is not a non-negative integer.
def _clip_rng(split_name, index, settings):
    namespaces = {"train": 20, "tune": 21, "test": 22}

    if split_name not in namespaces:
        raise ValueError("split_name must be train, tune or test")

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("clip index must be a non-negative integer")

    seed = [settings["seed"], namespaces[split_name], index]

    return np.random.default_rng(np.random.SeedSequence(seed))


# Returns clean audio and its record using the given clip generator. It stops
# with a ValueError if the selected background is silent or invalid.
def _clean_clip(split_name, index, recordings, split, settings, rng):
    namespaces = {"train": 20, "tune": 21, "test": 22}
    seed = [settings["seed"], namespaces[split_name], index]
    recording, start = _pick_background(split_name, index, recordings, split, settings, rng)
    audio = render_background(recording, start, settings["clip_seconds"], settings["sample_rate"])

    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        raise ValueError("background clip is silent")

    gain = float(0.1 / rms)
    audio *= gain
    labels, dropped = _mix_events(audio, settings, rng)

    source = {
        "recording_id": recording["recording_id"],
        "kind": recording["kind"],
        "file": Path(recording["background_path"]).name,
        "copies": recording["copies"],
    }
    clip = {
        "start_s": start,
        "duration_s": len(audio) / settings["sample_rate"],
        "gain": gain,
        "split": split_name,
        "dropped_events": dropped,
    }
    clip_id = f"{split_name}-{index:06d}"
    record = build_record(
        clip_id, seed, settings["sample_rate"], [source], None, [], clip=clip, events=labels
    )

    return audio, record


# Returns one independently rebuildable clean clip and its record. It stops with
# a ValueError if the split or index is invalid or the background is silent.
def make_clip(split_name, index, recordings, split, settings) -> tuple[np.ndarray, dict]:
    rng = _clip_rng(split_name, index, settings)

    return _clean_clip(split_name, index, recordings, split, settings, rng)


# ********************************
# Training clips and test cases
# ********************************


# Returns a train or tune clip with optional white noise and one rounding step.
# It stops with a ValueError for another split or an invalid background clip.
def make_labeled_clip(split_name, index, recordings, split, settings) -> tuple[np.ndarray, dict]:
    if split_name not in ("train", "tune"):
        raise ValueError("split_name must be train or tune for make_labeled_clip")

    rng = _clip_rng(split_name, index, settings)
    audio, record = _clean_clip(split_name, index, recordings, split, settings, rng)

    # Continue after making the events. Restarting the generator here would
    # reuse random draws from event synthesis for the background noise.
    if rng.random() < settings["clips"]["train_noise_chance"]:
        snr = float(rng.uniform(*settings["noise"]["snr_db"]))
        power = float(np.mean(audio**2))
        audio, step = add_noise(audio, snr, rng, stage="after_filter", reference_power=power)
        record["damage"].append(step)

    audio, rounding = quantize(audio, bits=settings["bits"], stage="after_filter")
    record["damage"].append(rounding)

    return audio, record


# Yields paired cases with the same noise before and after the filter. It stops
# after the requested number of conditions, or with a ValueError for a bad spec.
def make_test_cases(clean_audio, record, index, settings):
    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], 30, index]))
    power = float(np.mean(clean_audio**2))

    for condition in range(settings["clips"]["conditions_per_test_clip"]):
        spec = sample_filter(settings["filters"], rng)
        snr = float(rng.uniform(*settings["noise"]["snr_db"]))
        noise_seed = int(rng.integers(0, 2**32))

        # Start the saved noise position inside this test clip's namespace 30
        # stream.
        noise_sequence = [settings["seed"], 30, index]

        for stage in settings["noise"]["stages"]:
            noise_rng = np.random.default_rng(np.random.SeedSequence(noise_sequence))
            noise_rng.bit_generator.advance(noise_seed)

            audio = clean_audio
            damage = []

            if stage == "before_filter":
                audio, step = add_noise(audio, snr, noise_rng, stage=stage, reference_power=power)
                damage.append(step)

            audio, filter_record = apply_filter(audio, settings["sample_rate"], spec)

            if stage == "after_filter":
                audio, step = add_noise(audio, snr, noise_rng, stage=stage, reference_power=power)
                damage.append(step)

            audio, rounding = quantize(audio, bits=settings["bits"], stage="after_filter")
            damage.append(rounding)

            pair_id = f"{record['example_id']}-c{condition}"
            yield {
                "audio": audio,
                "case_id": f"{pair_id}-{stage}",
                "pair_id": pair_id,
                "spec": spec,
                "snr_db": snr,
                "noise_seed": noise_seed,
                "noise_sequence": noise_sequence,
                "stage": stage,
                "damage": damage,
                "filter": filter_record,
            }
