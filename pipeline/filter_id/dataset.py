"""
Layer 1, step 2: make the practice examples the model learns from.

The model needs examples where we already know the right answer. So we make them ourselves: take a 1-second piece of
clean speech, run it through a filter that we pick at random, add a little noise (before or after the filter), and round
it the way a recorder would. The filter we picked is the answer the model has to learn, and the example's record (see
pipeline/records.py) says exactly what was done to it.

The speech comes from the mono WAV files in the run's "source_dir". The filter is picked without looking at which
recording it is. That matters: if a certain voice always got a certain filter, the model could learn "this voice never
has much low end" & skip learning the filter.

Debugging tip: every example can be rebuilt by itself from (settings, index). If one looks wrong, call
make_example(index, make_sources(settings), settings) & look at the audio and the record it gives back.

This file has five groups:
    * Checking settings
    * Recordings
    * Rendering a source clip
    * Random filters
    * Making examples
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.degrade import STAGES, add_noise, quantize
from pipeline.filter_id.features import spectrum_features
from pipeline.filters import FILTER_TYPES, apply_filter
from pipeline.records import build_record, example_seed

# Every piece of speech gets scaled to this level before it's filtered. RMS is the average level of the audio, so all
# the examples start out equally loud.
SOURCE_RMS = 0.1

# build_dataset reports how far along it is every this many examples.
PROGRESS_EVERY = 500


# Everything build_dataset hands back. Each of the 4 fields has 1 entry per example, in the same order.
@dataclass
class Dataset:
    # The 96 numbers for each example, 1 row per example.
    features: np.ndarray

    # The right answers: {"type", "order", "cutoff"} for each example.
    labels: list[dict]

    # Which recording each example came from. We use it to split the data by recording.
    groups: np.ndarray

    # The full record of how each example was made.
    records: list[dict]


# ********************************
# Checking settings
# ********************************

_REQUIRED_SETTINGS = (
    "seed",
    "sample_rate",
    "source_dir",
    "clip_seconds",
    "examples_per_source",
    "test_fraction",
    "bits",
    "filters",
    "noise",
    "features",
)

# Settings every run needs inside "filters".
_REQUIRED_FILTER_SETTINGS = (
    "types",
    "orders",
    "highpass_cutoff_hz",
    "lowpass_cutoff_hz",
    "bandpass_low_hz",
    "bandpass_high_hz",
)


# Looks over a run's settings before anything gets made. If a setting is missing or doesn't make sense, it stops with a
# ValueError that names the first problem it found.
def check_settings(settings: dict) -> None:
    for key in _REQUIRED_SETTINGS:
        if key not in settings:
            raise ValueError(f'settings are missing "{key}"')

    if not isinstance(settings["source_dir"], str) or not settings["source_dir"]:
        raise ValueError('"source_dir" must name a folder of mono WAV files')

    if settings["examples_per_source"] < 1:
        raise ValueError(f"examples_per_source must be at least 1, got {settings['examples_per_source']}")

    if not 0 < settings["test_fraction"] < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {settings['test_fraction']}")

    filters = settings["filters"]

    for key in _REQUIRED_FILTER_SETTINGS:
        if key not in filters:
            raise ValueError(f'settings["filters"] is missing "{key}"')

    for kind in filters["types"]:
        if kind not in FILTER_TYPES:
            raise ValueError(f"unknown filter type in settings: {kind!r}")

        if not filters["orders"].get(kind):
            raise ValueError(f'settings["filters"]["orders"] has no orders for {kind!r}')

    # A band-pass's low edge has to stay below its high edge, so the two ranges can't overlap.
    highest_low_edge = filters["bandpass_low_hz"][1]
    lowest_high_edge = filters["bandpass_high_hz"][0]

    if highest_low_edge >= lowest_high_edge:
        raise ValueError("bandpass_low_hz has to end below where bandpass_high_hz starts")

    for stage in settings["noise"]["stages"]:
        if stage not in STAGES:
            raise ValueError(f"unknown noise stage in settings: {stage!r}")


# ********************************
# Recordings
# ********************************


# Finds every WAV file in source_dir & gives back 1 source per file, after checking each one. The file's name (without
# .wav) becomes its recording_id. Every file has to match the run's sample rate, be mono & be at least clip_seconds
# long. Each source also carries a "_path". Keys that start with "_" are private (like a path on this computer), so they
# stay out of the saved records. It stops with a ValueError if the folder has no WAV files, or if a file has the wrong
# sample rate, isn't mono, or is too short.
def wav_sources(source_dir, sample_rate: int, clip_seconds: float) -> list[dict]:
    paths = sorted(Path(source_dir).glob("*.wav"))

    if len(paths) == 0:
        raise ValueError(f"no .wav files in {source_dir}")

    clip_samples = round(clip_seconds * sample_rate)
    sources = []

    for path in paths:
        # Read the file's details (not the audio itself).
        info = sf.info(path)

        if info.samplerate != sample_rate:
            raise ValueError(f"{path.name} is {info.samplerate} Hz, but this run uses {sample_rate} Hz")

        if info.channels != 1:
            raise ValueError(f"{path.name} has {info.channels} channels; sources must be mono")

        if info.frames < clip_samples:
            raise ValueError(f"{path.name} is shorter than one {clip_seconds} s clip")

        source = {"recording_id": path.stem, "kind": "wav", "file": path.name, "_path": str(path)}
        sources.append(source)

    return sources


# Gives back every source for a run, which is the WAV files in its source_dir.
def make_sources(settings: dict) -> list[dict]:
    return wav_sources(settings["source_dir"], settings["sample_rate"], settings["clip_seconds"])


# ********************************
# Rendering a source clip
# ********************************


# Reads a random piece of 1 WAV file, n_samples long.
def _render_wav_clip(source: dict, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    # Pick a random starting point that leaves room for the whole clip.
    total_frames = sf.info(source["_path"]).frames
    start = int(rng.integers(0, total_frames - n_samples + 1))

    audio, _ = sf.read(source["_path"], start=start, frames=n_samples, dtype="float64")

    return audio


# Gives back a random piece of a source, scaled to SOURCE_RMS so every example starts at the same level. It stops with a
# ValueError if the piece comes out silent (like a silent stretch of the file), since silence can't be scaled to a
# level.
def render_source(source: dict, seconds: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = round(seconds * sample_rate)
    audio = _render_wav_clip(source, n_samples, rng)

    # A silent clip can't be scaled to a level.
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        raise ValueError(f"source {source['recording_id']} gave a silent clip")

    return audio * (SOURCE_RMS / rms)


# ********************************
# Random filters
# ********************************


# Picks a random number between low & high where every octave is equally likely. So 100-200 Hz comes up as often as 1-2
# kHz. We use it for cutoffs, since that's how we hear pitch.
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    log_value = rng.uniform(np.log(low), np.log(high))

    return float(np.exp(log_value))


# Picks a random Butterworth filter using the run's filter settings. The type & the order come from the settings' lists,
# each choice equally likely. The cutoff comes from that type's range, with every octave equally likely. It gives back a
# filter spec, which is a small dict with the type, family, order & cutoff.
def sample_filter(filters: dict, rng: np.random.Generator) -> dict:
    # The order of these random draws decides the dataset, so don't reorder them.
    kind = str(rng.choice(filters["types"]))
    order = int(rng.choice(filters["orders"][kind]))

    if kind == "bandpass":
        low = _log_uniform(rng, *filters["bandpass_low_hz"])
        high = _log_uniform(rng, *filters["bandpass_high_hz"])
        cutoff = [low, high]
    else:
        cutoff = _log_uniform(rng, *filters[f"{kind}_cutoff_hz"])

    return {"type": kind, "family": "butter", "order": order, "cutoff": cutoff}


# ********************************
# Making examples
# ********************************


# Applies the filter, and adds the noise on whichever side of the filter the example calls for. It gives back the audio,
# the filter's record & the noise step's record in a list.
def _filter_with_noise(
    audio: np.ndarray,
    spec: dict,
    snr_db: float,
    stage: str,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict, list[dict]]:
    damage = []

    # Noise before the filter: the filter shapes the noise too.
    if stage == "before_filter":
        audio, step = add_noise(audio, snr_db, rng, stage=stage)
        damage.append(step)

    audio, filter_record = apply_filter(audio, sample_rate, spec)

    # Noise after the filter: the noise covers the whole spectrum, even where the filter cut.
    if stage == "after_filter":
        audio, step = add_noise(audio, snr_db, rng, stage=stage)
        damage.append(step)

    return audio, filter_record, damage


# Gives back a copy of the source without its private keys, which are the ones that start with "_", like a file path on
# this computer.
def _public(source: dict) -> dict:
    public = {}

    for key, value in source.items():
        if not key.startswith("_"):
            public[key] = value

    return public


# Builds example number `index`, using its own seed so it comes out the same every time. It takes a random piece of the
# example's source, picks a random filter, adds noise before or after the filter at a random SNR (how many dB louder the
# speech is than the noise), and rounds to the run's bit depth. Each source gets examples_per_source examples in a row.
# It gives back the audio & a record of the exact source, filter, noise & rounding.
def make_example(index: int, sources: list[dict], settings: dict) -> tuple[np.ndarray, dict]:
    seed = example_seed(settings["seed"], index)
    rng = np.random.default_rng(seed)

    # Examples 0 to examples_per_source - 1 use source 0, the next ones use source 1, and so on.
    source = sources[index // settings["examples_per_source"]]
    sample_rate = settings["sample_rate"]

    # The order of these random draws decides the dataset, so don't reorder them.
    audio = render_source(source, settings["clip_seconds"], sample_rate, rng)
    spec = sample_filter(settings["filters"], rng)
    snr_db = float(rng.uniform(*settings["noise"]["snr_db"]))
    stage = str(rng.choice(settings["noise"]["stages"]))

    audio, filter_record, damage = _filter_with_noise(audio, spec, snr_db, stage, sample_rate, rng)

    # Round to the bit depth last, like a recorder saving the file.
    audio, rounding = quantize(audio, bits=settings["bits"], stage="after_filter")
    damage.append(rounding)

    example_id = f"ex-{index:06d}"
    record = build_record(example_id, seed, sample_rate, [_public(source)], filter_record, damage)

    return audio, record


# Gives back the right answer for 1 example (type, order & cutoff), taken from the filter that was really applied.
def label_of(record: dict) -> dict:
    applied = record["filter"]

    return {"type": applied["type"], "order": applied["order"], "cutoff": applied["cutoff"]}


# Makes every example for a run & turns each one into its 96 numbers & its right answer. The settings get checked first.
# If you pass a progress function, it gets called as progress(done, total) every PROGRESS_EVERY examples. It gives back
# a Dataset with 1 entry per example. It stops with a ValueError if the settings or a WAV file are bad.
def build_dataset(settings: dict, progress=None) -> Dataset:
    check_settings(settings)

    sources = make_sources(settings)
    total = len(sources) * settings["examples_per_source"]

    features = []
    labels = []
    groups = []
    records = []

    for index in range(total):
        audio, record = make_example(index, sources, settings)

        features.append(spectrum_features(audio, settings["sample_rate"], **settings["features"]))
        labels.append(label_of(record))
        groups.append(record["sources"][0]["recording_id"])
        records.append(record)

        # Report progress now and then, since a full build takes over a minute.
        examples_done = index + 1
        if progress is not None and examples_done % PROGRESS_EVERY == 0:
            progress(examples_done, total)

    return Dataset(np.array(features), labels, np.array(groups), records)
