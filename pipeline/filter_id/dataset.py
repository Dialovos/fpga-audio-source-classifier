"""
Training data for the filter ID model.

Each example is a source clip filtered with a known Butterworth filter, with noise
added before or after the filter and 24-bit rounding at the end. The filter it used
becomes its label, and its record (pipeline.records) says exactly what was done.

Until the team's recordings are ready, the sources are synthetic. Each source is
noise with its own random spectral shape (a tilt plus a few bumps), and some get a
steady tone or a thump on top. A source keeps its shape across all its examples,
like a real recording would, and the shape is picked independently of the filter.
That way the model can't learn "this source never has much low end" in place of
the filter. Set "source_dir" to a folder of mono WAV files at the run's sample rate
to train on real recordings instead.

Debugging tip: every example rebuilds on its own from (settings, index). If one
looks wrong, call make_example(index, make_sources(settings), settings) and look at
the audio and the record it returns.

This file has six groups:
    * Checking settings
    * Synthetic sources
    * Real recordings
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
from pipeline.signals import thump, tone

# Every source clip is scaled to this level (RMS) before it's filtered.
SOURCE_RMS = 0.1

# build_dataset reports its progress every this many examples.
PROGRESS_EVERY = 500

# Ranges for the synthetic source recipes. Changing any of these changes the dataset.
#
# The spectral tilt, in dB per octave, measured around TILT_PIVOT_HZ.
# The tilt only applies inside TILT_RANGE_HZ and stays flat outside it.
SLOPE_DB_PER_OCTAVE = (-6.0, 3.0)
TILT_PIVOT_HZ = 1000.0
TILT_RANGE_HZ = (20.0, 20_000.0)

# Each source gets 0 to MAX_BUMPS bumps (boosts or dips) in its spectrum.
MAX_BUMPS = 3
BUMP_CENTER_HZ = (40.0, 20_000.0)
BUMP_WIDTH_OCTAVES = (0.3, 1.5)
BUMP_GAIN_DB = (-12.0, 12.0)

# Some sources get a steady tone on top. Its level is relative to SOURCE_RMS.
TONE_CHANCE = 0.4
TONE_FREQ_HZ = (40.0, 2000.0)
TONE_LEVEL_DB = (-20.0, 0.0)

# Some sources get a thump on top. Its level is relative to SOURCE_RMS,
# its peak is THUMP_PEAK_FACTOR times that level, and it starts somewhere in
# the first THUMP_LATEST_ONSET (80%) of the clip.
THUMP_CHANCE = 0.3
THUMP_FREQ_HZ = (30.0, 200.0)
THUMP_LEVEL_DB = (-12.0, 6.0)
THUMP_PEAK_FACTOR = 4
THUMP_LATEST_ONSET = 0.8

# Keeps the random stream for source recipes separate from the examples' seeds.
SOURCES_SPAWN_KEY = (1,)


# Everything build_dataset makes. Each field has one entry per example, in example
# order.
@dataclass
class Dataset:
    # The spectrum features, one row per example.
    features: np.ndarray

    # The labels: {"type", "order", "cutoff"} per example.
    labels: list[dict]

    # The recording_id per example, used to split the data by recording.
    groups: np.ndarray

    # The full record per example.
    records: list[dict]


# ********************************
# Checking settings
# ********************************

_REQUIRED_SETTINGS = (
    "seed",
    "sample_rate",
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


# Stops if a run's settings (see pipeline/filter_id/settings.json) are missing
# something or don't make sense. The ValueError names the first bad or missing
# setting it finds.
def check_settings(settings: dict) -> None:
    for key in _REQUIRED_SETTINGS:
        if key not in settings:
            raise ValueError(f'settings are missing "{key}"')

    # Without a folder of recordings, we need to know how many synthetic sources to make.
    if not settings.get("source_dir") and "n_sources" not in settings:
        raise ValueError('settings need "n_sources" when there is no "source_dir"')

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
# Synthetic sources
# ********************************


# Returns a random number between low and high where every octave is equally
# likely. For example, 100-200 Hz comes up as often as 1-2 kHz.
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    log_value = rng.uniform(np.log(low), np.log(high))

    return float(np.exp(log_value))


# Returns one random bump (a boost or a dip) for a source's spectral shape.
def _random_bump(rng: np.random.Generator) -> dict:
    center_hz = _log_uniform(rng, *BUMP_CENTER_HZ)
    width_octaves = float(rng.uniform(*BUMP_WIDTH_OCTAVES))
    gain_db = float(rng.uniform(*BUMP_GAIN_DB))

    return {"center_hz": center_hz, "width_octaves": width_octaves, "gain_db": gain_db}


# Returns a random recipe for one synthetic source. The recipe has a spectral
# tilt, 0 to MAX_BUMPS bumps, and maybe a steady tone and/or a thump, and it stays
# the same for every example drawn from this source. The recording_id names the
# source so the data can be split by source, and rng is the random stream for
# source recipes.
def synthetic_source(recording_id: str, rng: np.random.Generator) -> dict:
    # The order of these random draws decides the recipes, so don't reorder them.
    slope = float(rng.uniform(*SLOPE_DB_PER_OCTAVE))
    n_bumps = int(rng.integers(0, MAX_BUMPS + 1))

    bumps = []
    for _ in range(n_bumps):
        bumps.append(_random_bump(rng))

    source = {
        "recording_id": recording_id,
        "kind": "synthetic",
        "slope_db_per_octave": slope,
        "bumps": bumps,
    }

    if rng.random() < TONE_CHANCE:
        freq_hz = _log_uniform(rng, *TONE_FREQ_HZ)
        level_db = float(rng.uniform(*TONE_LEVEL_DB))
        source["tone"] = {"freq_hz": freq_hz, "level_db": level_db}

    if rng.random() < THUMP_CHANCE:
        freq_hz = _log_uniform(rng, *THUMP_FREQ_HZ)
        level_db = float(rng.uniform(*THUMP_LEVEL_DB))
        source["thump"] = {"freq_hz": freq_hz, "level_db": level_db}

    return source


# Returns a synthetic source's spectral shape, which is its boost or cut in dB at
# each frequency. The shape is the tilt plus one bell curve per bump.
def shape_gain_db(freqs: np.ndarray, source: dict) -> np.ndarray:
    # The tilt: so many dB per octave above or below the pivot frequency.
    # Clipping the frequencies to TILT_RANGE_HZ keeps the tilt flat outside that range.
    clipped_freqs = np.clip(freqs, *TILT_RANGE_HZ)
    octaves_from_pivot = np.log2(clipped_freqs / TILT_PIVOT_HZ)
    gain = source["slope_db_per_octave"] * octaves_from_pivot

    # Each bump: a bell curve on a log-frequency axis, centered on the bump's frequency.
    for bump in source["bumps"]:
        # How far each frequency is from the bump's center, in bump widths.
        # The 1 Hz floor avoids taking the log of 0 Hz.
        octaves_from_center = np.log2(np.maximum(freqs, 1.0) / bump["center_hz"])
        distance = octaves_from_center / bump["width_octaves"]

        gain = gain + bump["gain_db"] * np.exp(-0.5 * distance**2)

    return gain


# ********************************
# Real recordings
# ********************************


# Returns one source per mono WAV file in source_dir, after checking each file.
# Each file's name (without .wav) becomes its recording_id. Every file has to
# match the run's sample rate and be at least clip_seconds long. Each source also
# gets a "_path", and keys starting with "_" are private, so they don't go in the
# records. It stops with a ValueError if the folder has no WAV files, or a file
# has the wrong sample rate, isn't mono, or is too short.
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


# Returns every source for a run. If the settings have a "source_dir", the sources
# are the WAV files in it. Otherwise they're n_sources synthetic recipes.
def make_sources(settings: dict) -> list[dict]:
    if settings.get("source_dir"):
        return wav_sources(settings["source_dir"], settings["sample_rate"], settings["clip_seconds"])

    seed_sequence = np.random.SeedSequence(settings["seed"], spawn_key=SOURCES_SPAWN_KEY)
    rng = np.random.default_rng(seed_sequence)

    sources = []
    for k in range(settings["n_sources"]):
        recording_id = f"synthetic-{k:04d}"
        sources.append(synthetic_source(recording_id, rng))

    return sources


# ********************************
# Rendering a source clip
# ********************************


# Returns a random piece of a WAV source, n_samples long.
def _render_wav_clip(source: dict, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    # Pick a random starting point that leaves room for the whole clip.
    total_frames = sf.info(source["_path"]).frames
    start = int(rng.integers(0, total_frames - n_samples + 1))

    audio, _ = sf.read(source["_path"], start=start, frames=n_samples, dtype="float64")

    return audio


# Returns a fresh clip of a synthetic source: shaped noise, plus its tone and
# thump if it has them.
def _render_synthetic_clip(source: dict, seconds: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = round(seconds * sample_rate)

    # Make white noise and move it into the frequency domain.
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples, 1 / sample_rate)

    # Shape it: turn the source's dB shape into a multiplier, apply it, and come back to audio.
    multiplier = 10 ** (shape_gain_db(freqs, source) / 20)
    audio = np.fft.irfft(spectrum * multiplier, n_samples)

    audio *= SOURCE_RMS / np.sqrt(np.mean(audio**2))

    if "tone" in source:
        # The tone's level in dB is relative to SOURCE_RMS. A sine's peak is sqrt(2) times its RMS.
        level = SOURCE_RMS * 10 ** (source["tone"]["level_db"] / 20)
        peak = level * np.sqrt(2)

        audio = audio + tone(source["tone"]["freq_hz"], seconds, sample_rate, amplitude=peak)

    if "thump" in source:
        # The thump starts at a random time and can't peak past full scale (1.0).
        level = SOURCE_RMS * 10 ** (source["thump"]["level_db"] / 20)
        onset = float(rng.uniform(0.0, THUMP_LATEST_ONSET * seconds))
        peak = min(1.0, THUMP_PEAK_FACTOR * level)

        audio = audio + thump(seconds, sample_rate, onset_s=onset, freq_hz=source["thump"]["freq_hz"], amplitude=peak)

    return audio


# Returns one clip of a source, scaled to SOURCE_RMS. Synthetic sources get fresh
# noise each time, and WAV sources get a random piece of the file. It stops with a
# ValueError if the clip comes out silent (for example, a silent stretch of a WAV
# file).
def render_source(source: dict, seconds: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    if source["kind"] == "wav":
        n_samples = round(seconds * sample_rate)
        audio = _render_wav_clip(source, n_samples, rng)
    else:
        audio = _render_synthetic_clip(source, seconds, sample_rate, rng)

    # A silent clip can't be scaled to a level.
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        raise ValueError(f"source {source['recording_id']} gave a silent clip")

    return audio * (SOURCE_RMS / rms)


# ********************************
# Random filters
# ********************************


# Returns a random Butterworth filter spec, using the run's filter settings. The
# type and order are picked from the settings' lists, each equally likely, and
# the cutoffs are picked so every octave in their range is equally likely.
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


# Adds noise on the chosen side of the filter, and applies the filter. It returns
# the audio, the filter's record, and the noise step's record in a list.
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


# Returns the source without its private keys, which are the ones starting with
# "_", like a file path on this computer.
def _public(source: dict) -> dict:
    public = {}

    for key, value in source.items():
        if not key.startswith("_"):
            public[key] = value

    return public


# Builds example number `index` from its own seed. It renders a clip of the
# example's source, picks a random filter, adds noise before or after the filter
# at a random SNR, and rounds to the run's bit depth. Each source gets
# examples_per_source examples in a row. It returns the audio and a record with
# the exact source, filter, noise, and rounding.
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


# Returns the labels the model learns for one example, taken from the filter that
# was really applied.
def label_of(record: dict) -> dict:
    applied = record["filter"]

    return {"type": applied["type"], "order": applied["order"], "cutoff": applied["cutoff"]}


# Makes every example for a run, and turns each one into features and labels. The
# settings get checked first. If a progress function is given, it's called as
# progress(done, total) every PROGRESS_EVERY examples. It returns a Dataset with
# one entry per example, and stops with a ValueError if the settings or a WAV
# source is bad.
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
