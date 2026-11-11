"""
Walk through every part of the pipeline from short notebook cells.

pipeline/notebooks/pipeline-walkthrough.ipynb calls these functions, and
pipeline-walkthrough-settings.json in the repo root holds the settings it
reads.

This file has six groups:
    * Settings
    * The test sound and the damage
    * Guessing and undoing the filter
    * Plots of the sound
    * The event model
    * Running the tests
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Patch
from scipy import signal

from pipeline.compensation import compensate
from pipeline.degrade import STAGES, add_noise, quantize
from pipeline.event_model.__main__ import main as event_model_main
from pipeline.event_model.comparison import _ai_inverse
from pipeline.event_model.dataset import make_clip
from pipeline.event_model.labels import EVENT_TYPES
from pipeline.event_model.model import load_model, prepare_clip
from pipeline.event_model.scoring import match_events
from pipeline.event_model.settings import check_settings
from pipeline.event_model.sources import load_split, recordings
from pipeline.filter_id.dataset import SOURCE_RMS
from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import load_model as load_filter_id
from pipeline.filters import apply_filter, design_filter, response_db
from pipeline.records import load_settings
from pipeline.signals import thump, tone, white_noise

SOUND_KINDS = ("wav", "tone", "thump", "noise")

VERSION_NAMES = {
    "clean": "Clean",
    "damaged": "Damaged",
    "filtered": "Filtered",
    "known_undone": "Known filter undone",
    "ai_undone": "AI estimate undone",
}

# ********************************
# Settings
# ********************************


# Returns the walkthrough settings from a JSON file. It stops with a ValueError
# if the seed is missing, the sound kind is unknown, or the noise stage is bad.
def read_settings(path="pipeline-walkthrough-settings.json") -> dict:
    cfg = load_settings(path)

    if cfg["sound"]["kind"] not in SOUND_KINDS:
        raise ValueError(f"sound.kind must be one of {SOUND_KINDS}")

    if cfg["noise"]["stage"] not in STAGES:
        raise ValueError(f"noise.stage must be one of {STAGES}")

    return cfg


# ********************************
# The test sound and the damage
# ********************************


# Returns one piece of a mono WAV file. It stops with a FileNotFoundError if the
# file is missing, and a ValueError if it isn't mono, is at the wrong rate, or
# is too short for the piece.
def _read_piece(path, start_s, seconds, sample_rate) -> np.ndarray:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} is missing. Put the WAV there or change sound.wav.")

    info = sf.info(path)

    if info.channels != 1:
        raise ValueError(f"{path.name} must be mono")

    if info.samplerate != sample_rate:
        raise ValueError(f"{path.name} is {info.samplerate} Hz, but sample_rate is {sample_rate}")

    first = round(start_s * sample_rate)
    count = round(seconds * sample_rate)

    if first < 0 or first + count > info.frames:
        raise ValueError(f"{path.name} is too short for {seconds} s from {start_s} s")

    audio, _ = sf.read(path, start=first, frames=count, dtype="float64")

    return audio


# Returns the clean test sound scaled to RMS 0.1, the level the filter ID model
# trained at. A WAV gives the piece from start_s, and the other kinds come from
# pipeline/signals.py. It stops with a ValueError if the sound is silent or a
# setting is bad.
def make_sound(cfg) -> np.ndarray:
    sound = cfg["sound"]
    sample_rate = cfg["sample_rate"]
    seconds = sound["seconds"]

    if sound["kind"] == "wav":
        audio = _read_piece(sound["wav"], sound["start_s"], seconds, sample_rate)
        what = f"{Path(sound['wav']).name} from {sound['start_s']} s"
    elif sound["kind"] == "tone":
        audio = tone(sound["freq_hz"], seconds, sample_rate)
        what = f"a {sound['freq_hz']} Hz tone"
    elif sound["kind"] == "thump":
        audio = thump(seconds, sample_rate, freq_hz=sound["freq_hz"])
        what = f"a {sound['freq_hz']} Hz thump"
    else:
        # The sound and the added noise get separate random streams from the same seed.
        rng = np.random.default_rng([cfg["seed"], 0])
        audio = white_noise(seconds, sample_rate, rng)
        what = "white noise"

    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        raise ValueError("the test sound is silent; pick another sound or start_s")

    print(f"{seconds} s of {what}, scaled to RMS {SOURCE_RMS}")

    return audio * (SOURCE_RMS / rms)


# Returns the damaged sound and its steps in order: the filter, the noise
# (before or after the filter, or none when snr_db is null), and rounding to
# the bit depth. The SNR is measured against the clean sound, so noise before
# and after the filter is the same amount. It stops with a ValueError if the
# filter spec or a noise or rounding setting is bad.
def damage(clean, cfg) -> tuple[np.ndarray, list]:
    snr_db = cfg["noise"]["snr_db"]
    stage = cfg["noise"]["stage"]
    clean_power = float(np.mean(clean**2))
    rng = np.random.default_rng([cfg["seed"], 1])
    steps = []
    audio = clean

    if snr_db is not None and stage == "before_filter":
        audio, step = add_noise(audio, snr_db, rng, stage=stage, reference_power=clean_power)
        steps.append(step)

    audio, step = apply_filter(audio, cfg["sample_rate"], cfg["filter"])
    steps.append(step)

    if snr_db is not None and stage == "after_filter":
        audio, step = add_noise(audio, snr_db, rng, stage=stage, reference_power=clean_power)
        steps.append(step)

    audio, step = quantize(audio, bits=cfg["bits"], stage="after_filter")
    steps.append(step)

    print(f"Filter: {_spec_text(cfg['filter'])}")

    if snr_db is None:
        print("Noise: none")
    else:
        print(f"Noise: {snr_db} dB SNR, {stage.replace('_', ' ')}")

    print(f"Rounding: {cfg['bits']}-bit, {steps[-1]['clipped_samples']} samples clipped")

    return audio, steps


# ********************************
# Guessing and undoing the filter
# ********************************


# Returns a filter spec as short text, like "bandpass order 4, 250-6000 Hz".
def _spec_text(spec) -> str:
    cutoff = spec["cutoff"]

    if isinstance(cutoff, list):
        cutoff_text = f"{cutoff[0]:.0f}-{cutoff[1]:.0f} Hz"
    else:
        cutoff_text = f"{cutoff:.0f} Hz"

    return f"{spec['type']} order {spec['order']}, {cutoff_text}"


# Returns the filter ID model's guess for the damaged sound as {"spec",
# "confidence"}, or None when the model file is missing. It stops with a
# ValueError if the model was trained at another sample rate.
def guess_filter(damaged, cfg):
    path = Path(cfg["filter_id_model"])

    if not path.is_file():
        print(f"No filter ID model at {path.as_posix()}, so there's no guess. Set filter_id_model to a trained one.")
        return None

    model, settings = load_filter_id(path)

    if settings["sample_rate"] != cfg["sample_rate"]:
        raise ValueError(f"the filter ID model was trained at {settings['sample_rate']} Hz")

    features = spectrum_features(damaged, settings["sample_rate"], **settings["features"])
    guess = model.predict(features)[0]
    confidence = guess["confidence"]

    print(f"Real filter: {_spec_text(cfg['filter'])}")
    print(f"Guess:       {_spec_text(guess['spec'])}")
    print(f"Confidence:  type {confidence['type']:.2f}, order {confidence['order']:.2f}")

    return guess


# Returns every version of the sound by name: clean, damaged, the filter undone
# with the real settings, and, when there's a guess, undone with the guess. A
# guess the inverse can't use is left out with a message. It stops with a
# ValueError if an inverse run fails.
def undo_filter(clean, damaged, guess, cfg) -> dict:
    method = cfg["undo"]["method"]
    command = None

    if method == "c":
        command = [cfg["undo"]["c_command"]]

    versions = {"clean": clean, "damaged": damaged}
    known, _ = compensate(damaged, cfg["sample_rate"], cfg["filter"], method=method, command=command)
    versions["known_undone"] = known

    if guess is not None:
        recovered, _, problem = _ai_inverse(damaged, cfg, guess["spec"], method, command)

        if problem is None:
            versions["ai_undone"] = recovered
        else:
            print(f"The guess can't be undone: {problem}")

    for name, audio in versions.items():
        peak = float(np.max(np.abs(audio)))
        print(f"{VERSION_NAMES[name]:<22} peak {peak:.3f}")

    return versions


# Returns the written paths after saving each version as a 32-bit float WAV, so
# levels past full scale survive for listening in a DAW.
def save_wavs(versions, cfg) -> list[str]:
    folder = Path(cfg["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = []

    for name, audio in versions.items():
        path = folder / f"{name}.wav"
        sf.write(path, audio, cfg["sample_rate"], subtype="FLOAT")
        paths.append(path.as_posix())

    print(f"Saved {len(paths)} WAVs in {folder.as_posix()}/")

    return paths


# ********************************
# Plots of the sound
# ********************************


# Returns a plot of the real filter's response, plus the guess's when there is
# one. A guess that can't be designed is left out with a message.
def plot_filters(cfg, guess=None):
    sample_rate = cfg["sample_rate"]
    freqs = np.geomspace(10, sample_rate / 2, 2000)
    specs = {"Real filter": cfg["filter"]}

    if guess is not None:
        specs["Filter ID guess"] = guess["spec"]

    figure, axis = plt.subplots(figsize=(9, 3.5), dpi=90, layout="constrained")

    for name, spec in specs.items():
        try:
            sos = design_filter(spec, sample_rate)
        except ValueError as error:
            print(f"Can't draw the {name.lower()}: {error}")
            continue

        gain = response_db(sos, sample_rate, freqs)
        axis.semilogx(freqs, gain, label=f"{name}: {_spec_text(spec)}")

    axis.axhline(-3, color="gray", linewidth=0.8, linestyle="--")
    axis.set_xlim(10, sample_rate / 2)
    axis.set_ylim(-80, 5)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Gain (dB)")
    axis.set_title("Filter response (dashed line: -3 dB)")
    axis.grid(alpha=0.2, which="both")
    axis.legend(loc="lower center", fontsize=8)

    return figure


# Returns a plot of each version's average spectrum, so you can see what the
# filter cut and how much undoing it brought back.
def plot_spectra(versions, cfg):
    sample_rate = cfg["sample_rate"]
    figure, axis = plt.subplots(figsize=(9, 3.5), dpi=90, layout="constrained")

    for name, audio in versions.items():
        segment = min(8192, len(audio))
        freqs, power = signal.welch(audio, fs=sample_rate, nperseg=segment)
        decibels = 10 * np.log10(np.maximum(power[1:], 1e-20))
        axis.semilogx(freqs[1:], decibels, label=VERSION_NAMES[name], linewidth=1)

    axis.set_xlim(20, sample_rate / 2)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power (dB/Hz)")
    axis.set_title("Average spectrum of each version")
    axis.grid(alpha=0.2, which="both")
    axis.legend(loc="lower left", fontsize=8)

    return figure


# ********************************
# The event model
# ********************************


# Returns the event model settings: the settings file, with the seed and clip
# counts from the walkthrough settings, pointed at the team's voiceovers. Each
# raw voiceover and its noisy copy make one recording. It stops with a
# FileNotFoundError listing missing WAVs and where to put them.
def event_model_settings(cfg) -> dict:
    section = cfg["event_model"]
    settings = load_settings(section["settings_file"])
    settings["seed"] = section["seed"]
    settings["clips"]["train_clips"] = section["train_clips"]
    settings["clips"]["tune_clips"] = section["tune_clips"]
    folder = Path(section["voiceover_dir"])
    groups = {}
    missing = []

    for name in ("VO1", "VO2", "VO3"):
        groups[name] = {"background": f"{name}.wav", "copies": [f"{name}_noise.wav"]}

        for filename in (f"{name}.wav", f"{name}_noise.wav"):
            if not (folder / filename).is_file():
                missing.append(filename)

    if missing:
        raise FileNotFoundError(f"Put the team's WAVs in {folder.as_posix()}/. Missing: {', '.join(missing)}")

    settings["sources"]["source_dir"] = folder.as_posix()
    settings["sources"]["recording_groups"] = groups
    check_settings(settings)

    return settings


# Returns a score rounded to three decimals, or n/a when it is undefined.
def _score_text(value) -> str:
    if value is None:
        return "n/a"

    return f"{value:.3f}"


# Returns after printing saved tune thresholds and scores, without test scores.
# It stops if the run's metrics file is missing or invalid.
def _print_tuning(run_dir) -> None:
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    print(f"{'Type':<12} {'Threshold':>9} {'Precision':>9} {'Recall':>8} {'F1':>8}")

    for kind in EVENT_TYPES:
        scores = metrics["tune_scores"]["scores"][kind]
        values = []

        for key in ("precision", "recall", "f1"):
            values.append(_score_text(scores[key]))

        print(f"{kind:<12} {metrics['thresholds'][kind]:>9.2f} {values[0]:>9} {values[1]:>8} {values[2]:>8}")

    print("Tune clips share recordings with training; held-out scores follow in the comparison.")


# Returns the saved model and settings, training only if model.joblib is absent.
# It stops on invalid settings or a mismatch with an existing run's settings.
def train_or_load(settings, run_dir) -> tuple:
    check_settings(settings)
    folder = Path(run_dir)
    model_path = folder / "model.joblib"

    if model_path.is_file():
        model, saved_settings = load_model(model_path)

        if settings != saved_settings:
            raise ValueError("Settings differ from the saved model; use its settings or another run folder.")

        print("Loaded the trained event model.")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        settings_path = folder / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        print("Training the event model; progress is saved in training.log.", flush=True)

        with (folder / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            event_model_main(["train", "--settings", str(settings_path), "--out", str(folder)])

        model, saved_settings = load_model(model_path)
        print("Training finished and the model was saved.")

    _print_tuning(folder)

    return model, saved_settings


# Returns audio, its labels and detections for one saved test-split clip. It
# stops if the split, sources or index are invalid, or the model is not fitted.
def make_test_clip(model, settings, run_dir, index=0) -> tuple:
    items = recordings(settings)
    split = load_split(Path(run_dir) / "split.json", items, settings)
    audio, record = make_clip("test", index, items, split, settings)
    clip = prepare_clip(audio, record, settings)
    detected = model.detect(clip.features, clip.n_frames, clip.duration_s)
    print(f"{record['example_id']}: held-out {clip.recording_id}, source start {record['clip']['start_s']:.1f} s")

    return audio, record, detected


# Returns after drawing intervals and onset ticks for one type and track.
def _draw_events(axis, events, kind, row, matched, unmatched_color) -> None:
    for index, event in enumerate(events):
        if event["type"] != kind:
            continue

        color = "seagreen"

        if index not in matched:
            color = unmatched_color

        axis.hlines(row, event["start_s"], event["end_s"], color=color, linewidth=7)
        axis.plot(event["start_s"], row, "|", color=color, markersize=12)


# Returns a modest-size spectrogram and per-type label/detection tracks using
# the scoring rules for hit colors. It stops if the audio or labels are invalid.
def plot_test_clip(audio, record, detected, settings):
    labeled = record["events"]
    counts = match_events(detected, labeled, settings["scoring"])

    # The spectrogram on top, then one short track per event type.
    tracks = len(EVENT_TYPES)
    figure, axes = plt.subplots(
        1 + tracks, 1, figsize=(11, 3 + 0.75 * tracks), dpi=90, sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * tracks}, layout="constrained",
    )
    frequencies, times, power = signal.spectrogram(
        audio, fs=settings["sample_rate"], nperseg=4096, noverlap=2048,
    )
    useful = (frequencies >= 20) & (frequencies <= 20000)
    decibels = 10 * np.log10(np.maximum(power[useful], 1e-14))
    peak = float(np.max(decibels))
    image = axes[0].pcolormesh(
        times, frequencies[useful], decibels, shading="auto",
        cmap="magma", vmin=peak - 80, vmax=peak,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(20, 20000)
    axes[0].set_yticks([20, 100, 1000, 10000, 20000], ["20", "100", "1k", "10k", "20k"])
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title(f"{record['example_id']} on {record['sources'][0]['recording_id']}")
    figure.colorbar(image, ax=list(axes), label="Power (dB/Hz)", shrink=0.65)

    for axis, kind in zip(axes[1:], EVENT_TYPES):
        matched_detections = set()
        matched_labels = set()

        for detection_index, label_index in counts[kind]["pairs"]:
            matched_detections.add(detection_index)
            matched_labels.add(label_index)

        _draw_events(axis, labeled, kind, 1, matched_labels, "darkorange")
        _draw_events(axis, detected, kind, 0, matched_detections, "crimson")
        axis.set_yticks([0, 1], ["Detected", "Labeled"])
        axis.set_ylim(-0.6, 1.6)
        axis.set_ylabel(kind)
        axis.set_title(
            f"{counts[kind]['tp']} hits, {counts[kind]['fn']} misses, {counts[kind]['fp']} false alarms",
            loc="left", fontsize=9,
        )
        axis.grid(axis="x", alpha=0.2)

    legend = [
        Patch(color="seagreen", label="Hit"),
        Patch(color="darkorange", label="Miss"),
        Patch(color="crimson", label="False alarm"),
    ]
    axes[0].legend(handles=legend, loc="upper right", fontsize=8)
    axes[-1].set_xlim(0, len(audio) / settings["sample_rate"])
    axes[-1].set_xlabel("Time in clip (s); ticks mark event onsets")

    return figure


# Returns the saved comparison metrics. It leaves out the AI estimate version
# when the filter ID file is absent, and stops on command errors. The C method
# needs the built C program at c_command, which only builds on Linux.
def compare_versions(run_dir, filter_id_path=None, method="python", c_command=None) -> dict:
    folder = Path(run_dir)
    output = folder / "comparison"
    arguments = [
        "compare", "--model", str(folder / "model.joblib"),
        "--method", method, "--out", str(output),
    ]

    if method == "c":
        arguments.extend(["--c-command", str(c_command)])

    if filter_id_path is not None and Path(filter_id_path).is_file():
        arguments.extend(["--filter-id-model", str(filter_id_path)])
        print("Filter ID model found; including AI estimate undone.")
    else:
        print("Filter ID model missing; AI estimate undone is omitted (three versions).")

    print("Running the comparison; progress is saved in comparison.log.", flush=True)

    with (folder / "comparison.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        event_model_main(arguments)

    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


# Returns an F1 chart after printing precision, recall and F1 per version/type.
# Undefined scores print as n/a and leave a gap in the chart.
def show_comparison(metrics):
    versions = list(metrics["versions"])
    print(f"{metrics['test_clips']} held-out clips, {metrics['case_count']} paired cases; inverse: {metrics['method']}")
    print(f"{'Version':<22} {'Type':<11} {'Precision':>9} {'Recall':>8} {'F1':>8}")

    for version in versions:
        for kind in EVENT_TYPES:
            scores = metrics["versions"][version]["scores"][kind]
            values = []

            for key in ("precision", "recall", "f1"):
                values.append(_score_text(scores[key]))

            print(f"{VERSION_NAMES[version]:<22} {kind:<11} {values[0]:>9} {values[1]:>8} {values[2]:>8}")

    figure, axis = plt.subplots(figsize=(10, 3.5), dpi=90, layout="constrained")
    positions = np.arange(len(versions))

    # Each version gets a group of bars, one per type, centered on its tick.
    width = 0.8 / len(EVENT_TYPES)

    for column, kind in enumerate(EVENT_TYPES):
        values = []

        for version in versions:
            value = metrics["versions"][version]["scores"][kind]["f1"]

            if value is None:
                value = np.nan

            values.append(value)

        offset = (column - (len(EVENT_TYPES) - 1) / 2) * width
        axis.bar(positions + offset, values, width=width, label=kind)

    labels = []

    for version in versions:
        labels.append(VERSION_NAMES[version].replace(" undone", "\nundone"))

    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Event F1")
    axis.set_title("Same held-out cases for every audio version")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    return figure


# Returns the command's events for one WAV. It stops with the command's message
# if the WAV is not mono at the model's sample rate, or cannot be read.
def detect_file(run_dir, wav_path) -> list[dict]:
    output = io.StringIO()

    with redirect_stdout(output):
        event_model_main(["detect", "--model", str(Path(run_dir) / "model.joblib"), str(wav_path)])

    return json.loads(output.getvalue())["events"]


# Returns after printing how many events of each type were found, then the
# first `shown` events with their type, start, end and confidence, or an
# explicit message when the event list is empty. A long recording can have
# hundreds of events, so the rest are only counted.
def print_events(events, shown=20) -> None:
    print(f"{len(events)} detected events")

    if not events:
        print("No events detected.")
        return

    for kind in EVENT_TYPES:
        count = 0

        for event in events:
            if event["type"] == kind:
                count += 1

        print(f"  {kind}: {count}")

    print(f"{'Type':<12} {'Start (s)':>10} {'End (s)':>10} {'Confidence':>11}")

    for event in events[:shown]:
        print(f"{event['type']:<12} {event['start_s']:>10.3f} {event['end_s']:>10.3f} {event['confidence']:>11.3f}")

    if len(events) > shown:
        print(f"... and {len(events) - shown} more")


# ********************************
# Running the tests
# ********************************


# Returns pytest's exit code after running the tests in path (all of them by
# default), printing any failed tests and pytest's last line. 0 means every
# test passed.
def run_tests(path="pipeline/tests") -> int:
    print("Running the tests; this takes a minute or two.", flush=True)
    command = [sys.executable, "-m", "pytest", "-q", "--color=no", path]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()

    for line in lines:
        if line.startswith(("FAILED", "ERROR")):
            print(line)

    if lines:
        print(lines[-1])

    return result.returncode
