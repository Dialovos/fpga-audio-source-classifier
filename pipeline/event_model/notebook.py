"""
Run the voiceover experiment from short notebook cells.

This file has three groups:
    * Settings and training
    * Test clips and plots
    * Comparison and WAV detection
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import signal

from pipeline.event_model.__main__ import main
from pipeline.event_model.dataset import make_clip
from pipeline.event_model.labels import EVENT_TYPES
from pipeline.event_model.model import load_model, prepare_clip
from pipeline.event_model.scoring import match_events
from pipeline.event_model.settings import check_settings
from pipeline.event_model.sources import load_split, recordings
from pipeline.records import load_settings

# ********************************
# Settings and training
# ********************************


# Returns default settings with three raw/noisy recording groups. It stops with
# a FileNotFoundError listing missing WAVs and where to put them.
def voiceover_settings(source_dir="data/test-data-prelim") -> dict:
    settings = load_settings(Path(__file__).with_name("settings.json"))
    folder = Path(source_dir)
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
            main(["train", "--settings", str(settings_path), "--out", str(folder)])

        model, saved_settings = load_model(model_path)
        print("Training finished and the model was saved.")

    _print_tuning(folder)

    return model, saved_settings


# ********************************
# Test clips and plots
# ********************************


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
    figure, axes = plt.subplots(
        4, 1, figsize=(11, 6), dpi=90, sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]}, layout="constrained",
    )
    frequencies, times, power = signal.spectrogram(
        audio, fs=settings["sample_rate"], nperseg=16384, noverlap=8192,
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


# ********************************
# Comparison and WAV detection
# ********************************


# Returns the saved comparison metrics using the Python inverse. It omits AI
# recovery when the filter ID file is absent and stops on command errors.
def compare_versions(run_dir, filter_id_path=None) -> dict:
    folder = Path(run_dir)
    output = folder / "comparison"
    arguments = [
        "compare", "--model", str(folder / "model.joblib"),
        "--method", "python", "--out", str(output),
    ]

    if filter_id_path is not None and Path(filter_id_path).is_file():
        arguments.extend(["--filter-id-model", str(filter_id_path)])
        print("Filter ID model found; including AI estimate undone.")
    else:
        print("Filter ID model missing; AI estimate undone is omitted (three versions).")

    print("Running the comparison; progress is saved in comparison.log.", flush=True)

    with (folder / "comparison.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        main(arguments)

    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


# Returns an F1 chart after printing precision, recall and F1 per version/type.
# Undefined scores print as n/a and leave a gap in the chart.
def show_comparison(metrics):
    names = {
        "clean": "Clean",
        "filtered": "Filtered",
        "known_undone": "Known filter undone",
        "ai_undone": "AI estimate undone",
    }
    versions = list(metrics["versions"])
    print(f"{metrics['test_clips']} held-out clips, {metrics['case_count']} paired cases; inverse: {metrics['method']}")
    print(f"{'Version':<22} {'Type':<11} {'Precision':>9} {'Recall':>8} {'F1':>8}")

    for version in versions:
        for kind in EVENT_TYPES:
            scores = metrics["versions"][version]["scores"][kind]
            values = []

            for key in ("precision", "recall", "f1"):
                values.append(_score_text(scores[key]))

            print(f"{names[version]:<22} {kind:<11} {values[0]:>9} {values[1]:>8} {values[2]:>8}")

    figure, axis = plt.subplots(figsize=(9, 3.5), dpi=90, layout="constrained")
    positions = np.arange(len(versions))

    for column, kind in enumerate(EVENT_TYPES):
        values = []

        for version in versions:
            value = metrics["versions"][version]["scores"][kind]["f1"]

            if value is None:
                value = np.nan

            values.append(value)

        axis.bar(positions + (column - 1) * 0.24, values, width=0.24, label=kind)

    labels = []

    for version in versions:
        labels.append(names[version].replace(" undone", "\nundone"))

    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Event F1")
    axis.set_title("Same held-out cases for every audio version")
    axis.legend(loc="upper right", ncols=3, fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    return figure


# Returns the command's events for one WAV. It stops with the command's message
# if the WAV is not mono at the model's sample rate, or cannot be read.
def detect_file(run_dir, wav_path) -> list[dict]:
    output = io.StringIO()

    with redirect_stdout(output):
        main(["detect", "--model", str(Path(run_dir) / "model.joblib"), str(wav_path)])

    return json.loads(output.getvalue())["events"]


# Returns after printing each event's type, start, end and confidence, or an
# explicit message when the event list is empty.
def print_events(events) -> None:
    print(f"{len(events)} detected events")

    if not events:
        print("No events detected.")
        return

    print(f"{'Type':<12} {'Start (s)':>10} {'End (s)':>10} {'Confidence':>11}")

    for event in events:
        print(f"{event['type']:<12} {event['start_s']:>10.3f} {event['end_s']:>10.3f} {event['confidence']:>11.3f}")
