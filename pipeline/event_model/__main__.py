"""
Train the event model, detect WAV events, or compare the four audio versions.

This file has three groups:
    * Training
    * Detecting and comparing
    * The command line
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.event_model.comparison import run_comparison
from pipeline.event_model.dataset import make_labeled_clip
from pipeline.event_model.features import add_context, frame_features
from pipeline.event_model.labels import EVENT_TYPES
from pipeline.event_model.model import EventModel, evaluate, load_model, prepare_clip, save_model
from pipeline.event_model.settings import check_settings
from pipeline.event_model.sources import load_split, make_split, recordings, save_split
from pipeline.filter_id.model import load_model as load_filter_id
from pipeline.records import load_settings, to_jsonable

# ********************************
# Training
# ********************************


# Returns after writing readable JSON with only plain Python values.
def _write_json(path, value) -> None:
    plain_values = to_jsonable(value)
    text = json.dumps(plain_values, indent=2) + "\n"

    Path(path).write_text(text, encoding="utf-8")


# Returns prepared train or tune clips, printing progress every ten clips. It
# stops with a ValueError if a background clip is invalid or silent.
def _build_clips(name, count, items, split, settings) -> list:
    clips = []

    for index in range(count):
        audio, record = make_labeled_clip(name, index, items, split, settings)
        clips.append(prepare_clip(audio, record, settings))
        del audio

        if (index + 1) % 10 == 0 or index + 1 == count:
            print(f"built {name} clips {index + 1}/{count}", flush=True)

    return clips


# Returns clip, frame and per-type yes counts for a prepared split.
def _clip_counts(clips) -> dict:
    frames = 0
    yes = np.zeros(len(EVENT_TYPES), dtype=np.int64)

    for clip in clips:
        frames += clip.n_frames
        yes += np.sum(clip.targets, axis=0, dtype=np.int64)

    yes_counts = {}

    for column, kind in enumerate(EVENT_TYPES):
        yes_counts[kind] = int(yes[column])

    return {
        "clips": len(clips),
        "frames": frames,
        "yes_frames": yes_counts,
    }


# Returns after training and saving a run, reporting only tune scores. It stops
# with a ValueError if settings, sources, or the split are invalid, or the
# output folder already contains a model.
def train(args) -> None:
    started = time.perf_counter()
    settings = load_settings(args.settings)
    check_settings(settings)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "model.joblib").exists():
        raise ValueError("out already contains model.joblib")

    times = {}

    items = recordings(settings)

    if args.split:
        split = load_split(args.split, items, settings)
    else:
        split = make_split(items, settings)

    save_split(out / "split.json", split)
    _write_json(out / "settings.json", settings)
    times["backgrounds_and_split"] = time.perf_counter() - started

    part = time.perf_counter()
    train_clips = _build_clips("train", settings["clips"]["train_clips"], items, split, settings)
    tune_clips = _build_clips("tune", settings["clips"]["tune_clips"], items, split, settings)
    times["clips_and_features"] = time.perf_counter() - part

    train_counts = _clip_counts(train_clips)
    tune_counts = _clip_counts(tune_clips)
    model = EventModel(EVENT_TYPES, settings, settings["seed"])
    part = time.perf_counter()
    model.fit(train_clips)
    times["fit"] = time.perf_counter() - part
    del train_clips

    part = time.perf_counter()
    model.tune_thresholds(tune_clips, settings["scoring"])
    tune = evaluate(model, tune_clips, settings["scoring"])
    times["tune_and_evaluate"] = time.perf_counter() - part
    del tune_clips

    save_model(out / "model.joblib", model, settings)
    times["total"] = time.perf_counter() - started
    metrics = {
        "train": train_counts,
        "tune": tune_counts,
        "thresholds": model.thresholds,
        "tune_scores": tune,
        "seconds": times,
        "note": "Tune clips share recordings with train clips. Test scores are only in the comparison.",
    }

    _write_json(out / "metrics.json", metrics)
    print(json.dumps(to_jsonable(metrics)), flush=True)


# ********************************
# Detecting and comparing
# ********************************


# Returns mono audio at the trained rate. It stops with a message naming the
# file if its channel count or sample rate does not match the model.
def _read_wav(path, sample_rate) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float64")

    if audio.ndim != 1:
        sys.exit(f"{path}: expected a mono WAV file")

    if rate != sample_rate:
        sys.exit(f"{path}: sample rate {rate}, model sample rate {sample_rate}")

    return audio


# Returns after printing one JSON line per WAV with the frozen model's event
# detections. It stops with a message for a stereo or wrong-rate file.
def detect(args) -> None:
    model, settings = load_model(args.model)

    for path in args.wavs:
        audio = _read_wav(path, settings["sample_rate"])
        features, times = frame_features(audio, settings["sample_rate"], settings["features"])
        features = add_context(features, settings["features"]["context"])

        n_frames = len(times)
        duration_s = len(audio) / settings["sample_rate"]
        events = model.detect(features, n_frames, duration_s)

        print(json.dumps({"file": str(path), "events": events}))


# Returns after writing the saved split's comparison, case records, and FLOAT
# examples. It stops with a ValueError if the saved settings differ or the split
# or inverse output is invalid.
def compare(args) -> None:
    model, settings = load_model(args.model)
    folder = Path(args.model).parent
    saved_settings = load_settings(folder / "settings.json")

    if settings != saved_settings:
        raise ValueError("settings.json differs from the model's saved settings")

    check_settings(settings)
    items = recordings(settings)
    split = load_split(folder / "split.json", items, settings)

    fid = None

    if args.filter_id_model:
        fid = load_filter_id(args.filter_id_model)

    command = None

    if args.c_command:
        command = [args.c_command]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_dir = out / "examples"
    metrics = run_comparison(
        model,
        settings,
        items,
        split,
        filter_id=fid,
        method=args.method,
        command=command,
        export_dir=export_dir,
    )

    metrics.pop("cases")
    _write_json(out / "metrics.json", metrics)
    print(json.dumps(to_jsonable(metrics)), flush=True)


# ********************************
# The command line
# ********************************


# Returns the parser for train, detect and compare with the plan's settings.
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline.event_model")
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train")
    training.add_argument("--settings", default=Path(__file__).with_name("settings.json"))
    training.add_argument("--split")
    training.add_argument("--out", required=True)
    training.set_defaults(run=train)

    detecting = commands.add_parser("detect")
    detecting.add_argument("--model", required=True)
    detecting.add_argument("wavs", nargs="+")
    detecting.set_defaults(run=detect)

    comparing = commands.add_parser("compare")
    comparing.add_argument("--model", required=True)
    comparing.add_argument("--filter-id-model")
    comparing.add_argument("--method", choices=["python", "c"], default="python")
    comparing.add_argument("--c-command")
    comparing.add_argument("--out", required=True)
    comparing.set_defaults(run=compare)

    return parser


# Returns after reading the command line and running the chosen command.
def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
