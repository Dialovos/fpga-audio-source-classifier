"""
Check the notebook helpers with tiny voiceovers and a small saved model.

This file has three groups:
    * Shared fixture and settings
    * Settings, training and test clips
    * Comparison and WAV detection
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import soundfile as sf

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pipeline.event_model import __main__ as commands
from pipeline.event_model import notebook as em
from pipeline.event_model.labels import EVENT_TYPES
from pipeline.event_model.sources import load_split, recordings
from pipeline.records import load_settings

# ********************************
# Shared fixture and settings
# ********************************

FS = 192000


# Returns six short FLOAT WAV paths with distinct raw/noisy levels.
@pytest.fixture(scope="module")
def voiceovers(tmp_path_factory):
    folder = tmp_path_factory.mktemp("voiceovers")
    times = np.arange(2 * FS) / FS

    for index, name in enumerate(("VO1", "VO2", "VO3")):
        raw = 0.1 * np.sin(2 * np.pi * (200 + index * 100) * times)
        sf.write(folder / f"{name}.wav", raw, FS, subtype="FLOAT")
        sf.write(folder / f"{name}_noise.wav", raw + 0.02, FS, subtype="FLOAT")

    return folder


# Returns a tiny trained run once per module, keeping the real command paths.
@pytest.fixture(scope="module")
def notebook_run(voiceovers, tmp_path_factory):
    settings = em.voiceover_settings(voiceovers)
    settings["seed"] = 7
    settings["clip_seconds"] = 0.6
    settings["edge_margin_s"] = 0.025
    settings["split"]["guard_seconds"] = 0.1
    settings["split"]["tune_fraction"] = 0.5
    settings["events"]["per_clip"] = [1, 1]

    for kind in ("rumble", "hiss"):
        settings["events"][kind]["duration_s"] = [0.1, 0.2]
        settings["events"][kind]["fade_s"] = [0.01, 0.02]

    settings["clips"]["train_clips"] = 6
    settings["clips"]["tune_clips"] = 2
    settings["clips"]["conditions_per_test_clip"] = 1
    settings["clips"]["export_examples"] = 0
    settings["features"]["hop"] = 3840
    settings["features"]["short_window"] = 1024
    settings["features"]["long_window"] = 4096
    settings["features"]["short_bands"] = 4
    settings["features"]["long_bands"] = 2
    settings["features"]["context"] = 0
    settings["model"]["max_iter"] = 2
    settings["model"]["max_leaf_nodes"] = 3
    settings["decoding"]["threshold_grid"] = [0.3, 0.5]
    folder = tmp_path_factory.mktemp("notebook-run")
    model, settings = em.train_or_load(settings, folder)

    return {"folder": folder, "model": model, "settings": settings}


# ********************************
# Settings, training and test clips
# ********************************


# Tests: defaults are retained and each raw/noisy pair becomes one recording.
# Why: noisy copies must stay out of the backgrounds and held-out split.
def test_voiceover_settings_group_raw_and_noisy(voiceovers, monkeypatch):
    monkeypatch.chdir(voiceovers.parent)
    settings = em.voiceover_settings(voiceovers.name)
    defaults = load_settings(Path(em.__file__).with_name("settings.json"))
    assert settings["sources"]["source_dir"] == voiceovers.name

    for key in defaults:
        if key != "sources":
            assert settings[key] == defaults[key]

    settings["clip_seconds"] = 0.6
    items = recordings(settings)
    assert len(items) == 3

    for item in items:
        name = item["recording_id"]
        assert Path(item["background_path"]).name == f"{name}.wav"
        assert item["copies"] == [f"{name}_noise.wav"]


# Tests: missing recordings give filenames and the expected relative folder.
# Why: teammates do not receive the gitignored audio when cloning the repo.
def test_missing_voiceovers_explain_where_to_put_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as error:
        em.voiceover_settings()

    message = str(error.value)
    assert "data/test-data-prelim/" in message
    assert "VO1.wav" in message
    assert "VO3_noise.wav" in message
    assert str(tmp_path) not in message


# Tests: training saves its artifacts and loading never calls the train command.
# Why: rerunning the notebook must preserve the expensive fitted model.
def test_train_or_load_reuses_saved_model(notebook_run, monkeypatch, capsys):
    folder = notebook_run["folder"]
    model_bytes = (folder / "model.joblib").read_bytes()
    assert (folder / "split.json").is_file()
    assert "trained rumble" in (folder / "training.log").read_text(encoding="utf-8")
    monkeypatch.setattr(em, "main", lambda argv: pytest.fail("Loading tried to run a command"))
    model, settings = em.train_or_load(notebook_run["settings"], folder)
    output = capsys.readouterr().out

    assert "Loaded" in output
    assert "Threshold" in output and "Precision" in output
    assert "built train clips" not in output
    assert str(folder) not in output
    assert settings == notebook_run["settings"]
    assert model.thresholds == notebook_run["model"].thresholds
    assert (folder / "model.joblib").read_bytes() == model_bytes


# Tests: changed settings stop before overwriting an existing run.
# Why: a loaded model must not silently be reported as a different experiment.
def test_saved_run_rejects_changed_settings(notebook_run):
    folder = notebook_run["folder"]
    before = (folder / "settings.json").read_bytes()
    settings = copy.deepcopy(notebook_run["settings"])
    settings["seed"] += 1

    with pytest.raises(ValueError, match="Settings differ"):
        em.train_or_load(settings, folder)

    assert (folder / "settings.json").read_bytes() == before


# Tests: the same held-out clip and detections rebuild from the saved split.
# Why: the notebook example must not sample a training recording or retune.
def test_test_clip_uses_saved_split_and_repeats(notebook_run):
    model = notebook_run["model"]
    settings = notebook_run["settings"]
    folder = notebook_run["folder"]
    audio, record, detected = em.make_test_clip(model, settings, folder)
    again, repeated, repeated_detections = em.make_test_clip(model, settings, folder)
    split = load_split(folder / "split.json", recordings(settings), settings)

    np.testing.assert_array_equal(audio, again)
    assert record == repeated
    assert detected == repeated_detections
    assert record["sources"][0]["recording_id"] in split["test"]
    assert record["clip"]["split"] == "test"

    with pytest.raises(ValueError, match="index"):
        em.make_test_clip(model, settings, folder, index=99)


# Tests: the plot shows per-type hit, miss and false-alarm counts and low bands.
# Why: the figure must make scoring errors visible at a small notebook size.
def test_clip_plot_marks_hits_misses_and_false_alarms(notebook_run):
    settings = notebook_run["settings"]
    audio, record, _ = em.make_test_clip(notebook_run["model"], settings, notebook_run["folder"])
    record["events"] = [
        {"type": "rumble", "start_s": 0.1, "end_s": 0.2},
        {"type": "hiss", "start_s": 0.2, "end_s": 0.3},
        {"type": "transient", "start_s": 0.4, "end_s": 0.401},
    ]
    detected = [
        {"type": "rumble", "start_s": 0.1, "end_s": 0.2},
        {"type": "transient", "start_s": 0.52, "end_s": 0.53},
    ]
    figure = em.plot_test_clip(audio, record, detected, settings)

    try:
        assert figure.axes[0].get_yscale() == "log"
        assert figure.axes[0].get_ylim() == (20, 20000)
        assert figure.axes[1].get_title(loc="left") == "1 hits, 0 misses, 0 false alarms"
        assert figure.axes[2].get_title(loc="left") == "0 hits, 1 misses, 0 false alarms"
        assert figure.axes[3].get_title(loc="left") == "0 hits, 1 misses, 1 false alarms"
        assert figure.axes[3].lines[0].get_color() == "darkorange"
        assert figure.axes[3].lines[1].get_color() == "crimson"
        output = io.BytesIO()
        figure.savefig(output, format="png")
        assert output.tell() < 300000
    finally:
        plt.close(figure)


# ********************************
# Comparison and WAV detection
# ********************************


# Tests: a missing filter ID model runs the real three-version Python comparison.
# Why: teammates should still obtain honest scores without the first AI layer.
def test_comparison_without_filter_id(notebook_run, capsys):
    folder = notebook_run["folder"]
    metrics = em.compare_versions(folder, folder / "missing.joblib")
    output = capsys.readouterr().out

    assert "AI estimate undone is omitted" in output
    assert "compared test clip" not in output
    assert metrics["method"] == "python"
    assert metrics["test_clips"] == 3
    assert metrics["case_count"] == 6
    assert metrics["ai_undone_included"] is False
    assert set(metrics["versions"]) == {"clean", "filtered", "known_undone"}
    assert (folder / "comparison" / "cases.jsonl").is_file()

    for version in metrics["versions"].values():
        assert version["case_count"] == 6
        assert set(EVENT_TYPES).issubset(version["scores"])


# Tests: an existing filter ID file is loaded and adds AI recovery on all cases.
# Why: the optional fourth version must follow the same command and case path.
def test_comparison_with_filter_id(notebook_run, monkeypatch, capsys):
    folder = notebook_run["folder"]
    filter_id_path = folder / "filter-id.joblib"
    filter_id_path.touch()
    settings_path = Path(em.__file__).parents[1] / "filter_id" / "settings.json"
    filter_settings = load_settings(settings_path)
    spec = {"type": "highpass", "family": "butter", "order": 1, "cutoff": 100}
    filter_model = SimpleNamespace(predict=lambda rows: [{"spec": spec}])
    monkeypatch.setattr(commands, "load_filter_id", lambda path: (filter_model, filter_settings))
    metrics = em.compare_versions(folder, filter_id_path)

    assert "including AI estimate undone" in capsys.readouterr().out
    assert metrics["ai_undone_included"] is True
    assert metrics["versions"]["ai_undone"]["case_count"] == metrics["case_count"]
    assert metrics["filter_id"]["type_accuracy"]["cases"] == metrics["case_count"]


# Tests: the printed table and chart retain all versions and undefined scores.
# Why: an undefined F1 must not look like a measured zero.
def test_comparison_display_keeps_undefined_scores(capsys):
    versions = {}

    for version in ("clean", "filtered", "known_undone", "ai_undone"):
        scores = {}

        for kind in EVENT_TYPES:
            scores[kind] = {"precision": 0.5, "recall": 0.75, "f1": 0.6}

        versions[version] = {"scores": scores}

    versions["clean"]["scores"]["rumble"]["f1"] = None
    metrics = {"test_clips": 3, "case_count": 6, "method": "python", "versions": versions}
    figure = em.show_comparison(metrics)

    try:
        output = capsys.readouterr().out
        assert "Precision" in output and "Recall" in output and "F1" in output
        assert "AI estimate undone" in output
        assert "n/a" in output
        assert len(figure.axes[0].patches) == 12
        assert np.isnan(figure.axes[0].patches[0].get_height())
    finally:
        plt.close(figure)


# Tests: WAV detection agrees with the command and prints all four event fields.
# Why: notebook detection must use the same frozen model and decoding path.
def test_detect_file_and_event_list(notebook_run, voiceovers, capsys):
    folder = notebook_run["folder"]
    path = voiceovers / "VO1.wav"
    detected = em.detect_file(folder, path)
    capsys.readouterr()
    commands.main(["detect", "--model", str(folder / "model.joblib"), str(path)])
    expected = json.loads(capsys.readouterr().out)["events"]
    assert detected == expected

    em.print_events([{"type": "hiss", "start_s": 1.25, "end_s": 2.5, "confidence": 0.875}])
    output = capsys.readouterr().out
    assert "hiss" in output and "1.250" in output and "2.500" in output and "0.875" in output
    assert "Confidence" in output
    assert str(folder) not in output
    em.print_events([])
    assert "No events detected" in capsys.readouterr().out


# Tests: stereo and wrong-rate WAVs retain the command's clear rejection.
# Why: the notebook must not silently resample or mix a teammate's recording.
@pytest.mark.parametrize("channels, rate, message", [(2, FS, "mono"), (1, 48000, "sample rate")])
def test_detect_file_rejects_bad_wavs(notebook_run, tmp_path, channels, rate, message):
    audio = np.ones((1000, channels))

    if channels == 1:
        audio = audio[:, 0]

    path = tmp_path / "bad.wav"
    sf.write(path, audio, rate, subtype="FLOAT")

    with pytest.raises(SystemExit, match=message):
        em.detect_file(notebook_run["folder"], path)
