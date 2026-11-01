"""
Check the offline app (pipeline/app.py) from start to finish.

These tests are the app's spec: pipeline/app.py says what each function has to
do, and the app is done when every test here passes. Run just these with:

    python -m pytest pipeline/tests/test_app.py

They train a tiny filter ID model and a tiny event model on short stand-in
recordings once, so the whole file runs in under a minute.

This file has five groups:
    * Shared fixtures
    * Loading the models and checking the files
    * The steps
    * Writing and showing the results
    * The command line
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.app import (
    analyze_file,
    check_files,
    count_events,
    describe_filter,
    find_events,
    identify_filter,
    load_models,
    main,
    summary,
    write_outputs,
)
from pipeline.compensation import compensate
from pipeline.event_model.__main__ import main as event_model_main
from pipeline.event_model.model import EventModel
from pipeline.filter_id.dataset import build_dataset
from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import FilterIdModel
from pipeline.filter_id.model import save_model as save_filter_id
from pipeline.filters import apply_filter, design_filter
from pipeline.records import load_settings, to_jsonable

# ********************************
# Shared fixtures
# ********************************

FS = 192000
EVENT_SETTINGS = Path(__file__).parents[1] / "event_model" / "settings.json"
BANDPASS = {"type": "bandpass", "family": "butter", "order": 4, "cutoff": [250, 6000]}
THREE_TYPES = ("rumble", "hiss", "transient")


# Returns the folder after writing count stand-in recordings: noise kept to
# 150 Hz - 3 kHz like speech, in 1 s phrases with quiet pauses, at RMS 0.1.
def write_backgrounds(folder, count, seconds):
    folder.mkdir(parents=True, exist_ok=True)
    speech_band = {"type": "bandpass", "family": "butter", "order": 4, "cutoff": [150, 3000]}
    n_samples = round(seconds * FS)
    time = np.arange(n_samples) / FS

    for index in range(count):
        rng = np.random.default_rng([11, index])
        noise, _ = apply_filter(rng.standard_normal(n_samples), FS, speech_band)
        envelope = np.maximum(0.05, np.sin(2 * np.pi * 0.5 * time + index))
        audio = noise * envelope
        audio *= 0.1 / np.sqrt(np.mean(audio**2))
        sf.write(folder / f"background-{index}.wav", audio, FS, subtype="FLOAT")

    return folder.as_posix()


# Returns the paths of a tiny trained filter ID model and a tiny trained event
# model, both trained once for this file on the stand-in recordings.
@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("app-models")
    source_dir = write_backgrounds(folder / "backgrounds", 3, 3.0)

    filter_id_settings = {
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
        "features": {"n_bands": 96, "f_min_hz": 20, "f_max_hz": 24000, "segment": 32768},
    }
    data = build_dataset(filter_id_settings)
    filter_id = FilterIdModel(7).fit(data.features, data.labels)
    filter_id_path = folder / "filter-id.joblib"
    save_filter_id(filter_id_path, filter_id, filter_id_settings)

    settings = load_settings(EVENT_SETTINGS)
    settings["seed"] = 7
    settings["sources"]["source_dir"] = source_dir
    settings["sources"]["recording_groups"] = {}
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
    settings_path = folder / "event-settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    event_run = folder / "event-run"

    with redirect_stdout(io.StringIO()):
        event_model_main(["train", "--settings", str(settings_path), "--out", str(event_run)])

    return {
        "filter_id": filter_id_path,
        "event_model": event_run / "model.joblib",
        "backgrounds": Path(source_dir),
    }


# Returns both models loaded by the app, once for this file.
@pytest.fixture(scope="module")
def models(model_files):
    return load_models(model_files["filter_id"], model_files["event_model"])


# Returns two 2 s recordings with a 250 Hz - 6 kHz band-pass applied, saved as
# FLOAT WAVs named clip-a.wav and clip-b.wav.
@pytest.fixture(scope="module")
def clips(model_files, tmp_path_factory):
    folder = tmp_path_factory.mktemp("app-clips")
    paths = []

    for index, name in enumerate(("clip-a", "clip-b")):
        background, _ = sf.read(model_files["backgrounds"] / f"background-{index}.wav", dtype="float64")
        filtered, _ = apply_filter(background[: 2 * FS], FS, BANDPASS)
        path = folder / f"{name}.wav"
        sf.write(path, filtered, FS, subtype="FLOAT")
        paths.append(path)

    return paths


# Returns the path of a stand-in C program that copies in.wav to out.wav and
# exits with 1, the way the real C program does when it works.
def fake_c_program(folder):
    script = folder / "fake_filter.py"
    script.write_text(
        "import soundfile as sf\n"
        "import sys\n"
        "audio, rate = sf.read('in.wav')\n"
        "sf.write('out.wav', audio, rate, subtype='FLOAT')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    return script


# Returns a made-up report with the given number of transients, for the summary
# tests.
def made_up_report(transients):
    events = [{"type": "rumble", "start_s": 0.5, "end_s": 1.75, "confidence": 0.8}]

    for index in range(transients):
        events.append({"type": "transient", "start_s": 2.0 + index, "end_s": 2.01 + index, "confidence": 0.6})

    return {
        "file": "cockpit.wav",
        "sample_rate": FS,
        "duration_s": 42.0,
        "filter": {"spec": BANDPASS, "confidence": {"type": 0.9, "order": 0.6}},
        "compensation": {"method": "python", "spec": BANDPASS},
        "events": events,
    }


# ********************************
# Loading the models and checking the files
# ********************************


# Tests: load_models gives back both models, the settings each was trained with,
# and their shared sample rate.
# Why: every later step needs the models and the exact settings they were trained
# with, or the features won't match training.
def test_load_models_returns_both_models(models):
    assert isinstance(models["filter_id"], FilterIdModel)
    assert isinstance(models["event_model"], EventModel)
    assert models["filter_id_settings"]["features"]["n_bands"] == 96
    assert models["event_settings"]["features"]["hop"] == 3840
    assert models["sample_rate"] == FS


# Tests: the event model file given as the filter ID model stops with a
# TypeError, the filter ID file given as the event model stops with a ValueError,
# and two models trained at different sample rates stop with a ValueError.
# Why: mixing up the two files is an easy mistake, and models at different rates
# would read the same audio differently.
def test_load_models_refuses_the_wrong_files(model_files, models, tmp_path):
    with pytest.raises(TypeError):
        load_models(model_files["event_model"], model_files["event_model"])

    with pytest.raises(ValueError):
        load_models(model_files["filter_id"], model_files["filter_id"])

    settings = dict(models["filter_id_settings"])
    settings["sample_rate"] = 48000
    other_rate = tmp_path / "filter-id-48k.joblib"
    save_filter_id(other_rate, models["filter_id"], settings)

    with pytest.raises(ValueError, match="sample rate"):
        load_models(other_rate, model_files["event_model"])


# Tests: good files pass the check without an error.
# Why: the check runs before anything else, so it can't refuse good recordings.
def test_check_files_accepts_good_files(clips):
    check_files(clips, FS, 1.0)


# Tests: a file that isn't a WAV, a stereo file, a 48 kHz file, a file shorter
# than min_seconds, and two files with the same name each stop with a ValueError
# that names the bad file.
# Why: the app checks every file first, so one bad file stops it before it
# writes anything, and the message has to say which file to fix.
def test_check_files_names_the_bad_file(clips, tmp_path):
    not_wav = tmp_path / "notes.wav"
    not_wav.write_text("not audio", encoding="utf-8")

    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((FS, 2)), FS, subtype="FLOAT")

    slow = tmp_path / "slow.wav"
    sf.write(slow, np.zeros(48000), 48000, subtype="FLOAT")

    short = tmp_path / "short.wav"
    sf.write(short, np.zeros(FS // 2), FS, subtype="FLOAT")

    twin_folder = tmp_path / "twin"
    twin_folder.mkdir()
    twin = twin_folder / clips[0].name
    sf.write(twin, np.zeros(FS), FS, subtype="FLOAT")

    for bad in (not_wav, stereo, slow, short, twin):
        with pytest.raises(ValueError, match=bad.name):
            check_files([clips[0], bad], FS, 1.0)


# ********************************
# The steps
# ********************************


# Tests: the filter guess is the filter ID model's prediction on spectrum
# features made with the model's own feature settings, and its spec is one the
# filter code accepts.
# Why: features made any other way than in training would make the guess
# meaningless, and the next step builds the inverse straight from the spec.
def test_identify_filter_matches_the_model(models, clips):
    audio, _ = sf.read(clips[0], dtype="float64")
    settings = models["filter_id_settings"]
    features = spectrum_features(audio, FS, **settings["features"])
    expected = models["filter_id"].predict(features)[0]

    guess = identify_filter(audio, models)

    assert guess == expected
    assert set(guess) == {"spec", "confidence"}
    design_filter(guess["spec"], FS)


# Tests: the events the app finds are the same ones the event model's detect
# command finds in the same file.
# Why: the app has to use the frozen model exactly the way it was trained and
# scored, or its events won't match the scores we report.
def test_find_events_matches_the_detect_command(model_files, models, clips, capsys):
    audio, _ = sf.read(clips[0], dtype="float64")
    event_model_main(["detect", "--model", str(model_files["event_model"]), str(clips[0])])
    expected = json.loads(capsys.readouterr().out)["events"]

    events = find_events(audio, models)

    assert events == expected

    for event in events:
        assert set(event) == {"type", "start_s", "end_s", "confidence"}


# Tests: analyze_file guesses the filter on the recording, undoes that guess with
# the Python inverse, finds the events in the undone audio, and puts all of it in
# the report.
# Why: this is the app's whole job. Each step has to use the one before it, in
# this order, or the report describes something the app didn't do.
def test_analyze_file_runs_the_steps_in_order(models, clips):
    audio, _ = sf.read(clips[0], dtype="float64")
    guess = identify_filter(audio, models)
    undone, _ = compensate(audio, FS, guess["spec"])

    report, compensated = analyze_file(clips[0], models)

    np.testing.assert_array_equal(compensated, undone)
    assert report["file"] == "clip-a.wav"
    assert report["sample_rate"] == FS
    assert report["duration_s"] == pytest.approx(2.0)
    assert report["filter"] == guess
    assert report["compensation"]["method"] == "python"
    assert report["compensation"]["spec"] == guess["spec"]
    assert report["events"] == find_events(undone, models)


# Tests: with method "c", the app undoes the filter with the C program at the
# given command and says so in the report. A C program that fails stops with a
# ValueError.
# Why: on Linux the app has to run Adam's C program, and a broken run must not
# quietly turn into a report.
def test_analyze_file_with_the_c_program(models, clips, tmp_path):
    script = fake_c_program(tmp_path)
    command = [sys.executable, str(script)]

    report, compensated = analyze_file(clips[0], models, method="c", command=command)

    assert report["compensation"]["method"] == "c"
    assert len(compensated) == 2 * FS

    broken = tmp_path / "broken.py"
    broken.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")

    with pytest.raises(ValueError):
        analyze_file(clips[0], models, method="c", command=[sys.executable, str(broken)])


# ********************************
# Writing and showing the results
# ********************************


# Tests: the outputs are named after the input file, the compensated audio is a
# 32-bit float WAV that keeps levels past full scale, the report is JSON that
# reads back the same, and a missing output folder gets made.
# Why: undone audio can go past full scale, and clipping it would hide what the
# inverse did. The names have to match so you can tell which output is which.
def test_write_outputs_names_and_formats(tmp_path):
    report = made_up_report(1)
    loud = np.array([0.25, 1.5, -2.0, 0.0])
    out = tmp_path / "new-folder"

    wav_path, report_path = write_outputs(out, report, loud)

    assert Path(wav_path) == out / "cockpit-compensated.wav"
    assert Path(report_path) == out / "cockpit-report.json"
    assert sf.info(wav_path).subtype == "FLOAT"
    assert sf.info(wav_path).samplerate == FS

    audio, _ = sf.read(wav_path, dtype="float64")
    np.testing.assert_allclose(audio, loud)

    saved = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert saved == to_jsonable(report)


# Tests: a filter spec comes out in words, with the type, the order, and the
# cutoff or band edges rounded to whole Hz.
# Why: the summary is for people, so "band-pass, order 4, 250 Hz to 6000 Hz"
# reads better than a JSON spec.
def test_describe_filter_in_words():
    assert describe_filter(BANDPASS) == "band-pass, order 4, 250 Hz to 6000 Hz"

    high = {"type": "highpass", "family": "butter", "order": 2, "cutoff": 312.4}
    assert describe_filter(high) == "high-pass, order 2, 312 Hz"

    low = {"type": "lowpass", "family": "butter", "order": 1, "cutoff": 8000.0}
    assert describe_filter(low) == "low-pass, order 1, 8000 Hz"


# Tests: the event counts list every type the model knows, in the model's order,
# including types with no events.
# Why: the event model is going from 3 types to 7, so the app can't hard-code the
# list, and a 0 count is worth showing.
def test_count_events_in_the_models_type_order():
    events = made_up_report(2)["events"]

    assert count_events(events, THREE_TYPES) == "rumble 1, hiss 0, transient 2"

    seven = ("rumble", "hum", "whine", "hiss", "alarm", "thump", "click")
    assert count_events(events, seven) == "rumble 1, hum 0, whine 0, hiss 0, alarm 0, thump 0, click 0"


# Tests: the summary names the file and its length, the filter in words with its
# confidences, which inverse undid it, and the event counts, then lists each
# event. With more than 20 events it skips the list and says they're in the
# report.
# Why: the summary is what someone reads first after a run, and a long recording
# shouldn't flood the terminal.
def test_summary_shows_the_key_facts():
    text = summary(made_up_report(2), THREE_TYPES)

    assert "cockpit.wav" in text
    assert "42.0 s" in text
    assert "band-pass, order 4, 250 Hz to 6000 Hz" in text
    assert "type 0.90" in text
    assert "order 0.60" in text
    assert "python" in text
    assert "rumble 1, hiss 0, transient 2" in text
    assert "0.50" in text and "1.75" in text
    assert text.count("transient") == 3

    long_text = summary(made_up_report(25), THREE_TYPES)

    assert "rumble 1, hiss 0, transient 25" in long_text
    assert "report" in long_text
    assert long_text.count("transient") == 1


# ********************************
# The command line
# ********************************


# Tests: the app writes a compensated WAV and a report for every file, prints a
# summary for each one, and each saved report matches analyze_file.
# Why: this is how everyone will run the app, so the command line has to do the
# same thing the functions do.
def test_main_writes_every_file_and_prints_summaries(model_files, models, clips, tmp_path, capsys):
    out = tmp_path / "out"
    argv = [
        "--filter-id-model", str(model_files["filter_id"]),
        "--event-model", str(model_files["event_model"]),
        "--out", str(out),
        str(clips[0]), str(clips[1]),
    ]

    main(argv)
    printed = capsys.readouterr().out

    for clip in clips:
        name = clip.stem
        assert (out / f"{name}-compensated.wav").is_file()
        assert name in printed

        saved = json.loads((out / f"{name}-report.json").read_text(encoding="utf-8"))
        report, _ = analyze_file(clip, models)
        assert saved == to_jsonable(report)


# Tests: with one bad file in the list, the app stops with a message naming it,
# and writes nothing, not even for the good file before it.
# Why: a batch shouldn't end half-written, with outputs for some files and not
# others.
def test_main_checks_every_file_before_writing(model_files, clips, tmp_path):
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((2 * FS, 2)), FS, subtype="FLOAT")
    out = tmp_path / "out"
    argv = [
        "--filter-id-model", str(model_files["filter_id"]),
        "--event-model", str(model_files["event_model"]),
        "--out", str(out),
        str(clips[0]), str(stereo),
    ]

    with pytest.raises(SystemExit, match="stereo.wav"):
        main(argv)

    assert not out.exists() or not any(out.iterdir())


# Tests: --method c without --c-command stops with a message about
# --c-command, and swapped model files stop with a message instead of a
# traceback.
# Why: these are the two easiest mistakes to make on the command line, and each
# should say what to fix.
def test_main_explains_command_line_mistakes(model_files, clips, tmp_path):
    out = tmp_path / "out"
    c_without_command = [
        "--filter-id-model", str(model_files["filter_id"]),
        "--event-model", str(model_files["event_model"]),
        "--method", "c",
        "--out", str(out),
        str(clips[0]),
    ]

    with pytest.raises(SystemExit, match="--c-command"):
        main(c_without_command)

    swapped = [
        "--filter-id-model", str(model_files["event_model"]),
        "--event-model", str(model_files["filter_id"]),
        "--out", str(out),
        str(clips[0]),
    ]

    with pytest.raises(SystemExit) as stopped:
        main(swapped)

    assert isinstance(stopped.value.code, str)
