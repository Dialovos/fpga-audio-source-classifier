"""
Check the walkthrough helpers with short sounds, tiny voiceovers, and a small
saved model.

This file has five groups:
    * Shared fixtures and settings
    * The settings file, the test sound, and the damage
    * Guessing, undoing, saving, and plotting
    * The event model
    * Running the tests
"""

from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import soundfile as sf

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pipeline import walkthrough as wt
from pipeline.compensation import compensate
from pipeline.event_model import __main__ as commands
from pipeline.event_model.labels import EVENT_TYPES
from pipeline.event_model.sources import load_split, recordings
from pipeline.filter_id.features import spectrum_features
from pipeline.filters import normalize_spec
from pipeline.records import load_settings

# ********************************
# Shared fixtures and settings
# ********************************

FS = 48000
ROOT = Path(__file__).parents[2]
SETTINGS_FILE = ROOT / "pipeline-walkthrough-settings.json"
EVENT_SETTINGS = ROOT / "pipeline" / "event_model" / "settings.json"


# Returns a fresh copy of the repo's walkthrough settings with a 0.25 s tone as
# the test sound, so each test can change it without touching the others.
def _cfg() -> dict:
    cfg = copy.deepcopy(wt.read_settings(SETTINGS_FILE))
    cfg["sound"]["kind"] = "tone"
    cfg["sound"]["seconds"] = 0.25
    cfg["sound"]["freq_hz"] = 1000

    return cfg


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
# The tiny event model settings go in their own file, like a teammate's copy.
@pytest.fixture(scope="module")
def notebook_run(voiceovers, tmp_path_factory):
    folder = tmp_path_factory.mktemp("walkthrough-run")
    settings = load_settings(EVENT_SETTINGS)
    settings["clip_seconds"] = 0.6
    settings["edge_margin_s"] = 0.025
    settings["split"]["guard_seconds"] = 0.1
    settings["split"]["tune_fraction"] = 0.5
    settings["events"]["per_clip"] = [1, 1]

    # Every type has to fit inside a 0.6 s clip, so each one gets trained.
    for kind in ("rumble", "hum", "whine", "hiss"):
        settings["events"][kind]["duration_s"] = [0.1, 0.2]
        settings["events"][kind]["fade_s"] = [0.01, 0.02]

    settings["events"]["alarm"]["duration_s"] = [0.25, 0.4]
    settings["events"]["alarm"]["beep_s"] = [0.05, 0.1]
    settings["events"]["alarm"]["gap_s"] = [0.03, 0.05]
    settings["clips"]["conditions_per_test_clip"] = 1
    settings["clips"]["export_examples"] = 0
    settings["features"]["hop"] = 960
    settings["features"]["short_window"] = 256
    settings["features"]["long_window"] = 1024
    settings["features"]["short_bands"] = 4
    settings["features"]["long_bands"] = 2
    settings["features"]["context"] = 0
    settings["model"]["max_iter"] = 2
    settings["model"]["max_leaf_nodes"] = 3
    settings["decoding"]["threshold_grid"] = [0.3, 0.5]
    settings_file = folder / "tiny-settings.json"
    settings_file.write_text(json.dumps(settings), encoding="utf-8")

    cfg = _cfg()
    cfg["event_model"]["settings_file"] = str(settings_file)
    cfg["event_model"]["voiceover_dir"] = str(voiceovers)
    cfg["event_model"]["seed"] = 7
    cfg["event_model"]["train_clips"] = 6
    cfg["event_model"]["tune_clips"] = 2
    run_dir = folder / "run"
    model, settings = wt.train_or_load(wt.event_model_settings(cfg), run_dir)

    return {"folder": run_dir, "model": model, "settings": settings, "cfg": cfg}


# ********************************
# The settings file, the test sound, and the damage
# ********************************


# Tests: the repo's settings file loads, its filter is valid, and its event
# model settings file exists.
# Why: a typo in the shipped settings would break the notebook's first cell.
def test_repo_settings_file_loads():
    cfg = wt.read_settings(SETTINGS_FILE)

    normalize_spec(cfg["filter"], cfg["sample_rate"])
    assert (ROOT / cfg["event_model"]["settings_file"]).is_file()
    assert cfg["undo"]["method"] in ("python", "c")

    paths = [cfg["sound"]["wav"], cfg["out_dir"], cfg["filter_id_model"], cfg["event_model"]["run_dir"]]

    for path in paths:
        assert not Path(path).is_absolute()


# Tests: an unknown sound kind or noise stage stops with the setting's name.
# Why: a misspelled choice should point at the line to fix, not fail later.
def test_read_settings_rejects_bad_choices(tmp_path):
    for section, key, value, message in [
        ("sound", "kind", "song", "sound.kind"),
        ("noise", "stage", "middle", "noise.stage"),
    ]:
        cfg = _cfg()
        cfg[section][key] = value
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            wt.read_settings(path)


# Tests: a WAV piece starts at start_s, has the right length, and comes out at
# the filter ID training level. Bad files stop with a clear message.
# Why: the filter ID model expects 1 s clips at RMS 0.1, and a wrong file
# should not be silently cut or resampled.
def test_make_sound_reads_wav_piece(tmp_path, capsys):
    ramp = np.linspace(0.01, 0.5, FS)
    path = tmp_path / "ramp.wav"
    sf.write(path, ramp, FS, subtype="FLOAT")
    cfg = _cfg()
    cfg["sound"]["kind"] = "wav"
    cfg["sound"]["wav"] = str(path)
    cfg["sound"]["start_s"] = 0.5
    cfg["sound"]["seconds"] = 0.25

    audio = wt.make_sound(cfg)
    piece = ramp[FS // 2:FS // 2 + FS // 4]

    assert len(audio) == FS // 4
    assert np.sqrt(np.mean(audio**2)) == pytest.approx(0.1)
    np.testing.assert_allclose(audio / audio[0], piece / piece[0], rtol=1e-6)
    assert "ramp.wav from 0.5 s" in capsys.readouterr().out

    cfg["sound"]["start_s"] = 0.9

    with pytest.raises(ValueError, match="too short"):
        wt.make_sound(cfg)

    cfg["sound"]["wav"] = str(tmp_path / "missing.wav")

    with pytest.raises(FileNotFoundError, match="missing.wav"):
        wt.make_sound(cfg)

    sf.write(tmp_path / "wrong-rate.wav", ramp, 192000, subtype="FLOAT")
    cfg["sound"]["wav"] = str(tmp_path / "wrong-rate.wav")
    cfg["sound"]["start_s"] = 0.0

    with pytest.raises(ValueError, match="192000 Hz"):
        wt.make_sound(cfg)


# Tests: the tone, thump, and noise sounds have the right length and level, and
# the noise repeats with the same seed.
# Why: these are the quick checks that don't need the team's recordings.
def test_make_sound_kinds():
    for kind in ("tone", "thump", "noise"):
        cfg = _cfg()
        cfg["sound"]["kind"] = kind
        audio = wt.make_sound(cfg)

        assert len(audio) == FS // 4
        assert np.sqrt(np.mean(audio**2)) == pytest.approx(0.1)

    np.testing.assert_array_equal(wt.make_sound(cfg), wt.make_sound(cfg))


# Tests: the damage runs filter, noise, and rounding in the right order, the
# SNR is measured against the clean sound, and a null SNR adds no noise.
# Why: the walkthrough should damage a sound the same way the plan describes,
# so what you hear matches what the models are tested on.
def test_damage_steps_and_snr():
    cfg = _cfg()
    clean = wt.make_sound(cfg)
    clean_power = np.mean(clean**2)

    damaged, steps = wt.damage(clean, cfg)
    again, _ = wt.damage(clean, cfg)

    np.testing.assert_array_equal(damaged, again)
    assert steps[0]["type"] == "bandpass"
    assert steps[1]["stage"] == "after_filter"
    assert steps[1]["noise_rms"] == pytest.approx(np.sqrt(clean_power / 10**3))
    assert steps[2]["bits"] == 24

    cfg["noise"]["stage"] = "before_filter"
    _, steps = wt.damage(clean, cfg)
    assert steps[0]["stage"] == "before_filter"
    assert steps[1]["type"] == "bandpass"

    cfg["noise"]["snr_db"] = None
    _, steps = wt.damage(clean, cfg)
    assert len(steps) == 2


# ********************************
# Guessing, undoing, saving, and plotting
# ********************************


# Tests: a missing filter ID model gives no guess and says which setting to fix.
# Why: teammates who haven't trained one should still get through the notebook.
def test_guess_filter_without_model(tmp_path, capsys):
    cfg = _cfg()
    cfg["filter_id_model"] = "results/none/model.joblib"

    assert wt.guess_filter(np.zeros(1000), cfg) is None
    assert "results/none/model.joblib" in capsys.readouterr().out


# Tests: the guess uses the model's own feature settings, prints the real and
# guessed filters, and refuses a model trained at another rate.
# Why: features made differently from training would make the guess meaningless.
def test_guess_filter_uses_model_features(tmp_path, monkeypatch, capsys):
    cfg = _cfg()
    path = tmp_path / "model.joblib"
    path.touch()
    cfg["filter_id_model"] = str(path)
    damaged, _ = wt.damage(wt.make_sound(cfg), cfg)
    features = {"n_bands": 24, "f_min_hz": 20, "f_max_hz": 24000, "segment": 8192}
    spec = {"type": "highpass", "family": "butter", "order": 2, "cutoff": 300.0}
    seen = []

    def predict(rows):
        seen.append(rows)
        return [{"spec": spec, "confidence": {"type": 0.9, "order": 0.6}}]

    model = SimpleNamespace(predict=predict)
    settings = {"sample_rate": FS, "features": features}
    monkeypatch.setattr(wt, "load_filter_id", lambda path: (model, settings))

    guess = wt.guess_filter(damaged, cfg)
    output = capsys.readouterr().out

    assert guess["spec"] == spec
    np.testing.assert_array_equal(seen[0], spectrum_features(damaged, FS, **features))
    assert "bandpass order 4, 250-6000 Hz" in output
    assert "highpass order 2, 300 Hz" in output
    assert "type 0.90, order 0.60" in output

    settings["sample_rate"] = 192000

    with pytest.raises(ValueError, match="192000"):
        wt.guess_filter(damaged, cfg)


# Tests: undoing gives the known version the same way the comparison does, adds
# the guess's version, and leaves out a guess the inverse can't use.
# Why: the walkthrough should show the real recovery, and a bad guess should
# not stop the notebook.
def test_undo_filter_versions(capsys):
    cfg = _cfg()
    clean = wt.make_sound(cfg)
    damaged, _ = wt.damage(clean, cfg)

    versions = wt.undo_filter(clean, damaged, None, cfg)
    known, _ = compensate(damaged, FS, cfg["filter"])

    assert list(versions) == ["clean", "damaged", "known_undone"]
    np.testing.assert_array_equal(versions["known_undone"], known)

    guess = {"spec": {"type": "highpass", "family": "butter", "order": 2, "cutoff": 300.0}}
    versions = wt.undo_filter(clean, damaged, guess, cfg)
    assert "ai_undone" in versions

    capsys.readouterr()
    bad = {"spec": {"type": "lowpass", "family": "butter", "order": 2, "cutoff": 500000.0}}
    versions = wt.undo_filter(clean, damaged, bad, cfg)
    assert "ai_undone" not in versions
    assert "can't be undone" in capsys.readouterr().out


# Tests: each version saves as a 32-bit float WAV that keeps levels past full
# scale, and the printed folder stays relative.
# Why: the undone audio can go past 1.0, and clipping it would hide what the
# inverse really did.
def test_save_wavs_keeps_float_levels(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg()
    cfg["out_dir"] = "out"
    loud = np.array([0.5, 1.5, -2.0])
    paths = wt.save_wavs({"clean": loud, "known_undone": loud * 2}, cfg)

    assert paths == ["out/clean.wav", "out/known_undone.wav"]
    audio, rate = sf.read(paths[1], dtype="float64")
    np.testing.assert_allclose(audio, loud * 2)
    assert rate == FS
    assert sf.info(paths[0]).subtype == "FLOAT"
    assert str(tmp_path) not in capsys.readouterr().out


# Tests: the filter plot draws the real filter and the guess, skipping a guess
# that can't be designed, and the spectrum plot draws one line per version.
# Why: the plots are how you see whether the guess and the undoing worked.
def test_sound_plots(capsys):
    cfg = _cfg()
    guess = {"spec": {"type": "highpass", "family": "butter", "order": 2, "cutoff": 300.0}}
    bad = {"spec": {"type": "lowpass", "family": "butter", "order": 2, "cutoff": 500000.0}}
    clean = wt.make_sound(cfg)
    damaged, _ = wt.damage(clean, cfg)
    figures = [
        wt.plot_filters(cfg, guess),
        wt.plot_filters(cfg, bad),
        wt.plot_spectra({"clean": clean, "damaged": damaged}, cfg),
    ]

    try:
        assert len(figures[0].axes[0].get_lines()) == 3
        assert len(figures[1].axes[0].get_lines()) == 2
        assert "Can't draw the filter id guess" in capsys.readouterr().out
        assert len(figures[2].axes[0].get_lines()) == 2
    finally:
        for figure in figures:
            plt.close(figure)


# ********************************
# The event model
# ********************************


# Tests: the event model settings keep the file's values, take the seed and clip
# counts from the walkthrough settings, and make each raw/noisy pair one
# recording.
# Why: noisy copies must stay out of the backgrounds and held-out split.
def test_event_model_settings_group_raw_and_noisy(voiceovers, monkeypatch):
    monkeypatch.chdir(voiceovers.parent)
    cfg = _cfg()
    cfg["event_model"]["settings_file"] = str(EVENT_SETTINGS)
    cfg["event_model"]["voiceover_dir"] = voiceovers.name
    cfg["event_model"]["seed"] = 11
    cfg["event_model"]["train_clips"] = 5
    cfg["event_model"]["tune_clips"] = 3

    settings = wt.event_model_settings(cfg)
    defaults = load_settings(EVENT_SETTINGS)

    assert settings["sources"]["source_dir"] == voiceovers.name
    assert settings["seed"] == 11
    assert settings["clips"]["train_clips"] == 5
    assert settings["clips"]["tune_clips"] == 3

    for key in defaults:
        if key not in ("sources", "seed", "clips"):
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
    cfg = _cfg()
    cfg["event_model"]["settings_file"] = str(EVENT_SETTINGS)

    with pytest.raises(FileNotFoundError) as error:
        wt.event_model_settings(cfg)

    message = str(error.value)
    assert "data/test-data-prelim-48k/" in message
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
    monkeypatch.setattr(wt, "event_model_main", lambda argv: pytest.fail("Loading tried to run a command"))
    model, settings = wt.train_or_load(notebook_run["settings"], folder)
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
        wt.train_or_load(settings, folder)

    assert (folder / "settings.json").read_bytes() == before


# Tests: the same held-out clip and detections rebuild from the saved split.
# Why: the notebook example must not sample a training recording or retune.
def test_test_clip_uses_saved_split_and_repeats(notebook_run):
    model = notebook_run["model"]
    settings = notebook_run["settings"]
    folder = notebook_run["folder"]
    audio, record, detected = wt.make_test_clip(model, settings, folder)
    again, repeated, repeated_detections = wt.make_test_clip(model, settings, folder)
    split = load_split(folder / "split.json", recordings(settings), settings)

    np.testing.assert_array_equal(audio, again)
    assert record == repeated
    assert detected == repeated_detections
    assert record["sources"][0]["recording_id"] in split["test"]
    assert record["clip"]["split"] == "test"

    with pytest.raises(ValueError, match="index"):
        wt.make_test_clip(model, settings, folder, index=99)


# Tests: the plot shows per-type hit, miss and false-alarm counts and low bands.
# Why: the figure must make scoring errors visible at a small notebook size.
def test_clip_plot_marks_hits_misses_and_false_alarms(notebook_run):
    settings = notebook_run["settings"]
    audio, record, _ = wt.make_test_clip(notebook_run["model"], settings, notebook_run["folder"])
    record["events"] = [
        {"type": "rumble", "start_s": 0.1, "end_s": 0.2},
        {"type": "hiss", "start_s": 0.2, "end_s": 0.3},
        {"type": "click", "start_s": 0.4, "end_s": 0.401},
    ]
    detected = [
        {"type": "rumble", "start_s": 0.1, "end_s": 0.2},
        {"type": "click", "start_s": 0.52, "end_s": 0.53},
    ]
    figure = wt.plot_test_clip(audio, record, detected, settings)

    # The spectrogram is the first axis, then one track per type in order.
    rumble = figure.axes[1 + EVENT_TYPES.index("rumble")]
    hiss = figure.axes[1 + EVENT_TYPES.index("hiss")]
    click = figure.axes[1 + EVENT_TYPES.index("click")]
    hum = figure.axes[1 + EVENT_TYPES.index("hum")]

    try:
        assert len(figure.axes) == 2 + len(EVENT_TYPES)
        assert figure.axes[0].get_yscale() == "log"
        assert figure.axes[0].get_ylim() == (20, 20000)
        assert rumble.get_title(loc="left") == "1 hits, 0 misses, 0 false alarms"
        assert hiss.get_title(loc="left") == "0 hits, 1 misses, 0 false alarms"
        assert click.get_title(loc="left") == "0 hits, 1 misses, 1 false alarms"
        assert hum.get_title(loc="left") == "0 hits, 0 misses, 0 false alarms"
        assert click.lines[0].get_color() == "darkorange"
        assert click.lines[1].get_color() == "crimson"
        output = io.BytesIO()
        figure.savefig(output, format="png")
        assert output.tell() < 300000
    finally:
        plt.close(figure)


# Tests: a missing filter ID model runs the real three-version Python comparison.
# Why: teammates should still obtain honest scores without the first AI layer.
def test_comparison_without_filter_id(notebook_run, capsys):
    folder = notebook_run["folder"]
    metrics = wt.compare_versions(folder, folder / "missing.joblib")
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
    features = {"n_bands": 96, "f_min_hz": 20, "f_max_hz": 24000, "segment": 8192}
    filter_settings = {"sample_rate": FS, "features": features}
    spec = {"type": "highpass", "family": "butter", "order": 1, "cutoff": 100}
    filter_model = SimpleNamespace(predict=lambda rows: [{"spec": spec}])
    monkeypatch.setattr(commands, "load_filter_id", lambda path: (filter_model, filter_settings))
    metrics = wt.compare_versions(folder, filter_id_path)

    assert "including AI estimate undone" in capsys.readouterr().out
    assert metrics["ai_undone_included"] is True
    assert metrics["versions"]["ai_undone"]["case_count"] == metrics["case_count"]
    assert metrics["filter_id"]["type_accuracy"]["cases"] == metrics["case_count"]


# Tests: the C method passes its method and program path on to the command.
# Why: undo.method in the settings has to reach the comparison, or switching to
# the C inverse would quietly keep using Python.
def test_comparison_passes_the_c_method(tmp_path, monkeypatch):
    seen = []

    def fake_main(argv):
        seen.append(argv)
        output = Path(argv[argv.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "metrics.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(wt, "event_model_main", fake_main)
    wt.compare_versions(tmp_path, None, "c", "pipeline/reversable_filters/filter")
    argv = seen[0]

    assert argv[argv.index("--method") + 1] == "c"
    assert argv[argv.index("--c-command") + 1] == "pipeline/reversable_filters/filter"


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
    figure = wt.show_comparison(metrics)

    try:
        output = capsys.readouterr().out
        assert "Precision" in output and "Recall" in output and "F1" in output
        assert "AI estimate undone" in output
        assert "n/a" in output
        assert len(figure.axes[0].patches) == 4 * len(EVENT_TYPES)
        assert np.isnan(figure.axes[0].patches[0].get_height())
    finally:
        plt.close(figure)


# Tests: WAV detection agrees with the command and prints all four event fields.
# Why: notebook detection must use the same frozen model and decoding path.
def test_detect_file_and_event_list(notebook_run, voiceovers, capsys):
    folder = notebook_run["folder"]
    path = voiceovers / "VO1.wav"
    detected = wt.detect_file(folder, path)
    capsys.readouterr()
    commands.main(["detect", "--model", str(folder / "model.joblib"), str(path)])
    expected = json.loads(capsys.readouterr().out)["events"]
    assert detected == expected

    wt.print_events([{"type": "hiss", "start_s": 1.25, "end_s": 2.5, "confidence": 0.875}])
    output = capsys.readouterr().out
    assert "hiss" in output and "1.250" in output and "2.500" in output and "0.875" in output
    assert "Confidence" in output
    assert str(folder) not in output
    wt.print_events([])
    assert "No events detected" in capsys.readouterr().out

    many = []

    for index in range(25):
        many.append({"type": "click", "start_s": index, "end_s": index + 0.02, "confidence": 0.5})

    wt.print_events(many)
    output = capsys.readouterr().out
    assert "click: 25" in output
    assert "19.000" in output and "20.000" not in output
    assert "... and 5 more" in output


# Tests: stereo and wrong-rate WAVs retain the command's clear rejection.
# Why: the notebook must not silently resample or mix a teammate's recording.
@pytest.mark.parametrize("channels, rate, message", [(2, FS, "mono"), (1, 192000, "sample rate")])
def test_detect_file_rejects_bad_wavs(notebook_run, tmp_path, channels, rate, message):
    audio = np.ones((1000, channels))

    if channels == 1:
        audio = audio[:, 0]

    path = tmp_path / "bad.wav"
    sf.write(path, audio, rate, subtype="FLOAT")

    with pytest.raises(SystemExit, match=message):
        wt.detect_file(notebook_run["folder"], path)


# ********************************
# Running the tests
# ********************************


# Tests: run_tests runs pytest with this Python on the given path, prints the
# failed tests and the last line, and returns pytest's exit code.
# Why: the notebook's last cell should show at a glance what failed.
def test_run_tests_reports_failures(monkeypatch, capsys):
    seen = []
    stdout = "..F\nFAILED pipeline/tests/test_x.py::test_one - boom\n1 failed, 2 passed in 0.10s\n"

    def fake_run(command, check, capture_output, text):
        seen.append(command)
        return SimpleNamespace(stdout=stdout, returncode=1)

    monkeypatch.setattr(wt.subprocess, "run", fake_run)
    code = wt.run_tests("pipeline/tests/test_x.py")
    output = capsys.readouterr().out

    assert code == 1
    assert seen[0] == [sys.executable, "-m", "pytest", "-q", "--color=no", "pipeline/tests/test_x.py"]
    assert "FAILED pipeline/tests/test_x.py::test_one - boom" in output
    assert "1 failed, 2 passed" in output
    assert "..F" not in output
