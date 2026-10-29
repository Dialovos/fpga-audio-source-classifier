"""
Checks for the filter ID model (the first AI layer).

These train on small sets of filtered noise made right here in the test, so they
run in seconds and don't need the data generator.
"""

import json

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from pipeline.filter_id import FilterIdModel, evaluate, load_model, save_model, spectrum_features, split_by_recording
from pipeline.filter_id.__main__ import main

FS = 192_000
CLIP_SECONDS = 0.5
ALL_TYPES = ["highpass", "lowpass", "bandpass"]
ORDER_4 = {"highpass": [4], "lowpass": [4], "bandpass": [4]}
CUTOFF_RANGES = {
    "highpass": (80, 1000),
    "lowpass": (2000, 16_000),
    "bandpass_low": (80, 1000),
    "bandpass_high": (2000, 16_000),
}


# Returns a number picked evenly on a log scale between low and high, so every
# octave in the range is equally likely.
def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


# Returns a clip of white noise tilted by tilt_db_per_octave around 1 kHz, so a
# weak low end isn't always a filter.
def source_clip(rng, tilt_db_per_octave):
    n_samples = int(CLIP_SECONDS * FS)
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples, 1 / FS)
    gain_db = tilt_db_per_octave * np.log2(np.clip(freqs, 20, 24_000) / 1000)
    return np.fft.irfft(spectrum * 10 ** (gain_db / 20), n_samples)


# Returns Butterworth second-order sections for a label. A band-pass is a
# high-pass at the low edge followed by a low-pass at the high edge, each with
# half the total order, the same way the C filters build it.
def butterworth(label):
    if label["type"] == "bandpass":
        low, high = label["cutoff"]
        poles_per_side = label["order"] // 2
        highpass = signal.butter(poles_per_side, low, "highpass", fs=FS, output="sos")
        lowpass = signal.butter(poles_per_side, high, "lowpass", fs=FS, output="sos")
        return np.vstack([highpass, lowpass])
    return signal.butter(label["order"], label["cutoff"], label["type"], fs=FS, output="sos")


# Builds labeled examples from filtered noise: n_sources recordings, each with
# its own tilt, and per_source examples from each. The type and order are picked
# from the given lists, and noise goes in 40 dB below the filtered clip. It
# returns the features, the labels, the records, and each row's recording ID.
def make_examples(types, orders, n_sources=60, per_source=4, seed=0):
    rng = np.random.default_rng(seed)
    features, labels, records, groups = [], [], [], []
    for source in range(n_sources):
        recording_id = f"rec-{source:03d}"
        tilt = rng.uniform(-6, 3)
        for _ in range(per_source):
            kind = str(rng.choice(types))
            order = int(rng.choice(orders[kind]))
            if kind == "bandpass":
                cutoff = [log_uniform(rng, *CUTOFF_RANGES["bandpass_low"]),
                          log_uniform(rng, *CUTOFF_RANGES["bandpass_high"])]
            else:
                cutoff = log_uniform(rng, *CUTOFF_RANGES[kind])
            label = {"type": kind, "order": order, "cutoff": cutoff}
            audio = signal.sosfilt(butterworth(label), source_clip(rng, tilt))
            audio += 0.01 * np.std(audio) * rng.standard_normal(len(audio))
            features.append(spectrum_features(audio, FS))
            labels.append(label)
            records.append({"recording_id": recording_id, "damage": [{"stage": "after_filter"}]})
            groups.append(recording_id)
    return np.array(features), labels, records, np.array(groups)


# Trains on most of the recordings and scores on the rest. It returns the
# features, the trained model, and the scores.
def train_and_score(types, orders):
    features, labels, records, groups = make_examples(types, orders)
    train, test = split_by_recording(groups, 0.25, seed=0)
    model = FilterIdModel(seed=0).fit(features[train], [labels[i] for i in train])
    metrics = evaluate(model, features[test], [labels[i] for i in test], [records[i] for i in test])
    return features, model, metrics


# Trains once on every filter type at order 4, so the type is the main thing to
# learn, and shares the result with every test that asks for it.
@pytest.fixture(scope="module")
def all_types():
    return train_and_score(ALL_TYPES, ORDER_4)


# Saves a trained model the way predict expects, with default feature settings.
def save_for_predict(model, folder):
    path = folder / "model.joblib"
    save_model(path, model, {"sample_rate": FS, "features": {}})
    return path


# Tests: the train/test split puts every recording entirely on one side.
# Why: the proposal holds out whole source recordings. If one recording showed up in both sets, the model could
#   score well by recognizing the source, and the test score would overstate how well it works on new audio.
def test_split_keeps_each_recording_on_one_side():
    groups = np.repeat([f"rec-{k:02d}" for k in range(20)], 5)
    train, test = split_by_recording(groups, 0.25, seed=0)
    assert set(groups[train]).isdisjoint(groups[test])
    assert len(train) + len(test) == len(groups)
    assert len(set(groups[test])) == 5


# Tests: trained on a small set with every type at order 4, the model gets the type right on held-out recordings
#   at least 80% of the time (chance is about 33%).
# Why: this is the most basic thing the first layer has to do. If it can't learn the type on an easy setup,
#   something in the features or the training is broken.
def test_model_learns_the_filter_type_on_new_sources(all_types):
    _, _, metrics = all_types
    assert metrics["type_accuracy"] >= 0.8


# Tests: trained on high-pass filters of order 1 and 4, the model gets the order right on held-out recordings
#   at least 75% of the time (chance is 50%).
# Why: the proposal has the AI estimate the order too, and the compensation filter needs it to undo the right
#   amount. This checks the order step learns something real, separately from the type step.
def test_model_learns_the_order_on_new_sources():
    _, _, metrics = train_and_score(["highpass"], {"highpass": [1, 4]})
    assert metrics["order_accuracy_given_true_type"] >= 0.75


# Tests: every prediction is a Butterworth spec with an order from training, cutoffs inside the training ranges
#   (a band-pass's low edge below its high edge), and confidences between 0 and 1.
# Why: the compensation filter gets built straight from this spec. A prediction it can't use, like a band-pass
#   with its edges flipped or a cutoff past what it was trained on, would crash it or build the wrong inverse.
def test_predictions_are_specs_the_compensation_filter_can_use(all_types):
    features, model, _ = all_types
    for result in model.predict(features[:30]):
        spec = result["spec"]
        assert spec["type"] in ALL_TYPES
        assert spec["family"] == "butter"
        assert spec["order"] == 4
        if spec["type"] == "bandpass":
            low, high = spec["cutoff"]
            assert CUTOFF_RANGES["bandpass_low"][0] <= low < high <= CUTOFF_RANGES["bandpass_high"][1]
        else:
            lowest, highest = CUTOFF_RANGES[spec["type"]]
            assert lowest <= spec["cutoff"] <= highest
        assert 0.0 <= result["confidence"]["type"] <= 1.0
        assert 0.0 <= result["confidence"]["order"] <= 1.0


# Tests: a saved model loads back with its settings and gives the same predictions.
# Why: the app and the compensation filter will load a model trained earlier. If saving lost anything, they'd
#   get different answers from the ones we scored.
def test_saved_model_predicts_the_same(all_types, tmp_path):
    features, model, _ = all_types
    path = save_for_predict(model, tmp_path)
    loaded, loaded_settings = load_model(path)
    assert loaded_settings == {"sample_rate": FS, "features": {}}
    assert loaded.predict(features[:20]) == model.predict(features[:20])


# Tests: predict prints one JSON line per WAV file, in order, with the file name and the predicted spec, and it
#   reads 32-bit float WAVs. A 3 kHz 4th-order low-pass comes back as a low-pass.
# Why: the app and the compensation filter read this output, and the team's files are 32-bit float. If a line
#   went missing or the format changed, they'd undo the wrong filter or none at all.
def test_predict_prints_one_json_line_per_wav(all_types, tmp_path, capsys):
    _, model, _ = all_types
    path = save_for_predict(model, tmp_path)
    label = {"type": "lowpass", "order": 4, "cutoff": 3000.0}
    audio = signal.sosfilt(butterworth(label), source_clip(np.random.default_rng(1), 0.0))
    for name in ("a.wav", "b.wav"):
        sf.write(tmp_path / name, audio.astype(np.float32), FS, subtype="FLOAT")
    main(["predict", "--model", str(path), str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["file"].endswith("a.wav")
    assert first["spec"]["type"] == "lowpass"


# Tests: predict stops with a message on a stereo WAV or on one at a different sample rate.
# Why: the features only mean something for mono audio at the rate the model was trained on. A 48 kHz or stereo
#   file would quietly give a wrong answer, so it's better to stop.
def test_predict_stops_on_stereo_or_wrong_rate(all_types, tmp_path):
    _, model, _ = all_types
    path = save_for_predict(model, tmp_path)
    sf.write(tmp_path / "stereo.wav", np.zeros((FS, 2), dtype=np.float32), FS, subtype="FLOAT")
    sf.write(tmp_path / "low-rate.wav", np.zeros(48_000, dtype=np.float32), 48_000, subtype="FLOAT")
    for name in ("stereo.wav", "low-rate.wav"):
        with pytest.raises(SystemExit):
            main(["predict", "--model", str(path), str(tmp_path / name)])
