"""Checks for settings files, seeds, and example records."""

import json

import numpy as np
import pytest

from pipeline.records import SCHEMA_VERSION, build_record, example_seed, load_settings, read_record, write_record


def sample_record(**overrides):
    fields = {
        "example_id": "ex-000001",
        "seed": 123,
        "sample_rate": 192_000,
        "sources": [{"recording_id": "rec-03", "role": "event", "level_db": -6.0}],
        "filter_record": {"type": "highpass", "family": "butter", "order": 4, "cutoff": 250.0},
        "damage": [
            {"noise": "white gaussian", "stage": "before_filter", "snr_db": 30.0},
            {"bits": 24, "stage": "after_filter", "clipped_samples": 0},
        ],
    }
    fields.update(overrides)
    return build_record(**fields)


# Tests: a settings file with an integer seed loads, and one with no seed, a negative seed, a float seed,
#   True, or the string "42" gets refused.
# Why: the seed is what makes a whole run repeatable, and the proposal's contribution is a reproducible
#   comparison. A missing or odd seed would give us a dataset nobody can rebuild, so we'd rather stop before
#   generating anything.
def test_settings_load_and_need_a_seed(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"seed": 42, "sample_rate": 192000}), encoding="utf-8")
    assert load_settings(good)["seed"] == 42

    for bad_seed in (None, -1, 1.5, True, "42"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"seed": bad_seed}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_settings(bad)


# Tests: an example's seed is the same every time for the same run seed and example number, 1000 examples get
#   1000 different seeds, a different run seed changes it, and a negative run seed gets refused.
# Why: each example needs its own random stream that we can recreate from just the run seed and its number.
#   If two examples shared a seed, they'd get the exact same noise, which can leak between the training and
#   test sets.
def test_example_seeds_repeat_and_differ_by_index():
    assert example_seed(42, 0) == example_seed(42, 0)
    seeds = {example_seed(42, i) for i in range(1000)}
    assert len(seeds) == 1000
    assert example_seed(42, 5) != example_seed(43, 5)
    with pytest.raises(ValueError):
        example_seed(-1, 0)


# Tests: a record has exactly the fields we expect, carries the schema version, and turns numpy numbers into
#   plain int and float so json.dumps works.
# Why: numpy integers make json.dumps crash, and that kind of crash tends to show up halfway through a long
#   generation run. A fixed set of fields means every record can be read the same way later.
def test_record_has_every_field_in_plain_python():
    record = sample_record(seed=np.int64(123), sample_rate=np.float64(192_000))
    assert record["schema_version"] == SCHEMA_VERSION
    assert set(record) == {"schema_version", "example_id", "seed", "sample_rate", "sources", "filter", "damage"}
    assert type(record["seed"]) is int
    assert type(record["sample_rate"]) is float
    json.dumps(record)  # must not raise


# Tests: numpy arrays (like the filter's coefficients) and tuples in a record come out as plain lists.
# Why: the coefficients come out of SciPy as a numpy array, and they're the most important thing we save,
#   since they're the reference for rebuilding the filter and for compensating with the known filter.
def test_record_turns_numpy_arrays_into_lists():
    record = sample_record(filter_record={"sos": np.ones((2, 6)), "cutoff": (250, 6000)})
    assert record["filter"]["sos"] == [[1.0] * 6, [1.0] * 6]
    assert record["filter"]["cutoff"] == [250, 6000]


# Tests: writing a record to a file and reading it back gives the same record, and the keys are saved in
#   sorted order.
# Why: the file is what we actually keep, so it has to come back exactly the same. Sorted keys keep every file
#   in the same order, so comparing two records only shows real differences.
def test_record_round_trips_through_a_file(tmp_path):
    record = sample_record()
    path = tmp_path / "ex-000001.json"
    write_record(record, path)
    assert read_record(path) == record
    text = path.read_text(encoding="utf-8")
    assert text.index('"damage"') < text.index('"example_id"')  # keys are sorted


# Tests: build_record refuses an empty example ID, sources or damage that aren't a list, and a damage list
#   with a plain string in it instead of a step's record.
# Why: every example needs an ID to be found again, and the lists have to hold real records. A loose format
#   here would give us records we can't read back the same way.
@pytest.mark.parametrize(
    "overrides",
    [
        {"example_id": ""},
        {"sources": {"recording_id": "rec-03"}},
        {"damage": {"bits": 24, "stage": "after_filter"}},
        {"damage": ["24-bit"]},
    ],
)
def test_record_rejects_bad_fields(overrides):
    with pytest.raises(ValueError):
        sample_record(**overrides)


# Tests: build_record refuses a record with no sources, or with a source that has a missing, empty,
#   or non-text recording ID.
# Why: the proposal holds out whole source recordings for testing, not just different filter versions of the
#   same recording. We can only split that way if every example says which recording it came from.
@pytest.mark.parametrize(
    "sources",
    [
        [],
        [{"role": "event"}],
        [{"recording_id": ""}],
        [{"recording_id": 3}],
        [{"recording_id": "rec-03"}, {"role": "speech"}],
    ],
)
def test_every_source_names_its_recording(sources):
    with pytest.raises(ValueError):
        sample_record(sources=sources)


# Tests: a damage list with noise before the filter and rounding after it is accepted and kept in order.
#   A step with no stage, an unknown stage, or a before-filter step listed after an after-filter step gets refused.
# Why: the proposal compares noise added before the filter with noise added after it, so every step has to say
#   which side of the filter it was on. The list is in the order the steps ran, so a before-filter step can't
#   come after an after-filter one.
def test_damage_steps_say_where_they_happened():
    record = sample_record()
    assert [step["stage"] for step in record["damage"]] == ["before_filter", "after_filter"]
    bad_lists = [
        [{"noise": "white gaussian", "snr_db": 30.0}],
        [{"noise": "white gaussian", "stage": "middle", "snr_db": 30.0}],
        [{"bits": 24, "stage": "after_filter"}, {"noise": "white gaussian", "stage": "before_filter"}],
    ]
    for damage in bad_lists:
        with pytest.raises(ValueError):
            sample_record(damage=damage)
