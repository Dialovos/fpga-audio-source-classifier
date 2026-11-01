"""
Settings files, seeds, and per-example records.

Every generated example saves its own record: the exact filter settings, the
sources it used, their levels, the damage applied, and its seed. A seed alone
can't rebuild an example once the code or the source files change, so the
record keeps the actual values too.

Every source has to name the recording it came from ("recording_id"), because
we hold out whole source recordings for testing. Synthetic sounds get an ID
too, like "synthetic-tone-100hz". The rest of each source dict is up to the
test data method.

Damage steps go in the order they were applied, and each one says whether it
happened before or after the filter, since the proposal tests noise both ways.

This file has three groups:
    * Settings and seeds
    * Building a record
    * Saving and loading records
"""

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from pipeline.degrade import STAGES

# Bump this if the record layout changes, so old and new records can be told apart.
SCHEMA_VERSION = 2


# ********************************
# Settings and seeds
# ********************************


# Reads a JSON settings file and returns it as a dict. It stops with a
# ValueError if there's no "seed", or if the seed isn't a whole number 0 or
# above.
def load_settings(path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    settings = json.loads(text)

    seed = settings.get("seed")

    # Python treats True as the number 1, so True and False are ruled out on
    # purpose, along with 1.5 and "42".
    is_whole_number = isinstance(seed, Integral) and not isinstance(seed, bool)

    if not is_whole_number or seed < 0:
        raise ValueError(f'settings need a non-negative integer "seed", got {seed!r}')

    return settings


# Returns the seed for example number `index` in a run. The same (base_seed,
# index) always gives the same seed, and different examples get random streams
# that don't overlap. It stops with a ValueError if either number is negative.
def example_seed(base_seed: int, index: int) -> int:
    if base_seed < 0 or index < 0:
        raise ValueError("base_seed and index must be non-negative")

    # numpy's SeedSequence mixes the two numbers into one well-spread seed.
    mixer = np.random.SeedSequence([base_seed, index])
    state = mixer.generate_state(1, dtype=np.uint32)

    return int(state[0])


# ********************************
# Building a record
# ********************************


# Stops if the example ID is empty.
def _check_example_id(example_id: str) -> None:
    if not example_id:
        raise ValueError("example_id can't be empty")


# Returns True if the value is a list and every item in it is a dict.
def _is_list_of_dicts(value) -> bool:
    if not isinstance(value, list):
        return False

    for item in value:
        if not isinstance(item, dict):
            return False

    return True


# Stops unless sources is a list of dicts with at least one source in it, and
# every source has a "recording_id".
def _check_sources(sources) -> None:
    if not _is_list_of_dicts(sources):
        raise ValueError("sources must be a list of dicts")

    if len(sources) == 0:
        raise ValueError("an example needs at least one source")

    # Every source has to say which recording it came from, so we can split by recording.
    for source in sources:
        recording_id = source.get("recording_id")

        if not isinstance(recording_id, str) or recording_id == "":
            raise ValueError(f'every source needs a "recording_id" string, got {recording_id!r}')


# Stops unless damage is a list of dicts where every step has a known stage, and
# all the before_filter steps come first.
def _check_damage(damage) -> None:
    if not _is_list_of_dicts(damage):
        raise ValueError("damage must be a list of dicts, in the order applied")

    stages = []
    for step in damage:
        stages.append(step.get("stage"))

    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"every damage step needs a stage from {STAGES}, got {stages}")

    # The list is in the order the steps ran, so once an after_filter step shows up,
    # a before_filter step can't come later.
    seen_after_filter = False
    for stage in stages:
        if stage == "after_filter":
            seen_after_filter = True
        elif seen_after_filter:
            raise ValueError(f"damage steps must be in the order applied, before_filter first, got {stages}")


# Returns after checking clip and events. It stops with a ValueError if a field
# is missing, a number is invalid, or an event falls outside the clip.
def _check_clip_events(clip, events) -> None:
    if clip is None:
        if events is not None:
            raise ValueError("events needs clip")

        return

    if not isinstance(clip, dict) or not clip:
        raise ValueError("clip must be a dict")

    for key in ("start_s", "duration_s", "gain"):
        value = clip.get(key)
        is_number = isinstance(value, Real) and not isinstance(value, bool)

        if not is_number or not np.isfinite(value):
            raise ValueError(f"clip {key} must be a finite number")

        if value < 0 or (key != "start_s" and value == 0):
            raise ValueError(f"clip {key} is out of range")

    if events is None:
        return

    if not _is_list_of_dicts(events):
        raise ValueError("events must be a list of dicts")

    for event in events:
        if not isinstance(event.get("type"), str) or not event["type"].strip():
            raise ValueError("event type must be a non-empty string")

        for key in ("start_s", "end_s"):
            value = event.get(key)
            is_number = isinstance(value, Real) and not isinstance(value, bool)

            if not is_number or not np.isfinite(value):
                raise ValueError(f"event {key} must be a finite number")

        if not 0 <= event["start_s"] < event["end_s"] <= clip["duration_s"]:
            raise ValueError("event start_s and end_s must fit inside clip duration_s")


# Collects everything needed to rebuild or audit one example: its ID (like
# "ex-000001"), its own seed, the sample rate, its sources (each with a
# "recording_id"), the record from apply_filter, and the records from the damage
# steps in the order they ran. It returns the record as plain Python, ready to
# save as JSON. It stops with a ValueError if the ID is empty, or the sources or
# damage are malformed.
def build_record(
    example_id: str,
    seed: int,
    sample_rate: float,
    sources: list[dict],
    filter_record: dict,
    damage: list[dict],
    *,
    clip=None,
    events=None,
) -> dict:
    _check_example_id(example_id)
    _check_sources(sources)
    _check_damage(damage)
    _check_clip_events(clip, events)

    record = {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "seed": seed,
        "sample_rate": sample_rate,
        "sources": sources,
        "filter": filter_record,
        "damage": damage,
    }

    if clip is not None:
        record["clip"] = clip

    if events is not None:
        record["events"] = events

    return to_jsonable(record)


# Turns numpy values and tuples into plain Python so json can save them. It
# works through dicts and lists all the way down by calling itself on each item.
# Dict keys become text, tuples and numpy arrays become lists, and numpy numbers
# become regular int or float.
def to_jsonable(value):
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            converted[str(key)] = to_jsonable(item)
        return converted

    if isinstance(value, (list, tuple)):
        converted_items = []
        for item in value:
            converted_items.append(to_jsonable(item))
        return converted_items

    if isinstance(value, np.ndarray):
        return value.tolist()

    # A single numpy number, like np.int64(5): turn it into a plain Python number.
    if isinstance(value, np.generic):
        return value.item()

    return value


# ********************************
# Saving and loading records
# ********************************


# Saves a record as readable JSON. The keys are sorted, so two records always
# list things in the same order and comparing them only shows real differences.
def write_record(record: dict, path) -> None:
    text = json.dumps(to_jsonable(record), indent=2, sort_keys=True)

    Path(path).write_text(text + "\n", encoding="utf-8")


# Loads a record saved by write_record.
def read_record(path) -> dict:
    text = Path(path).read_text(encoding="utf-8")

    return json.loads(text)
