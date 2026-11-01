"""
Compare clean, filtered, known-filter recovery and blind filter-ID recovery on the same cases.

This file has three groups:
    * Version measurements
    * Filter ID scores
    * Running the comparison
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.compensation import _filter, compensate
from pipeline.degrade import quantize
from pipeline.event_model.dataset import make_clip, make_test_cases, test_clip_count
from pipeline.event_model.model import prepare_clip
from pipeline.event_model.scoring import add_counts, match_events, precision_recall
from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import cutoff_edges
from pipeline.records import to_jsonable

# ********************************
# Version measurements
# ********************************


# Returns detections for one full-audio version using the saved settings. It
# stops with a ValueError if the audio is invalid or the model is not fitted.
def _detect(model, audio, record, settings) -> list[dict]:
    clip = prepare_clip(audio, record, settings)

    return model.detect(clip.features, clip.n_frames, clip.duration_s)


# Returns after adding a case's counts, peak, and actual clipping count to one
# version's totals.
def _add_version(bucket, version, counts, audio, clipped) -> None:
    totals = bucket.setdefault(version, {
        "case_count": 0,
        "counts": {},
        "peak": 0.0,
        "clipped_samples": 0,
    })
    totals["case_count"] += 1
    add_counts(totals["counts"], counts)

    peak = float(np.max(np.abs(audio)))
    totals["peak"] = max(totals["peak"], peak)
    totals["clipped_samples"] += clipped


# Returns after adding readable scores to each version's total counts.
def _finish(bucket) -> None:
    for totals in bucket.values():
        totals["scores"] = precision_recall(totals["counts"])


# Returns after writing every version as FLOAT WAV without changing its levels.
def _export(folder, case_id, versions, sample_rate) -> None:
    for version, audio in versions.items():
        path = folder / f"{case_id}-{version}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")


# ********************************
# Filter ID scores
# ********************************


# Returns after adding blind type and order hits and the cutoff errors from
# cases where the predicted type is right.
def _filter_id_counts(totals, spec, predicted) -> None:
    right_type = predicted["type"] == spec["type"]
    totals["type_hits"].append(right_type)
    totals["joint_hits"].append(right_type and predicted["order"] == spec["order"])

    if right_type:
        errors = []
        actual_edges = cutoff_edges(spec["type"], spec["cutoff"])
        predicted_edges = cutoff_edges(predicted["type"], predicted["cutoff"])

        for actual, guess in zip(actual_edges, predicted_edges):
            errors.append(abs(float(np.log2(guess / actual))))

        totals["cutoff_errors"].extend(errors)


# Returns filter ID accuracy and cutoff summaries with their case and edge
# counts. A value is None when there were no cases to score.
def _filter_id_scores(totals) -> dict:
    count = len(totals["type_hits"])
    errors = totals["cutoff_errors"]

    type_accuracy = None
    joint_accuracy = None

    if count > 0:
        type_accuracy = float(np.mean(totals["type_hits"]))
        joint_accuracy = float(np.mean(totals["joint_hits"]))

    median_error = None

    if errors:
        median_error = float(np.median(errors))

    cutoff_summary = {
        "median": median_error,
        "cases": sum(totals["type_hits"]),
        "edges": len(errors),
    }

    return {
        "type_accuracy": {"value": type_accuracy, "cases": count},
        "type_and_order_accuracy": {"value": joint_accuracy, "cases": count},
        "cutoff_error_octaves": cutoff_summary,
    }


# Returns recovered audio and its record. An unsupported prediction returns
# unchanged audio with a failure message. It stops with a ValueError if an
# otherwise supported inverse run fails.
def _ai_inverse(audio, settings, predicted, method, command) -> tuple[np.ndarray, dict, str | None]:
    try:
        _filter(predicted, settings["sample_rate"])
    except ValueError as error:
        record = {
            "method": method,
            "spec": predicted,
            "not_compensated": True,
            "peak": float(np.max(np.abs(audio))),
        }
        return audio, record, str(error)

    output, record = compensate(audio, settings["sample_rate"], predicted, method=method, command=command)

    return output, record, None


# ********************************
# Running the comparison
# ********************************


# Returns a case's audio versions and complete record, including unsupported
# predictions. It stops with a ValueError if a supported inverse run fails.
def _compare_case(model, settings, case, record, clean, clean_detections, filter_id, method, command):
    versions = {"clean": clean, "filtered": case["audio"]}
    detections = {"clean": clean_detections}
    sample_rate = settings["sample_rate"]
    known, known_record = compensate(
        case["audio"], sample_rate, case["spec"], method=method, command=command
    )

    versions["known_undone"] = known
    compensation = {"known_undone": known_record}
    predicted = None
    failure = None

    if filter_id is not None:
        fid_model, fid_settings = filter_id
        features = spectrum_features(case["audio"], settings["sample_rate"], **fid_settings["features"])
        prediction_rows = features.reshape(1, -1)
        predicted = fid_model.predict(prediction_rows)[0]["spec"]
        recovered, recovery_record, failure = _ai_inverse(case["audio"], settings, predicted, method, command)
        versions["ai_undone"] = recovered
        compensation["ai_undone"] = recovery_record

    for version, audio in versions.items():
        if version != "clean":
            detections[version] = _detect(model, audio, record, settings)

    case_record = {
        "clip_record": record,
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "spec": case["spec"],
        "predicted_spec": predicted,
        "snr_db": case["snr_db"],
        "noise_seed": case["noise_seed"],
        "noise_sequence": case["noise_sequence"],
        "stage": case["stage"],
        "damage": case["damage"],
        "filter": case["filter"],
        "compensation": compensation,
        "failure": failure,
        "detections": detections,
    }

    return versions, case_record


# Returns after adding each version's case scores to overall, noise-stage, and
# SNR-bin totals.
def _count_case(totals, by_stage, by_snr, case, versions, case_record, clean_clipped, settings) -> None:
    snr_bin = "[30, 40]"

    if case["snr_db"] < 30:
        snr_bin = "[20, 30)"

    for version, audio in versions.items():
        detections = case_record["detections"][version]
        labels = case_record["clip_record"]["events"]
        counts = match_events(detections, labels, settings["scoring"])

        clipped = 0

        if version == "clean":
            clipped = clean_clipped
        elif version == "filtered":
            clipped = case["damage"][-1]["clipped_samples"]

        _add_version(totals, version, counts, audio, clipped)
        _add_version(by_stage[case["stage"]], version, counts, audio, clipped)
        _add_version(by_snr[snr_bin], version, counts, audio, clipped)


# Returns all case-matched totals and records, optionally writing JSON lines and
# FLOAT examples. It stops with a ValueError for invalid input or a failed
# inverse run, except for unsupported blind predictions.
def run_comparison(model, settings, recordings, split, filter_id=None, method="python", command=None, export_dir=None):
    started = time.perf_counter()

    if method not in ("python", "c"):
        raise ValueError("method must be python or c")

    export = None

    if export_dir is not None:
        export = Path(export_dir)

    if export is not None:
        export.mkdir(parents=True, exist_ok=True)

    totals = {}
    by_stage = {"before_filter": {}, "after_filter": {}}
    by_snr = {"[20, 30)": {}, "[30, 40]": {}}
    fid_totals = {"type_hits": [], "joint_hits": [], "cutoff_errors": []}
    case_records = []
    count = test_clip_count(recordings, split, settings)

    for index in range(count):
        clean, record = make_clip("test", index, recordings, split, settings)
        clean_rounded, clean_rounding = quantize(clean, bits=settings["bits"], stage="after_filter")
        clean_detections = _detect(model, clean_rounded, record, settings)

        do_export = export is not None and index < settings["clips"]["export_examples"]

        if do_export:
            clean_path = export / f"{record['example_id']}-clean.wav"
            sf.write(clean_path, clean_rounded, settings["sample_rate"], subtype="FLOAT")

        for case in make_test_cases(clean, record, index, settings):
            versions, case_record = _compare_case(
                model, settings, case, record, clean_rounded, clean_detections,
                filter_id, method, command
            )

            if filter_id is not None:
                _filter_id_counts(fid_totals, case["spec"], case_record["predicted_spec"])

            clean_clipped = clean_rounding["clipped_samples"]
            _count_case(
                totals, by_stage, by_snr, case, versions, case_record,
                clean_clipped, settings
            )
            case_records.append(case_record)

            if do_export:
                _export(export, case["case_id"], versions, settings["sample_rate"])

            del versions

        print(f"compared test clip {index + 1}/{count}", flush=True)

    _finish(totals)

    for bucket in by_stage.values():
        _finish(bucket)

    for bucket in by_snr.values():
        _finish(bucket)

    if export is not None:
        with (export.parent / "cases.jsonl").open("w", encoding="utf-8") as output:
            for case_record in case_records:
                output.write(json.dumps(to_jsonable(case_record)) + "\n")

    filter_id_scores = None

    if filter_id is not None:
        filter_id_scores = _filter_id_scores(fid_totals)

    return {
        "test_clips": count,
        "case_count": len(case_records),
        "method": method,
        "ai_undone_included": filter_id is not None,
        "versions": totals,
        "by_noise_stage": by_stage,
        "by_snr_bin": by_snr,
        "filter_id": filter_id_scores,
        "cases": case_records,
        "seconds": time.perf_counter() - started,
    }
