"""
Match events once each and report counts, precision, recall, and F1.

This file has two groups:
    * Matching events
    * Adding counts and reporting scores
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.event_model.labels import EVENT_TYPES

# ********************************
# Matching events
# ********************************


# Returns original indices of one event type, keeping their given order.
def _indices(events, kind) -> list[int]:
    indices = []

    for index, event in enumerate(events):
        if event["type"] == kind:
            indices.append(index)

    return indices


# Returns counts and original index pairs for each type. Each event can match
# only once, using the onset collar and the offset rule for longer sounds.
def match_events(detected, labeled, rules) -> dict:
    counts = {}

    for kind in EVENT_TYPES:
        detections = _indices(detected, kind)
        labels = _indices(labeled, kind)
        cost = np.full((len(detections), len(labels)), 1e6)

        for row, detection_index in enumerate(detections):
            detection = detected[detection_index]

            for column, label_index in enumerate(labels):
                label = labeled[label_index]
                difference = abs(detection["start_s"] - label["start_s"])
                fits = difference <= rules["onset_collar_s"][kind] + 1e-12

                if kind in rules["offset_types"]:
                    duration = label["end_s"] - label["start_s"]
                    collar = max(rules["offset_collar_s"], rules["offset_fraction"] * duration)
                    fits = fits and abs(detection["end_s"] - label["end_s"]) <= collar + 1e-12

                if fits:
                    cost[row, column] = difference

        pairs = []
        rows, columns = linear_sum_assignment(cost)

        for row, column in zip(rows, columns):
            if cost[row, column] < 1e6:
                pairs.append((detections[row], labels[column]))

        count = len(pairs)
        counts[kind] = {
            "tp": count,
            "fp": len(detections) - count,
            "fn": len(labels) - count,
            "pairs": pairs,
        }

    return counts


# Returns detections that are not entirely within the excluded margins.
def drop_edge_detections(detected, duration_s, margin_s) -> list[dict]:
    kept = []

    for event in detected:
        if event["end_s"] > margin_s and event["start_s"] < duration_s - margin_s:
            kept.append(event)

    return kept


# ********************************
# Adding counts and reporting scores
# ********************************


# Returns totals after adding each type's counts in place. Missing totals start
# at zero so the same function can add the first clip and all later clips.
def add_counts(totals, counts) -> dict:
    for kind in EVENT_TYPES:
        current = totals.setdefault(kind, {"tp": 0, "fp": 0, "fn": 0})

        for key in ("tp", "fp", "fn"):
            current[key] += counts[kind][key]

    return totals


# Returns counts and scores for one type. A score is None when there is nothing
# to divide by, rather than inventing a precision or recall.
def _scores(counts) -> dict:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = None

    if tp + fp > 0:
        precision = tp / (tp + fp)

    recall = None

    if tp + fn > 0:
        recall = tp / (tp + fn)

    f1 = None

    if 2 * tp + fp + fn > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Returns per-type and overall scores from total counts.
def precision_recall(counts) -> dict:
    scores = {}
    overall = {"tp": 0, "fp": 0, "fn": 0}

    for kind in EVENT_TYPES:
        current = counts.get(kind, {"tp": 0, "fp": 0, "fn": 0})
        scores[kind] = _scores(current)

        for key in overall:
            overall[key] += current[key]

    scores["overall"] = _scores(overall)

    return scores
