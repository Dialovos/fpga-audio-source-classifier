"""
Turn event labels into frame targets and frame probabilities back into events.

This file has three groups:
    * Frame targets
    * Smoothing and runs
    * Decoding events
"""

from __future__ import annotations

import numpy as np

# The event types the model learns, in column order.
EVENT_TYPES = ("rumble", "hum", "whine", "hiss", "alarm", "thump", "click")

# Short sounds can be shorter than one frame step. Their nearest frame is always
# a target, every one of their yes frames is kept for training, and they are
# scored on their start only.
SHORT_TYPES = ("thump", "click")

# ********************************
# Frame targets
# ********************************


# Returns uint8 frame targets using centers in [start_s, end_s). The nearest
# frame to a short sound's start is also marked, so a click shorter than one
# frame step still has a target. A tie uses the earlier frame.
def events_to_frames(events, n_frames, sample_rate, hop, classes) -> np.ndarray:
    targets = np.zeros((n_frames, len(classes)), dtype=np.uint8)

    for event in events:
        if event["type"] not in classes:
            continue

        column = classes.index(event["type"])

        # Work in frame numbers; ceil chooses the first center at or after each
        # boundary.
        first = max(0, int(np.ceil(event["start_s"] * sample_rate / hop)))
        end = min(n_frames, int(np.ceil(event["end_s"] * sample_rate / hop)))
        targets[first:end, column] = 1

        if event["type"] in SHORT_TYPES and n_frames:
            position = event["start_s"] * sample_rate / hop
            nearest = int(np.ceil(position - 0.5))
            nearest = int(np.clip(nearest, 0, n_frames - 1))
            targets[nearest, column] = 1

    return targets


# ********************************
# Smoothing and runs
# ********************************


# Returns the average of the frames around each frame. Only existing frames are
# used at the start and end of the clip.
def _smooth(values, size) -> np.ndarray:
    half = size // 2
    smoothed = np.empty(len(values))

    for frame in range(len(values)):
        first = max(0, frame - half)
        end = min(len(values), frame + half + 1)
        smoothed[frame] = np.mean(values[first:end])

    return smoothed


# Returns inclusive yes runs, merging gaps of up to the given number of seconds.
def _runs(yes, hop_s, merge_gap_s) -> list[list[int]]:
    runs = []

    for frame in np.flatnonzero(yes):
        frame = int(frame)

        if runs and (frame - runs[-1][1] - 1) * hop_s <= merge_gap_s + 1e-12:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])

    return runs


# ********************************
# Decoding events
# ********************************


# Returns detections sorted by start and type after smoothing, joining gaps, and
# dropping short runs. It stops with a ValueError if the probability shape does
# not match the frame count and classes.
def frames_to_events(probs, n_frames, sample_rate, hop, duration_s, classes, thresholds, decoding) -> list[dict]:
    if np.shape(probs) != (n_frames, len(classes)):
        raise ValueError("probs shape must match n_frames and classes")

    detected = []
    hop_s = hop / sample_rate

    for column, kind in enumerate(classes):
        rules = decoding[kind]
        smoothed = _smooth(probs[:, column], rules["smooth_frames"])
        runs = _runs(smoothed >= thresholds[kind], hop_s, rules["merge_gap_s"])

        for first, last in runs:
            start = max(0.0, first * hop_s - hop_s / 2)
            end = min(duration_s, last * hop_s + hop_s / 2)

            if end <= start or end - start + 1e-12 < rules["min_duration_s"]:
                continue

            confidence = float(np.max(smoothed[first:last + 1]))
            event = {
                "type": kind,
                "start_s": start,
                "end_s": end,
                "confidence": confidence,
            }
            detected.append(event)

    detected.sort(key=event_order)

    return detected


# Returns an event's start and type as the key for sorting events.
def event_order(event):
    return event["start_s"], event["type"]
