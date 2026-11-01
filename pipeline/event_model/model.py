"""
Train one event classifier per type, tune its threshold, and save the model.

This file has three groups:
    * Preparing clips
    * The model
    * Scoring and saved models
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.event_model.features import add_context, frame_features
from pipeline.event_model.labels import EVENT_TYPES, events_to_frames, frames_to_events
from pipeline.event_model.scoring import add_counts, drop_edge_detections, match_events, precision_recall

# ********************************
# Preparing clips
# ********************************


# Keeps only features, labels and identifying details after the audio has been
# measured.
@dataclass
class Clip:
    features: np.ndarray
    n_frames: int
    targets: np.ndarray
    events: list[dict]
    duration_s: float
    clip_id: str
    recording_id: str


# Returns a prepared Clip without keeping the audio. It stops with a ValueError
# if the audio is empty, non-finite, or not mono.
def prepare_clip(audio, record, settings) -> Clip:
    sample_rate = settings["sample_rate"]
    frame_settings = settings["features"]
    features, times = frame_features(audio, sample_rate, frame_settings)
    features = add_context(features, frame_settings["context"])

    events = record.get("events", [])
    n_frames = len(times)
    hop = frame_settings["hop"]
    targets = events_to_frames(events, n_frames, sample_rate, hop, EVENT_TYPES)

    duration_s = len(audio) / sample_rate
    clip_id = record["example_id"]
    recording_id = record["sources"][0]["recording_id"]

    return Clip(features, n_frames, targets, events, duration_s, clip_id, recording_id)


# Returns selected training features and targets, keeping every positive
# transient frame. It stops with a ValueError if there are no training clips.
def _training_rows(clips, settings, classes) -> tuple[np.ndarray, np.ndarray]:
    if not clips:
        raise ValueError("fit needs train clips")

    transient = None

    if "transient" in classes:
        transient = classes.index("transient")

    count = 0

    for clip in clips:
        keep = _training_mask(clip, settings, transient)
        count += int(np.count_nonzero(keep))

    # Allocate the selected rows once as float64, which the classifier needs.
    # This avoids keeping both per-clip copies and a joined matrix in memory.
    features = np.empty((count, clips[0].features.shape[1]), dtype=np.float64)
    targets = np.empty((count, len(classes)), dtype=np.uint8)
    first = 0

    for clip in clips:
        keep = _training_mask(clip, settings, transient)
        end = first + int(np.count_nonzero(keep))
        features[first:end] = clip.features[keep]
        targets[first:end] = clip.targets[keep]
        first = end

    return features, targets


# Returns how many times each training row is used for one type: once for a no
# frame, and about (no frames / yes frames) times for a yes frame, so the rare
# yes frames count as much in total as the no frames. Repeating the rows does
# the same job as class_weight="balanced", but without weights scikit-learn
# picks its bin edges about 150 times faster.
def _balanced_repeats(target) -> np.ndarray:
    yes = int(np.count_nonzero(target))
    no = len(target) - yes
    repeats = np.ones(len(target), dtype=np.int64)
    repeats[target == 1] = max(1, round(no / yes))

    return repeats


# Returns the rows kept for training, including every positive transient frame.
def _training_mask(clip, settings, transient) -> np.ndarray:
    keep = np.zeros(clip.n_frames, dtype=bool)
    keep[::settings["model"]["train_frame_step"]] = True

    if transient is not None:
        keep |= clip.targets[:, transient] == 1

    return keep


# ********************************
# The model
# ********************************


# Learns one independent yes-or-no classifier for each event type.
class EventModel:
    # Returns an untrained model with the given settings and thresholds of 0.5.
    def __init__(self, classes, settings, seed):
        self.classes = tuple(classes)
        self.settings = settings
        self.seed = seed
        self.models = []
        self.thresholds = {}

        for kind in self.classes:
            self.thresholds[kind] = 0.5

    # Returns the fitted model. It stops with a ValueError naming a type if it
    # has no positive training rows, or if there are no training clips.
    def fit(self, clips) -> EventModel:
        features, targets = _training_rows(clips, self.settings, self.classes)

        for column, kind in enumerate(self.classes):
            if not np.any(targets[:, column]):
                raise ValueError(f"fit has no yes frames for {kind}")

        self.models = []
        settings = self.settings["model"]

        for column, kind in enumerate(self.classes):
            classifier = HistGradientBoostingClassifier(
                max_iter=settings["max_iter"],
                learning_rate=settings["learning_rate"],
                max_leaf_nodes=settings["max_leaf_nodes"],
                early_stopping=False,
                random_state=self.seed,
            )

            # Balance the yes and no frames by repeating the yes rows.
            target = targets[:, column]
            rows = np.repeat(np.arange(len(target)), _balanced_repeats(target))

            classifier.fit(features[rows], target[rows])
            self.models.append(classifier)
            del rows
            print(f"trained {kind}", flush=True)

        return self

    # Returns one probability column per type. It stops with a ValueError if fit
    # has not finished.
    def predict_frames(self, features) -> np.ndarray:
        if len(self.models) != len(self.classes):
            raise ValueError("model must be fitted before predict_frames")

        columns = []

        for classifier in self.models:
            probabilities = classifier.predict_proba(features)
            columns.append(probabilities[:, 1])

        return np.column_stack(columns)

    # Returns decoded detections after dropping events entirely within the edge
    # margins. It stops with a ValueError if fit has not finished.
    def detect(self, features, n_frames, duration_s) -> list[dict]:
        probabilities = self.predict_frames(features)
        detected = self._decode(probabilities, n_frames, duration_s, self.thresholds)

        return drop_edge_detections(detected, duration_s, self.settings["edge_margin_s"])

    # Returns decoded probabilities using this model's frame and decoding
    # settings.
    def _decode(self, probabilities, n_frames, duration_s, thresholds) -> list[dict]:
        sample_rate = self.settings["sample_rate"]
        hop = self.settings["features"]["hop"]
        decoding = self.settings["decoding"]

        return frames_to_events(
            probabilities, n_frames, sample_rate, hop, duration_s,
            self.classes, thresholds, decoding
        )

    # Returns one type's F1 at a threshold over all tune clips. A missing F1 is
    # counted as zero because there is no evidence that threshold is better.
    def _threshold_score(self, clips, probabilities, kind, threshold, rules) -> float:
        column = self.classes.index(kind)
        sample_rate = self.settings["sample_rate"]
        hop = self.settings["features"]["hop"]
        decoding = self.settings["decoding"]
        thresholds = {kind: threshold}
        classes = (kind,)
        counts = {}

        for clip, probs in zip(clips, probabilities):
            column_probs = probs[:, column:column + 1]
            detected = frames_to_events(
                column_probs, clip.n_frames, sample_rate, hop,
                clip.duration_s, classes, thresholds, decoding
            )
            margin_s = self.settings["edge_margin_s"]
            detected = drop_edge_detections(detected, clip.duration_s, margin_s)
            matched = match_events(detected, clip.events, rules)
            add_counts(counts, matched)

        score = precision_recall(counts)[kind]["f1"]

        if score is None:
            score = 0.0

        return score

    # Returns and saves the best tune thresholds. Ties prefer the threshold
    # closest to 0.5, then the lower value. It stops with a ValueError if fit
    # has not finished.
    def tune_thresholds(self, clips, rules) -> dict:
        probabilities = []

        for clip in clips:
            probabilities.append(self.predict_frames(clip.features))

        for kind in self.classes:
            best_score = -1.0
            best_threshold = 0.5

            for threshold in self.settings["decoding"]["threshold_grid"]:
                score = self._threshold_score(clips, probabilities, kind, threshold, rules)
                tie = (round(abs(threshold - 0.5), 12), threshold)
                best_tie = (round(abs(best_threshold - 0.5), 12), best_threshold)

                if score > best_score or (score == best_score and tie < best_tie):
                    best_score = score
                    best_threshold = threshold

            self.thresholds[kind] = best_threshold

        return dict(self.thresholds)


# ********************************
# Scoring and saved models
# ********************************


# Returns per-type and overall counts and scores on prepared clips. It stops
# with a ValueError if the model has not finished fitting.
def evaluate(model, clips, rules) -> dict:
    counts = {}

    for clip in clips:
        detected = model.detect(clip.features, clip.n_frames, clip.duration_s)
        matched = match_events(detected, clip.events, rules)
        add_counts(counts, matched)

    return {"counts": counts, "scores": precision_recall(counts)}


# Returns after saving the fitted model and the settings used for its features.
def save_model(path, model, settings) -> None:
    joblib.dump({"format": 1, "model": model, "settings": settings}, path)


# Returns a saved model and its settings. It stops with a ValueError if the
# model format is different or the file does not contain an EventModel.
def load_model(path) -> tuple[EventModel, dict]:
    saved = joblib.load(path)

    if saved.get("format") != 1 or not isinstance(saved.get("model"), EventModel):
        raise ValueError("event model format must be 1 with an EventModel")

    return saved["model"], saved["settings"]
