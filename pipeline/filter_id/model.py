"""
The filter ID model: gradient-boosted trees from scikit-learn on spectrum features.

It works in steps. One classifier picks the type. Then, for that type, a second
classifier picks the order, and one regressor per edge predicts the cutoff. Cutoffs
are predicted in octaves (log2 of Hz), so a 10% miss counts the same at 100 Hz and at
10 kHz. This is the baseline; a small PyTorch model can replace it later.

Besides the band levels, the trees also get the slope between neighboring bands.
Trees can't easily subtract one feature from another, and a filter's knee is a change
in slope, so handing them the slope directly helps a lot.

This file has five groups:
    * Tree settings
    * Feature and cutoff helpers
    * The model
    * Scoring
    * Saving and loading
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

# Bump this if the saved file's layout changes, so old files fail with a clear message.
MODEL_FORMAT = 1

# How many trees each model builds, and how big a step each new tree takes.
TREE_ITERATIONS = 300
LEARNING_RATE = 0.1


# ********************************
# Tree settings
# ********************************


# Returns an untrained tree classifier, which picks a category like the type or
# the order. Early stopping is off, so training always runs the same way.
def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=TREE_ITERATIONS,
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        random_state=seed,
    )


# Returns an untrained tree regressor, which predicts a number like a cutoff.
# Early stopping is off, so training always runs the same way.
def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=TREE_ITERATIONS,
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        random_state=seed,
    )


# Stands in for the order classifier when the training data has only one order
# for a type. A real classifier needs at least two choices to learn from, so this
# one just always answers with the one order it saw.
class _OnlyChoice:
    # Remembers the one order every prediction will get.
    def __init__(self, value):
        self.classes_ = np.array([value])

    # Returns a probability of 1 for the one order, for every row.
    def predict_proba(self, features):
        return np.ones((len(features), 1))


# ********************************
# Feature and cutoff helpers
# ********************************


# Returns the band levels with the slope between each pair of neighboring bands
# added on the end. It takes one row of features or many rows, and always returns
# rows (a 2-D array).
def _with_slopes(features) -> np.ndarray:
    X = np.atleast_2d(np.asarray(features, dtype=np.float64))

    # The slope between neighbors is just the difference: band 2 minus band 1, band 3 minus band 2, ...
    slopes = np.diff(X, axis=1)

    return np.hstack([X, slopes])


# Returns a cutoff as a list of edges in Hz: [low, high] for a band-pass, and
# [cutoff] for the other types.
def cutoff_edges(kind: str, cutoff) -> list[float]:
    if kind == "bandpass":
        return [float(cutoff[0]), float(cutoff[1])]

    return [float(cutoff)]


# Returns the names the scores use for a type's cutoff edges.
def _edge_names(kind: str) -> list[str]:
    if kind == "bandpass":
        return ["bandpass_low", "bandpass_high"]

    return [kind]


# Returns the items at the given row numbers, in that order.
def _pick_rows(items: list, rows) -> list:
    picked = []

    for row in rows:
        picked.append(items[row])

    return picked


# ********************************
# The model
# ********************************


# Predicts a filter's type, order, and cutoffs from spectrum features.
class FilterIdModel:
    # Makes an untrained model. The seed makes training repeat exactly.
    def __init__(self, seed: int = 0):
        self.seed = seed

        # These get filled in by fit.
        self.type_model = None
        self.order_models = {}
        self.cutoff_models = {}
        self.cutoff_ranges = {}

    # Trains the model on examples: spectrum features with one row per example,
    # and one {"type", "order", "cutoff"} label per row. It returns the model
    # itself, now trained, and stops with a ValueError if the features and labels
    # have different lengths.
    def fit(self, features, labels: list[dict]) -> FilterIdModel:
        X = _with_slopes(features)

        if len(X) != len(labels):
            raise ValueError(f"got {len(X)} feature rows but {len(labels)} labels")

        # Start fresh, so nothing is left over from an earlier fit.
        self.order_models = {}
        self.cutoff_models = {}
        self.cutoff_ranges = {}

        # Step 1: learn the type from every example.
        types = []
        for label in labels:
            types.append(label["type"])
        types = np.array(types)

        self.type_model = _classifier(self.seed).fit(X, types)

        # Step 2: for each type, learn the order and cutoffs from just that type's examples.
        for kind in np.unique(types).tolist():
            rows = np.flatnonzero(types == kind)
            labels_of_kind = _pick_rows(labels, rows)

            self._fit_order(kind, X[rows], labels_of_kind)
            self._fit_cutoffs(kind, X[rows], labels_of_kind)

        return self

    # Trains the order classifier for one type.
    def _fit_order(self, kind: str, X: np.ndarray, labels: list[dict]) -> None:
        orders = []
        for label in labels:
            orders.append(label["order"])
        orders = np.array(orders)

        # If every example of this type has the same order, there's nothing to learn.
        if len(np.unique(orders)) == 1:
            self.order_models[kind] = _OnlyChoice(orders[0])
        else:
            self.order_models[kind] = _classifier(self.seed).fit(X, orders)

    # Trains one cutoff regressor per edge for one type. The regressors learn log2
    # of the cutoff (octaves), not Hz.
    def _fit_cutoffs(self, kind: str, X: np.ndarray, labels: list[dict]) -> None:
        edges_hz = []
        for label in labels:
            edges_hz.append(cutoff_edges(kind, label["cutoff"]))

        # One row per example, one column per edge (two for band-pass, one otherwise).
        octaves = np.log2(edges_hz)

        regressors = []
        for edge in range(octaves.shape[1]):
            regressor = _regressor(self.seed).fit(X, octaves[:, edge])
            regressors.append(regressor)

        self.cutoff_models[kind] = regressors

        # Remember the lowest and highest cutoff seen in training. Predictions get
        # clipped to this range, so they're always cutoffs the filter code accepts.
        self.cutoff_ranges[kind] = (octaves.min(axis=0), octaves.max(axis=0))

    # Stops if predict is called before the model is trained.
    def _check_trained(self) -> None:
        if self.type_model is None:
            raise RuntimeError("the model isn't trained yet; call fit first or load a saved model")

    # Returns the predicted cutoff edges in Hz for one type, with one row per
    # example and one column per edge.
    def _predict_cutoffs(self, X: np.ndarray, kind: str) -> np.ndarray:
        lowest, highest = self.cutoff_ranges[kind]

        columns = []
        for edge, regressor in enumerate(self.cutoff_models[kind]):
            # Predict in octaves, keep it inside the training range, then turn octaves back into Hz.
            octaves = regressor.predict(X)
            octaves = np.clip(octaves, lowest[edge], highest[edge])
            columns.append(2**octaves)

        return np.column_stack(columns)

    # Predicts the order and cutoffs for every row, assuming every row is the given type.
    def _given_type(self, X: np.ndarray, kind: str) -> list[dict]:
        order_model = self.order_models[kind]
        order_proba = order_model.predict_proba(X)
        best_order = np.argmax(order_proba, axis=1)

        edges_hz = self._predict_cutoffs(X, kind)

        results = []

        for row in range(len(X)):
            if kind == "bandpass":
                cutoff = [float(edges_hz[row, 0]), float(edges_hz[row, 1])]
            else:
                cutoff = float(edges_hz[row, 0])

            order = int(order_model.classes_[best_order[row]])
            order_confidence = float(order_proba[row, best_order[row]])

            spec = {"type": kind, "family": "butter", "order": order, "cutoff": cutoff}
            results.append({"spec": spec, "confidence": {"order": order_confidence}})

        return results

    # Predicts the filter for one row of spectrum features or many rows. If kind is
    # given, it skips the type step and assumes that type, which is how the order
    # and cutoffs get scored on their own. It returns one {"spec", "confidence"}
    # result per row, and the spec has everything the compensation filter needs
    # to undo the filter. It stops with a RuntimeError if the model isn't trained.
    def predict(self, features, kind: str | None = None) -> list[dict]:
        self._check_trained()
        X = _with_slopes(features)

        # If the type is given, every row gets that type with full confidence.
        if kind is not None:
            results = self._given_type(X, kind)

            for result in results:
                result["confidence"]["type"] = 1.0

            return results

        # Step 1: pick the most likely type for each row.
        type_proba = self.type_model.predict_proba(X)
        best_type = np.argmax(type_proba, axis=1)
        predicted_types = self.type_model.classes_[best_type]

        # Step 2: for each type, run that type's order and cutoff models on its rows,
        # then put each result back in its row's place.
        results = []
        for _ in range(len(X)):
            results.append({})

        for predicted in np.unique(predicted_types).tolist():
            rows = np.flatnonzero(predicted_types == predicted)
            results_for_type = self._given_type(X[rows], predicted)

            for row, result in zip(rows, results_for_type):
                result["confidence"]["type"] = float(type_proba[row, best_type[row]])
                results[row] = result

        return results


# Splits the examples into train and test, with every recording entirely on one
# side. groups is the recording_id of each example, test_fraction is about what
# share of the recordings goes to test, and the seed makes the split repeat
# exactly. It returns the train row numbers and the test row numbers.
def split_by_recording(groups, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)

    # The splitter only needs the groups; the zeros stand in for the features.
    placeholder = np.zeros(len(groups))
    train, test = next(splitter.split(placeholder, groups=groups))

    return train, test


# ********************************
# Scoring
# ********************************


# Returns the share of True values (0 to 1), or NaN if there's nothing to count.
def _accuracy(hits) -> float:
    if len(hits) == 0:
        return float("nan")

    return float(np.mean(hits))


# Scores the order and cutoff steps with each example's true type given. It
# returns the order hits (True/False per example, grouped by type) and the cutoff
# errors (in octaves, grouped by edge name).
def _scores_given_true_type(model: FilterIdModel, X: np.ndarray, labels: list[dict]) -> tuple[dict, dict]:
    true_types = []
    for label in labels:
        true_types.append(label["type"])

    order_hits = {}
    cutoff_errors = {}

    for kind in sorted(set(true_types)):
        rows = []
        for i, true_type in enumerate(true_types):
            if true_type == kind:
                rows.append(i)

        results = model.predict(X[rows], kind=kind)

        for i, result in zip(rows, results):
            order_hit = result["spec"]["order"] == labels[i]["order"]
            order_hits.setdefault(kind, []).append(order_hit)

            # How far off is each cutoff edge, in octaves? (1 octave = a factor of 2.)
            true_edges = cutoff_edges(kind, labels[i]["cutoff"])
            predicted_edges = cutoff_edges(kind, result["spec"]["cutoff"])

            for name, true_hz, predicted_hz in zip(_edge_names(kind), true_edges, predicted_edges):
                error = abs(float(np.log2(predicted_hz / true_hz)))
                cutoff_errors.setdefault(name, []).append(error)

    return order_hits, cutoff_errors


# Counts how often each true type was predicted as each type, as
# {true type: {predicted type: count}}.
def _type_confusion(predicted: list[dict], labels: list[dict]) -> dict:
    confusion = {}

    for result, label in zip(predicted, labels):
        true_type = label["type"]
        guess = result["spec"]["type"]

        row = confusion.setdefault(true_type, {})
        row[guess] = row.get(guess, 0) + 1

    return confusion


# Splits the accuracy by whether the noise went in before or after the filter.
def _scores_by_noise_stage(type_hits: np.ndarray, both_hits: np.ndarray, records: list[dict]) -> dict:
    # The first damage step is the noise, and its stage says which side of the filter it was on.
    stages = []
    for record in records:
        stages.append(record["damage"][0]["stage"])
    stages = np.array(stages)

    by_stage = {}

    for stage in sorted(set(stages.tolist())):
        in_stage = stages == stage

        by_stage[stage] = {
            "examples": int(np.sum(in_stage)),
            "type_accuracy": _accuracy(type_hits[in_stage]),
            "type_and_order_accuracy": _accuracy(both_hits[in_stage]),
        }

    return by_stage


# Scores a trained model on held-out examples. It takes the spectrum features,
# the true {"type", "order", "cutoff"} label for each row, and the full record for
# each row (used to split the scores by noise stage). It returns a dict of scores
# that can be saved as JSON. Accuracies go from 0 to 1, and cutoff errors are in
# octaves (0.1 octave is about 7%). It stops with a ValueError if the features,
# labels, and records have different lengths.
def evaluate(model: FilterIdModel, features, labels: list[dict], records: list[dict]) -> dict:
    X = np.atleast_2d(np.asarray(features))

    if not len(X) == len(labels) == len(records):
        raise ValueError(f"got {len(X)} feature rows, {len(labels)} labels, and {len(records)} records")

    predicted = model.predict(X)

    type_hits = []
    both_hits = []

    for result, label in zip(predicted, labels):
        type_right = result["spec"]["type"] == label["type"]
        order_right = result["spec"]["order"] == label["order"]

        type_hits.append(type_right)
        both_hits.append(type_right and order_right)

    type_hits = np.array(type_hits)
    both_hits = np.array(both_hits)

    order_hits, cutoff_errors = _scores_given_true_type(model, X, labels)

    all_order_hits = []
    order_accuracy_by_type = {}

    for kind, hits in order_hits.items():
        all_order_hits.extend(hits)
        order_accuracy_by_type[kind] = _accuracy(hits)

    # The median cutoff error, and the 90th percentile (9 out of 10 errors are smaller).
    cutoff_summary = {}

    for name, errors in cutoff_errors.items():
        cutoff_summary[name] = {
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
        }

    return {
        "examples": len(labels),
        "type_accuracy": _accuracy(type_hits),
        "type_and_order_accuracy": _accuracy(both_hits),
        "order_accuracy_given_true_type": _accuracy(all_order_hits),
        "order_accuracy_given_true_type_by_type": order_accuracy_by_type,
        "cutoff_error_octaves": cutoff_summary,
        "by_noise_stage": _scores_by_noise_stage(type_hits, both_hits, records),
        "type_confusion": _type_confusion(predicted, labels),
    }


# ********************************
# Saving and loading
# ********************************


# Saves a trained model with the settings it was trained on. Predicting needs the
# same feature settings the model was trained with, so they're saved together.
def save_model(path, model: FilterIdModel, settings: dict) -> None:
    saved = {"format": MODEL_FORMAT, "model": model, "settings": settings}

    joblib.dump(saved, path)


# Loads a model saved by save_model and returns the model and the settings it was
# trained on. Only load files you made yourself, since joblib files can run code
# when they're loaded. It stops with a ValueError if the file was saved in a
# different format.
def load_model(path) -> tuple[FilterIdModel, dict]:
    saved = joblib.load(path)

    if saved.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} has model format {saved.get('format')!r}, expected {MODEL_FORMAT}")

    return saved["model"], saved["settings"]
