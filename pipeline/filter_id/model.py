"""
Layer 1, steps 3 & 4: the model that names the filter, plus how we split, score & save it.

The model is a set of gradient-boosted trees from scikit-learn. A decision tree is a chain of yes/no questions about the
numbers. Boosting builds many small trees in rounds, where each new round works on what the earlier ones got wrong.

It answers in steps. One classifier picks the type. Then, for that type, a second classifier picks the order, and 1
regressor per edge predicts the cutoff. (A classifier picks from a list of choices & a regressor predicts a number.)
Cutoffs are predicted in octaves (log2 of Hz), so being 10% off counts the same at 100 Hz as at 10 kHz.

Besides the 96 band levels, the trees also get the slope between neighboring bands. Trees can't easily subtract one
number from another, and a filter's knee is a change in slope, so handing them the slope directly helps a lot.

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

# The version of the saved file's layout. Bump it if the layout changes, so an old file fails with a clear message &
# doesn't load wrong.
MODEL_FORMAT = 1

# How many rounds of trees each model builds, and how big a step each new round takes.
TREE_ITERATIONS = 300
LEARNING_RATE = 0.1


# ********************************
# Tree settings
# ********************************


# Makes an untrained tree classifier, which picks from a list of choices (like the type or the order). Early stopping is
# off, so training runs the same way every time.
def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=TREE_ITERATIONS,
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        random_state=seed,
    )


# Makes an untrained tree regressor, which predicts a number (like a cutoff). Early stopping is off here too.
def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=TREE_ITERATIONS,
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        random_state=seed,
    )


# A stand-in for the order classifier when the training data only has 1 order for a type. A real classifier needs at
# least 2 choices to learn from, so this one just always answers with the 1 order it saw.
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


# Gives back the band levels with the slope between each pair of neighboring bands added on the end, so 96 numbers
# become 191. It takes 1 row of features or many, and always gives back rows (a 2-D array).
def _with_slopes(features) -> np.ndarray:
    X = np.atleast_2d(np.asarray(features, dtype=np.float64))

    # The slope between neighbors is just the difference: band 2 minus band 1, band 3 minus band 2, ...
    slopes = np.diff(X, axis=1)

    return np.hstack([X, slopes])


# Gives back a cutoff as a list of edges in Hz: [low, high] for a band-pass & [cutoff] for the other 2 types. That way
# the rest of the code can treat every type the same.
def cutoff_edges(kind: str, cutoff) -> list[float]:
    if kind == "bandpass":
        return [float(cutoff[0]), float(cutoff[1])]

    return [float(cutoff)]


# Gives back the names the scores use for a type's cutoff edges.
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


# The model itself. It predicts a filter's type, order & cutoffs from the spectrum features.
class FilterIdModel:
    # Makes an untrained model. The seed makes training repeat exactly.
    def __init__(self, seed: int = 0):
        self.seed = seed

        # These get filled in by fit.
        self.type_model = None
        self.order_models = {}
        self.cutoff_models = {}
        self.cutoff_ranges = {}

    # Trains the model. features has 1 row per example & labels has 1 {"type", "order", "cutoff"} answer per row. It
    # gives back the model itself, now trained. It stops with a ValueError if the features & labels have different
    # lengths.
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

    # Trains the order classifier for 1 type.
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

    # Trains 1 cutoff regressor per edge for 1 type. The regressors learn log2 of the cutoff (octaves), not Hz.
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

        # Remember the lowest and highest cutoff seen in training. Predictions get clipped to this range, so they're
        # always cutoffs the filter code accepts.
        self.cutoff_ranges[kind] = (octaves.min(axis=0), octaves.max(axis=0))

    # Stops if predict gets called before the model is trained.
    def _check_trained(self) -> None:
        if self.type_model is None:
            raise RuntimeError("the model isn't trained yet; call fit first or load a saved model")

    # Gives back the predicted cutoff edges in Hz for 1 type, with 1 row per example & 1 column per edge.
    def _predict_cutoffs(self, X: np.ndarray, kind: str) -> np.ndarray:
        lowest, highest = self.cutoff_ranges[kind]

        columns = []
        for edge, regressor in enumerate(self.cutoff_models[kind]):
            # Predict in octaves, keep it inside the training range, then turn octaves back into Hz.
            octaves = regressor.predict(X)
            octaves = np.clip(octaves, lowest[edge], highest[edge])
            columns.append(2**octaves)

        return np.column_stack(columns)

    # Predicts the order & cutoffs for every row, treating every row as the given type.
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

    # Predicts the filter for 1 row of features or many. Normally it picks the type first. If kind is given, it skips
    # that step & treats every row as that type, which is how the order & cutoffs get scored on their own. It gives back
    # 1 {"spec", "confidence"} result per row. The spec has everything the inverse filter needs to undo the filter. It
    # stops with a RuntimeError if the model isn't trained.
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

        # Step 2: for each type, run that type's order and cutoff models on its rows, then put each result back in its
        # row's place.
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


# Splits the examples into train & test, with every recording entirely on one side. That way the test is always on
# voices the model never trained on. groups is the recording_id of each example, test_fraction is ~what share of the
# recordings go to test, and the seed makes the split come out the same every time. It gives back the train row numbers
# & the test row numbers.
def split_by_recording(groups, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)

    # The splitter only needs the groups; the zeros stand in for the features.
    placeholder = np.zeros(len(groups))
    train, test = next(splitter.split(placeholder, groups=groups))

    return train, test


# ********************************
# Scoring
# ********************************


# Gives back the share of True values (0 to 1), or NaN if there's nothing to count.
def _accuracy(hits) -> float:
    if len(hits) == 0:
        return float("nan")

    return float(np.mean(hits))


# Scores the order & cutoff steps with each example's true type handed to the model, so a wrong type doesn't drag those
# scores down. It gives back the order hits (True/False per example, grouped by type) & the cutoff errors (in octaves,
# grouped by edge name).
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


# Counts how often each true type got predicted as each type, as {true type: {predicted type: count}}. It shows which
# types get mixed up.
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


# Scores a trained model on examples it never trained on. It takes the features, the right answer for each row & the
# full record for each row (used to split the scores by where the noise went). It gives back a dict of scores that can
# be saved as JSON. Accuracies go from 0 to 1, and cutoff errors are in octaves (0.1 octave is ~7%). It stops with a
# ValueError if the features, labels & records have different lengths.
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


# Saves a trained model together with the settings it was trained on. Predicting needs the same feature settings the
# model was trained with, so they travel in the same file.
def save_model(path, model: FilterIdModel, settings: dict) -> None:
    saved = {"format": MODEL_FORMAT, "model": model, "settings": settings}

    joblib.dump(saved, path)


# Loads a model saved by save_model & gives back the model and its settings. Only load files you made yourself, since
# joblib files can run code when they're loaded. It stops with a ValueError if the file was saved in a different format.
def load_model(path) -> tuple[FilterIdModel, dict]:
    saved = joblib.load(path)

    if saved.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} has model format {saved.get('format')!r}, expected {MODEL_FORMAT}")

    return saved["model"], saved["settings"]
