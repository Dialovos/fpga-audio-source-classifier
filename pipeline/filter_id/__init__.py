"""
The filter ID model (the first AI layer).

It looks at a filtered clip and predicts the filter that was applied: the type
(high-pass, low-pass, or band-pass), the order, and the cutoffs. Each prediction
comes with a filter spec the compensation filter can use to undo it.

    python -m pipeline.filter_id predict --model model.joblib clip.wav
"""

from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import FilterIdModel, evaluate, load_model, save_model, split_by_recording

# The names other code can import straight from pipeline.filter_id.
__all__ = [
    "FilterIdModel",
    "evaluate",
    "load_model",
    "save_model",
    "spectrum_features",
    "split_by_recording",
]
