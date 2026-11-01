"""
Build, train, detect, and compare labeled audio events.

This file has one group:
    * Main names
"""

from __future__ import annotations

from pipeline.event_model.dataset import make_clip, make_labeled_clip, make_test_cases, test_clip_count
from pipeline.event_model.labels import EVENT_TYPES, events_to_frames, frames_to_events
from pipeline.event_model.model import Clip, EventModel, evaluate, load_model, prepare_clip, save_model
from pipeline.event_model.settings import check_settings

# ********************************
# Main names
# ********************************

__all__ = [
    "EVENT_TYPES",
    "Clip",
    "EventModel",
    "check_settings",
    "evaluate",
    "events_to_frames",
    "frames_to_events",
    "load_model",
    "make_clip",
    "make_labeled_clip",
    "make_test_cases",
    "prepare_clip",
    "save_model",
    "test_clip_count",
]
