"""
The offline app: find the filter on a recording, undo it, and find the events.

    python -m pipeline.app --filter-id-model FILE --event-model FILE --out DIR clip.wav [more.wav ...]

For each mono 48 kHz WAV file, the app does three steps in order. The filter ID
model guesses the filter, the compensation filter undoes that guess, and the
event model finds the events in the undone (compensated) audio. Then the app
writes a report and the compensated WAV to --out, and prints a short summary of
the file. Every file gets checked before the first one is analyzed, so a bad
file stops the app before anything is written.

This file has four groups:
    * Loading the models and checking the files
    * The steps
    * Writing and showing the results
    * The command line
"""

# ============================================================================
# TO DO (Vadin): this file is a blank spec. Write the app here.
#
# Every function the app needs is described below, in the order to write them,
# with what it's expected to do. The function names have to match, since the
# tests call them by name, but how you write each one is up to you.
#
# The tests in pipeline/tests/test_app.py check every one of these. The app is
# done when they all pass:
#
#     python -m pytest pipeline/tests/test_app.py
#
# When everything passes, add an "Offline app" section to the README with the
# command above, push to this branch, and the draft PR gets marked ready.
# ============================================================================

# The imports and the two constants below are already set up. Most of the work
# is calling code that already exists.
#
# The line right below tells ruff not to flag imports that aren't used yet.
# Delete it once all your functions are written, so ruff can catch any import
# you didn't end up needing.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.compensation import compensate
from pipeline.event_model.model import clip_features
from pipeline.event_model.model import load_model as load_event_model
from pipeline.filter_id.features import spectrum_features
from pipeline.filter_id.model import FilterIdModel
from pipeline.filter_id.model import load_model as load_filter_id
from pipeline.records import to_jsonable

# How each filter type is written in the summary.
FILTER_NAMES = {"highpass": "high-pass", "lowpass": "low-pass", "bandpass": "band-pass"}

# The summary lists each event only when there are this many or fewer, so a long
# recording doesn't flood the terminal. The report always has every event.
MAX_LISTED_EVENTS = 20

# ********************************
# Loading the models and checking the files
# ********************************


# load_models
#
# Loads the two trained models the app runs on and gives back everything the
# later steps need in one dictionary. It takes the path of the filter ID model's
# file first and then the path of the event model's file. Each file was saved
# together with the settings that model was trained with, and both parts of the
# pipeline already have a function that loads their own kind of file and gives
# back the model and its settings.
#
# Both kinds of file are saved the same way, so the filter ID loader will happily
# open an event model file by mistake. Check that what it gives back really is a
# filter ID model, and stop with a TypeError if it isn't. The event model loader
# already refuses a file that isn't an event model, so let its error through.
# The two models also have to agree on the sample rate, or they'd read the same
# audio differently; if they don't, stop with a ValueError whose message mentions
# the sample rate.
#
# The dictionary holds the filter ID model, its settings, the event model, its
# settings, and the sample rate both expect, under the keys "filter_id",
# "filter_id_settings", "event_model", "event_settings", and "sample_rate".
def load_models(filter_id_path, event_model_path) -> dict:
    pass


# check_files
#
# Makes sure every recording can be analyzed before the app touches any of them,
# so one bad file stops the run before anything gets written. It takes the list
# of WAV file paths, the sample rate the models expect, and the shortest length a
# file can be, in seconds (the filter ID model needs at least one clip of the
# length it was trained on). It gives back nothing when every file is fine.
#
# It looks at each file's details without loading the audio, and stops with a
# ValueError at the first file that can't be read as a WAV file at all, has more
# than one channel, is at a different sample rate, or is shorter than the
# minimum. Two files with the same name are a problem too, even in different
# folders, since their outputs would be saved under the same names and write over
# each other. Every message has to name the file, so people know which one to
# fix.
def check_files(paths, sample_rate, min_seconds) -> None:
    pass


# ********************************
# The steps
# ********************************


# identify_filter
#
# Asks the filter ID model what filter was applied to a whole recording. It takes
# the recording's audio (a 1-D array of samples) and the loaded models, and gives
# back the model's answer for that recording: a dictionary with the filter spec
# (its type, family, order, and cutoff, in the form the filter and compensation
# code accepts) and how confident the model is. The model can only answer for
# features made exactly the way its training features were, so make the spectrum
# features from the audio with the feature settings saved with the filter ID
# model.
def identify_filter(audio, models) -> dict:
    pass


# find_events
#
# Runs the event model over a recording and gives back every event it finds, as
# a list of dictionaries with each event's type, start and end time in seconds,
# and confidence. It takes the audio and the loaded models. It has to make the
# frame features exactly the way the event model's own detect command does, from
# the settings saved with the event model, because the test compares its events
# with what python -m pipeline.event_model detect finds in the same file. Reading
# how detect does it in pipeline/event_model/__main__.py is the quickest way in.
# Depending on those settings, the features can be our own, the pretrained PANNs
# model's, or both, and the event model has one function that makes whichever
# ones its settings ask for, so the app doesn't need to know which.
def find_events(audio, models) -> list[dict]:
    pass


# analyze_file
#
# Does the app's whole job for one recording that already passed check_files. It
# takes the file's path, the loaded models, which inverse undoes the filter, and
# the command that runs the C program. The inverse is "python" by default (our
# Python copy of Adam's inverse) or "c" (Adam's C program itself), and the
# command is only needed for "c": a list like
# ["pipeline/reversable_filters/filter"].
#
# It reads the audio, asks identify_filter for the filter, and undoes that guessed
# filter with the compensation code, which handles both inverses and gives back
# the undone audio plus a record of how it was undone. Then it runs find_events on
# the undone audio, not the original, since finding the buried events after
# undoing the filter is the point of the project. If the C program fails, the
# compensation code stops with a ValueError; let it through.
#
# It gives back two things: the report, and the undone (compensated) audio so it
# can be saved next. The report is a dictionary with the file's name (just the
# name, not its folder), the sample rate, the length in seconds, the filter guess,
# the compensation record, and the events, under the keys "file",
# "sample_rate", "duration_s", "filter", "compensation", and "events".
def analyze_file(path, models, method="python", command=None) -> tuple[dict, np.ndarray]:
    pass


# ********************************
# Writing and showing the results
# ********************************


# write_outputs
#
# Saves the results for one recording and gives back where it saved them. It
# takes the output folder (make it if it doesn't exist yet), the report, and the
# compensated audio, and writes two files named after the input file (the
# report's file name without its extension): <name>-compensated.wav and
# <name>-report.json.
#
# The audio has to be saved as a 32-bit float WAV at the report's sample rate.
# Undoing a filter can push the audio past full scale, and any other format would
# clip it, which would hide what the inverse really did. The report gets saved as
# readable JSON; numpy numbers can't go into JSON directly, and pipeline.records
# has a helper that turns them into plain ones. It gives back the two paths, the
# audio's first.
def write_outputs(out, report, compensated) -> tuple[Path, Path]:
    pass


# describe_filter
#
# Turns a filter spec into words for the summary: the type the way people say it
# (FILTER_NAMES above has those), the order, and the cutoff rounded to whole Hz,
# or both edges for a band-pass, whose cutoff is a [low, high] pair. For example
# "band-pass, order 4, 250 Hz to 6000 Hz", "high-pass, order 2, 312 Hz", or
# "low-pass, order 1, 8000 Hz".
def describe_filter(spec) -> str:
    pass


# count_events
#
# Says how many events of each type are in a list of events, as one line of text
# like "rumble 1, hiss 0, transient 2". It takes the events and the types to
# count, in the order to show them, and lists every type, including the ones with
# none. The types come from the event model itself (its classes) instead of being
# written into the app, since the event model has 7 types now and may get more.
def count_events(events, types) -> str:
    pass


# summary
#
# Turns one report into a short block of text for the terminal. It takes the
# report and the event model's list of types. It's the first thing people read
# after a run, so it shows one fact per line: the file name and its length in
# seconds (one decimal), the filter guess in words with the model's confidence in
# the type and the order (two decimals each), which inverse undid it, and how many
# events were found in total and of each type.
#
# Then it lists each event with its type, start and end times (two decimals), and
# confidence, but only when there are MAX_LISTED_EVENTS or fewer. With more than
# that it skips the list and adds a line saying every event is in the report, so
# a long recording doesn't flood the terminal. A layout like this works, though
# the test only checks the facts, not the spacing:
#
#     cockpit.wav (42.0 s)
#       Filter: band-pass, order 4, 250 Hz to 6000 Hz
#         confidence: type 0.90, order 0.60
#       Undone with: the python inverse
#       Events: 3 (rumble 1, hiss 0, transient 2)
#         rumble        0.50 s to     1.75 s, confidence 0.80
#         ...
def summary(report, types) -> str:
    pass


# ********************************
# The command line
# ********************************


# main
#
# Runs the whole app from the command line. It takes the list of command-line
# words (or reads the real command line when it's given nothing), with these
# options: --filter-id-model and --event-model for the two model files, --out for
# the folder the outputs go in, --method for the inverse (python or c, python by
# default), --c-command for the path to Adam's C program (only needed with c),
# and then one or more WAV files. Setting up the options with argparse in a small
# helper function keeps main short.
#
# It first stops with a message if --method c came without --c-command. Then it
# loads the models and checks every file before analyzing any of them; the
# shortest file allowed is the clip length saved in the filter ID model's
# settings. If loading or checking fails, it stops with a short error message
# instead of a traceback (sys.exit with the message as text does that). Then, for
# each file in turn, it analyzes it, writes its outputs, and prints its summary
# and where the two files went.
#
# At the very bottom of the file, the usual if __name__ == "__main__" line calls
# main, so python -m pipeline.app works.
def main(argv=None) -> None:
    pass
