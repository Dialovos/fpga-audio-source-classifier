"""
The offline app: find the filter on a recording, undo it, and find the events.

    python -m pipeline.app --filter-id-model FILE --event-model FILE --out DIR clip.wav [more.wav ...]

For each mono 192 kHz WAV file, the app does three steps in order. The filter ID
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
# with its exact name and arguments, what it returns, and when it stops with an
# error. Write each def right under its description and keep the description as
# the function's summary comment.
#
# The tests in pipeline/tests/test_app.py check every one of these. The app is
# done when they all pass:
#
#     python -m pytest pipeline/tests/test_app.py
#
# A good order is one group at a time, rerunning the tests as you go. Until a
# function exists, the whole test file stops at its import line, so it's fine
# to start with every function as a stub (def ...: pass) and fill them in.
#
# House rules (same as the rest of pipeline/):
#     * Simple, plain code: short functions, for loops instead of clever
#       one-liners, clear names.
#     * Comments like the other files: a # summary right above each function
#       saying what it returns and when it stops, and a banner per group (the
#       ones below). Delete these TO DO notes when you're done.
#     * Ruff has to pass: ruff check pipeline --line-length 120
#     * Don't change the tests. If one looks wrong, ask first.
#
# When everything passes, add an "Offline app" section to the README with the
# command above, then open a PR.
# ============================================================================

# Imports you'll need (most of the work is calling code that already exists):
#     argparse, json, sys, and Path from pathlib
#     numpy as np, soundfile as sf
#     compensate                    from pipeline.compensation
#     add_context, frame_features   from pipeline.event_model.features
#     load_model                    from pipeline.event_model.model (import it as load_event_model)
#     spectrum_features             from pipeline.filter_id.features
#     FilterIdModel, load_model     from pipeline.filter_id.model (import load_model as load_filter_id)
#     to_jsonable                   from pipeline.records
#
# Two constants:
#     FILTER_NAMES = {"highpass": "high-pass", "lowpass": "low-pass", "bandpass": "band-pass"}
#     MAX_LISTED_EVENTS = 20    (the summary lists each event only up to this many)

# ********************************
# Loading the models and checking the files
# ********************************


# load_models(filter_id_path, event_model_path) -> dict
#
# Returns both trained models with the settings they were trained on, as a dict
# with these keys: "filter_id", "filter_id_settings", "event_model",
# "event_settings", and "sample_rate" (the event model's).
#   * load_filter_id(path) and load_event_model(path) each return (model, settings).
#   * Both model files use the same format number, so load_filter_id will happily
#     load an event model file. Check that the filter ID model is really a
#     FilterIdModel, and stop with a TypeError if it isn't.
#   * load_event_model already stops with a ValueError if the file isn't an event
#     model. Let that error through.
#   * Stop with a ValueError that says "sample rate" if the two models were
#     trained at different sample rates.


# check_files(paths, sample_rate, min_seconds) -> None
#
# Returns nothing if every file is OK. Stops with a ValueError that names the
# file (put the path in the message) if a file:
#   * can't be read as a WAV file. sf.info(path) raises sf.LibsndfileError or
#     OSError for those; catch them and raise the ValueError instead.
#   * isn't mono (info.channels),
#   * isn't at the models' sample rate (info.samplerate),
#   * is shorter than min_seconds (info.frames compared with min_seconds x sample_rate),
#   * has the same file name (Path(path).stem) as an earlier file in the list,
#     since their outputs would write over each other.
# Use sf.info, not sf.read, so the check doesn't load whole recordings.


# ********************************
# The steps
# ********************************


# identify_filter(audio, models) -> dict
#
# Returns the filter ID model's guess for the whole recording, as {"spec",
# "confidence"}. Make the features exactly like training did:
# spectrum_features(audio, sample_rate, **filter_id_settings["features"]), then
# take models["filter_id"].predict(features)[0].


# find_events(audio, models) -> list[dict]
#
# Returns the events the event model finds in the audio, each a dict with
# "type", "start_s", "end_s", and "confidence". Use the event model's settings
# (models["event_settings"]):
#   1. features, times = frame_features(audio, sample_rate, settings["features"])
#   2. features = add_context(features, settings["features"]["context"])
#   3. return models["event_model"].detect(features, len(times), len(audio) / sample_rate)
# This is the same thing `python -m pipeline.event_model detect` does, and the
# test checks you get the same events.


# analyze_file(path, models, method="python", command=None) -> tuple[dict, np.ndarray]
#
# Runs the three steps on one checked WAV file and returns (report, compensated).
#   1. Read the file: sf.read(path, dtype="float64").
#   2. guess = identify_filter(audio, models)
#   3. compensated, compensation = compensate(audio, sample_rate, guess["spec"],
#      method=method, command=command). method is "python" (our copy of the C
#      inverse) or "c" (Adam's C program, which needs command, a list like
#      ["pipeline/reversable_filters/filter"]). compensate stops with a ValueError
#      if the C program fails; let that through.
#   4. events = find_events(compensated, models). The events come from the
#      compensated audio, not the original.
# The report is a dict with: "file" (just the file name, like "clip.wav"),
# "sample_rate", "duration_s", "filter" (the guess), "compensation" (the record
# compensate returned), and "events".


# ********************************
# Writing and showing the results
# ********************************


# write_outputs(out, report, compensated) -> tuple[Path, Path]
#
# Writes two files to the folder out (make it if it doesn't exist), named after
# the input file's stem, and returns their paths as (wav_path, report_path):
#   * <stem>-compensated.wav: the compensated audio as a 32-bit float WAV
#     (sf.write(..., subtype="FLOAT")), so nothing past full scale gets clipped.
#   * <stem>-report.json: json.dumps(to_jsonable(report), indent=2). to_jsonable
#     turns numpy numbers into plain ones so json can save them.


# describe_filter(spec) -> str
#
# Returns a filter spec in words, with the cutoffs rounded to whole Hz:
#     "band-pass, order 4, 250 Hz to 6000 Hz"
#     "high-pass, order 2, 312 Hz"
#     "low-pass, order 1, 8000 Hz"
# Use FILTER_NAMES for the type. A band-pass's cutoff is a [low, high] list.


# count_events(events, types) -> str
#
# Returns how many events of each type are in the list, for every type in
# types, in that order, including 0s:
#     "rumble 1, hiss 0, transient 2"
# types comes from the event model (models["event_model"].classes), not a
# hard-coded list, since the event model is going from 3 types to 7.


# summary(report, types) -> str
#
# Returns a short summary of one report for the terminal, one fact per line:
#     cockpit.wav (42.0 s)
#       Filter: band-pass, order 4, 250 Hz to 6000 Hz
#         confidence: type 0.90, order 0.60
#       Undone with: the python inverse
#       Events: 3 (rumble 1, hiss 0, transient 2)
#         rumble        0.50 s to     1.75 s, confidence 0.80
#         transient     2.00 s to     2.01 s, confidence 0.60
#         ...
# List each event (type, start, end, confidence) only if there are
# MAX_LISTED_EVENTS or fewer. Otherwise skip the list and add a line saying
# every event is in the report. The test checks the facts, not the exact
# spacing, so this layout is a suggestion.


# ********************************
# The command line
# ********************************


# _build_parser() -> argparse.ArgumentParser
#
# Returns the parser for the app's options:
#     --filter-id-model   required, a filter ID model.joblib
#     --event-model       required, an event model.joblib
#     --method            "python" or "c", default "python"
#     --c-command         the C filter program, needed with --method c
#     --out               required, the folder for the reports and WAVs
#     wavs                one or more WAV files (nargs="+")


# main(argv=None) -> None
#
# Runs the app:
#   1. Read the command line with _build_parser().parse_args(argv).
#   2. If --method is c and there's no --c-command, stop with
#      sys.exit("...") and a message that mentions --c-command.
#   3. Load the models and check every file before analyzing any of them. The
#      min_seconds for check_files is the filter ID model's clip length,
#      models["filter_id_settings"]["clip_seconds"]. If either step raises a
#      ValueError or TypeError, stop with sys.exit(f"error: {error}") so people
#      get a message instead of a traceback.
#   4. For each file: analyze_file, write_outputs, then print the summary and
#      where the two files were saved.
#
# At the very bottom of the file, add the usual:
#     if __name__ == "__main__":
#         main()
