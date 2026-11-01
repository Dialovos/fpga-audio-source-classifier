# fpga-audio-source-classifier

CSC 4240 team project. An AI model identifies an unknown filter on an audio channel, a DSP inverse filter tries to bring back what the filter turned down, and a second AI model detects the sound events that were buried. It runs offline on WAV files on Windows or Linux, and we also plan to run a small version of the filter ID model on a MicroBlaze on an Artix-7 FPGA.

- Plan: `docs/project-plan.docx`
- How we use Git: `git-protocol.docx`
- What we tried and what happened: `docs/experiment-log.md`
- How the whole project fits together: `docs/project-map/README.md` (diagrams, with the draw.io source next to them)
- How the filter ID code fits together: `docs/code-map/README.md`
- How we planned the event model: `docs/event-model-map/README.md` (the code in `pipeline/event_model/` may differ in the details)

## Setup

You need Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (Linux / macOS)
pip install -r requirements.txt
```

## Test

Run this from the repo root:

```
python -m pytest -q
```

The C filter in `pipeline/reversable_filters/` has its own checks, which build it, compare its output with Python, and try it on bad calls. They need Linux with gcc, make, and FFTW (on Ubuntu: `sudo apt install build-essential libfftw3-dev`), so the run above leaves them out. Run them on their own:

```
python -m pytest pipeline/tests/check_c_filter.py
```

## Pipeline walkthrough

The easiest way to try everything is `pipeline-walkthrough.ipynb` in the repo root. It takes one sound through the whole pipeline: it filters it, adds noise, has the filter ID model guess the filter, and undoes the filter with the real settings and with the guess. It saves every version as a WAV so you can listen to it in a DAW. Then it trains or loads the event model on the team's voiceovers, plots a test clip, runs the four-version comparison, finds the events in a WAV, and runs the tests.

Open it in VS Code or Jupyter with the `.venv` kernel. Change `pipeline-walkthrough-settings.json` to try a different sound, filter, noise level, or model, then rerun the cells. The notebook's first cell lists what each setting does. The voiceovers go in `data/test-data-prelim/`.

## Filter ID model (first AI layer)

Predict the filter on mono WAV files at the model's sample rate (192 kHz):

```
python -m pipeline.filter_id predict --model model.joblib clip.wav
```

Each prediction prints as one JSON line with a `spec` (type, family, order, cutoff) and how confident the model is in the type and the order. A band-pass order is the total for both sides, so the C compensation filter in `pipeline/reversable_filters/`, which counts each side, gets half. Training isn't in the repo yet. It comes once the model's been tested on the team's recordings.

## Event model (second AI layer)

The event model finds rumble, hiss, and transients (clicks and knocks) in the audio, and gives each one a start, an end, and a confidence.

To train it from the command line (takes ~6 minutes):

```
python -m pipeline.event_model train --out results/<YYYY-MM-DD>-<your-name>-<short-name>
```

It trains on the recordings in `sources.source_dir` of `pipeline/event_model/settings.json`, which is the team's voiceovers in `data/test-data-prelim/` by default, with each noisy copy grouped under its raw file. To use other recordings, point it at a folder of mono 192 kHz WAV files and list any noisy copies in `recording_groups`.

To find the events in mono 192 kHz WAV files:

```
python -m pipeline.event_model detect --model results/<run>/model.joblib clip.wav
```

Each file prints as one JSON line with its events.

To compare detection on four versions of the same held-out test clips (clean, filtered, filter undone with the known settings, and filter undone with the filter ID model's guess):

```
python -m pipeline.event_model compare --model results/<run>/model.joblib --filter-id-model <filter-id-model.joblib> --out results/<run>/compare
```

It writes the scores to `metrics.json`, one line per test case to `cases.jsonl`, and a few example WAVs to `examples/`. Leave out `--filter-id-model` to skip the AI guess version. The filter gets undone with our Python copy of the C inverse. On Linux, where the C program builds, add `--method c --c-command pipeline/reversable_filters/filter`.

## Status

The filter ID model is in `pipeline/filter_id/` and the event model is in `pipeline/event_model/`, with their tests in `pipeline/tests/`. The functions they share (filters, test sounds, noise and rounding, records, and the Python copy of the C inverse in `compensation.py`) are in `pipeline/`. The compensation filter's C code is in `pipeline/reversable_filters/`. The functions the walkthrough notebook calls are in `pipeline/walkthrough.py`.
