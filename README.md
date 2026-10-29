# fpga-audio-source-classifier

CSC 4240 team project. An AI model identifies an unknown filter on an audio channel, a DSP inverse filter tries to bring back what the filter turned down, and a second AI model detects the sound events that were buried. It runs offline on WAV files on Windows or Linux, and we also plan to run a small version of the filter ID model on a MicroBlaze on an Artix-7 FPGA.

- Plan: `docs/project-plan.docx`
- How we use Git: `git-protocol.docx`
- What we tried and what happened: `docs/experiment-log.md`
- How the whole project fits together: `docs/project-map/README.md` (diagrams, with the draw.io source next to them)
- How the filter ID code fits together: `docs/code-map/README.md`
- How the event model could work: `docs/event-model-map/README.md` (a suggested plan, since it isn't built yet)

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

## Filter ID model (first AI layer)

Predict the filter on mono WAV files at the model's sample rate (192 kHz):

```
python -m pipeline.filter_id predict --model model.joblib clip.wav
```

Each prediction prints as one JSON line with a `spec` (type, family, order, cutoff) and how confident the model is in the type and the order. A band-pass order is the total for both sides, so the C compensation filter in `pipeline/reversable_filters/`, which counts each side, gets half. Training isn't in the repo yet. It comes once the model's been tested on the team's recordings.

## Status

The filter ID model is in `pipeline/filter_id/`, with its tests in `pipeline/tests/`. The compensation filter's C code is in `pipeline/reversable_filters/`.
