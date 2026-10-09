# fpga-audio-source-classifier

CSC 4240 team project. An AI model identifies an unknown filter on an audio channel, a DSP inverse filter tries to bring back what the filter turned down, and a second AI model detects the sound events that were buried. We build on the PC first, and parts of it run on a MicroBlaze on an Artix-7 FPGA.

- Plan: `docs/project-plan.docx`
- How we use Git: `git-protocol.docx`
- What we tried and what happened: `docs/experiment-log.md`

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

## Status

Early setup. The Python pipeline lives in `pipeline/`, with its tests in `pipeline/tests/`.
