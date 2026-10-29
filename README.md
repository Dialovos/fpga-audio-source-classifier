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

## Status

Early setup. The Python pipeline lives in `pipeline/`, with its tests in `pipeline/tests/`.
