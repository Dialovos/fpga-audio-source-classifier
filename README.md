# fpga-audio-source-classifier

CSC 4240 team project. A recording goes through an unknown filter that cuts out part of the sound, and our system tries to get it back in 3 parts. Layer 1 is an AI model that looks at the filtered recording & figures out which filter it went through. The inverse filter is DSP written in C, and it uses that answer to bring back what the filter turned down. Layer 2 is a second AI model that finds the sound events that were buried.

Right now this repo has layer 1 (`pipeline/filter_id/`) and the inverse filter (`pipeline/reversable_filters/`, with a Python copy of it in `pipeline/compensation.py`). Layer 2 is being written. The earlier version of layer 2, the notebook & the diagrams are on the `archive/event-model` branch if you need them. Our first plan also had an offline app that tied the 3 parts into one command. The team dropped it, so it isn't part of the project anymore. Our notes on what we tried & what happened are in `docs/experiment-log.md`.

## All the commands, in order

Run these from the repo folder. The sections below explain each step, and the recordings have to be in place before the `train` line (step 3).

```
git clone https://github.com/Dialovos/fpga-audio-source-classifier.git
cd fpga-audio-source-classifier
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (Linux / macOS)
pip install -r requirements.txt
python -m pytest pipeline/tests -q
python -m pipeline.filter_id train --out results/my-run
python -m pipeline.filter_id predict --model results/my-run/model.joblib clip.wav
```

## 1. Set up

You need Python 3.12, which is the version our results were made with. The `venv` lines make a private Python environment in the repo folder, and the `pip` line installs the 5 packages we use into it.

## 2. Check that it works

```
python -m pytest pipeline/tests -q
```

It should end with `115 passed` after ~30 seconds. The tests make their own audio, so you don't need the recordings for this step.

The C filter has its own checks. They build the C program, compare its output with Python, and try it on bad calls. They need Linux with gcc, make & FFTW (on Ubuntu: `sudo apt install build-essential libfftw3-dev`), so the command above leaves them out. Run them by themselves:

```
python -m pytest pipeline/tests/check_c_filter.py
```

On Windows or macOS that command skips every check & says why, so it can't fail on a machine that can't build the C program.

## 3. Put the recordings in place

The speech is the CSTR VCTK Corpus (version 0.80), saved as one mono 48 kHz WAV per speaker. That's 109 files & ~15 GB, which is too big for Git. Copy the 109 clean WAVs into `data/adam-good/clean/` inside the repo folder. Git ignores everything inside `data/`, so they won't get committed.

To train on other recordings, use any folder of mono 48 kHz WAVs that are at least 1 second long. Make a copy of `pipeline/filter_id/settings.json`, change `source_dir` in the copy to your folder, and add `--settings your-copy.json` to the train command.

## 4. Train layer 1

```
python -m pipeline.filter_id train --out results/my-run
```

First it cuts 200 random 1-second pieces of speech from each recording. It runs each piece through a random filter that it picks itself, so it knows the right answer, and then it adds a little noise. It sets 20% of the speakers aside for testing, and the model never trains on them. Then it trains on the other 80% & scores itself on the speakers it set aside.

It writes 3 files into the new folder. `model.joblib` is the trained model, which is ~20 MB, so Git ignores it. `metrics.json` has the scores & which speakers were set aside. `settings.json` has the settings the run used. With the 109 recordings the whole thing takes ~5 minutes. `--out` has to be a folder that doesn't exist yet, so an old run never gets overwritten.

To check your run against ours, open your `metrics.json` next to `results/2026-11-12-le-filter-id-speech-109/metrics.json`. On the same machine, two runs come out identical except for `seconds`. On a different machine the scores should be the same or very close. If `test_recording_ids` is different, your WAV files aren't the same as ours.

On the 22 speakers it set aside, our run names the right filter type 95.9% of the time & gets the type and the order both right 81.6% of the time. Its cutoff is typically off by ~0.3 octaves, which is ~23%.

## 5. Ask it about a recording

```
python -m pipeline.filter_id predict --model results/my-run/model.joblib clip.wav
```

The WAV has to be mono & 48 kHz, and you can list more than one. Each file prints as one line:

```
{"file": "clip.wav", "spec": {"type": "bandpass", "family": "butter", "order": 2, "cutoff": [125.4, 3508.5]}, "confidence": {"order": 0.50, "type": 0.95}}
```

The `type` is `lowpass`, `highpass`, or `bandpass`. The `order` is how steep the filter is. It's 1, 2, or 4, and a band-pass counts both of its sides together, so it's 2, 4, or 8. The `cutoff` is where the filter starts cutting, in Hz, and a band-pass has two of them: `[low, high]`. The `confidence` is how sure the model is about the type & the order, from 0 to 1.

The model always names a filter. It has no "this wasn't filtered" answer, so only give it filtered audio. Also, the C inverse filter counts each side of a band-pass by itself, so it gets half of the band-pass order.

## Credits

The speech recordings are from the CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.80), by Christophe Veaux, Junichi Yamagishi & Kirsten MacDonald, The Centre for Speech Technology Research, University of Edinburgh. It's shared under the Open Data Commons Attribution License (ODC-By) v1.0.
