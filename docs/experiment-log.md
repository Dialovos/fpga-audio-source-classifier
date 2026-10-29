# Experiment Log

Add a new entry at the top every time you run something worth remembering, including things that didn't work. The report needs the ideas we tried and dropped, and this is where they come from.

Copy this template for each entry:

```
## YYYY-MM-DD - short title (your name)

What I tried:
Why:
What happened:
Results folder: results/<YYYY-MM-DD>-<your-name>-<short-name>/ (if any)
Next:
```

## 2026-10-29 - filter ID on the preliminary test data (Le)

What I tried: I ran the filter ID model on the preliminary test data, which is 3 raw voiceovers (VO1-VO3) and a noisy copy of each (VO1_noise-VO3_noise). The model was trained on synthetic sources only. From each file I made 200 test examples the same way the training examples are made: a random 1 s piece scaled to the same level, a random Butterworth filter, white noise at 20-40 dB SNR before or after the filter, and 24-bit rounding. Each noisy copy got the same piece positions, filters, and SNRs as its raw file. I also ran the model on 100 unfiltered 1 s pieces of each file.

Why: This is the model's first check on real speech, and the compensation filter can only undo the filter if the specs it gets are right.

What happened: The model doesn't carry over to real speech yet. On the raw voiceovers, type accuracy was 0.700 and type + order was 0.447 (0.573 with the noise before the filter, 0.335 after), with median cutoff errors of 0.58-0.63 octaves. The noisy copies were worse: type 0.643 and type + order 0.382 (0.498 before, 0.279 after), with median cutoff errors of 0.52-0.71 octaves. The 3 voiceovers scored about the same (0.68-0.73 type accuracy on the raw files), so it isn't one bad file. The biggest mistake on the raw files was calling a high-pass a band-pass (108 of 210). Speech has almost nothing above 8 kHz (0.02-0.6% of its power), so a high-passed voice already looks cut at both ends. On the noisy copies, a band-pass mostly got called a high-pass (94 of 190). With no filter at all, it said low-pass for 75-100% of the pieces, since it has no "no filter" answer and speech's natural drop at the top looks like a low-pass to it. None of the synthetic sources drop off that steeply at the top. As for the files themselves, all six are mono, 192 kHz, 32-bit float, and 301.8 s long. The raw voiceovers are quiet (-42 to -50 dBFS RMS). The noise in the noisy copies is on the whole time, at a steady -27 dBFS in every second. It's mostly below 80 Hz (about 65% of its power) and above 8 kHz (about 29%), and it's 15-23 dB louder than the voice.

Results folder: results/2026-10-29-le-filter-id-prelim-data/ (metrics.json and settings.json; the WAVs stay out of Git)

Next: Give the synthetic sources a speech-like top end, like a steep drop somewhere above 4-8 kHz, or train on the voiceovers too while keeping the event model's test voiceover out. The noisy copies are much noisier than the 20-40 dB SNR we train and test at, so they'd need their own test.
