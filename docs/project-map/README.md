# Project Map

These eight diagrams show how the whole project fits together, so everyone can see where their part sits, what it takes in, and what it hands to the next part. The source is `project-map.drawio`, with one tab per diagram. You can open and edit it at app.diagrams.net, or in VS Code with the Draw.io Integration extension. Most of the project isn't built yet, so most boxes show the plan from the approved proposal and the role sheet. Treat the planned boxes and the files between them as a starting point the team can change, not a final design. Running on the MicroBlaze was a stretch goal in the proposal, and this map assumes we do it. The filter ID model, which is built, has its own detailed map in `docs/code-map/`.

The colors mean the same thing in every diagram. Blue is code that's built, green is the shared pipeline code, orange is a file or data that gets passed between parts, purple is the hardware and firmware, and a dashed box is a part that isn't built yet. An arrow means one part feeds the next, and a dashed arrow means it's used for training or set up ahead of time rather than run every time.

## 1. The whole project

![The whole project](1-whole-project.png)

The project starts with 10 to 20 source recordings and the event labels that say what happens in them and when. The test data generator turns those into labeled test data, and the filter ID model trains on it. Everything inside the offline app box runs when someone analyzes a recording. The recording goes into the filter ID model, which predicts the filter. The compensation filter uses that prediction to undo the filter, and the event model looks for events in the result. What comes out is a report with the predicted filter and the detected events, plus the compensated audio.

The evaluation takes the event model's results on held-out test data and measures whether undoing the filter really helps. The board lane below it is the live side. The capstone board's ADC and FPGA stream the audio to the MicroBlaze, which runs a small version of the filter ID model, and the PC reads its results over UART. The dashed arrow from the filter ID model is the export, which happens ahead of time on the PC. The shared pipeline code along the bottom is used by every part.

## 2. Shared pipeline

![Shared pipeline](2-shared-pipeline.png)

The shared pipeline is four small files in `pipeline/` that every part uses, so the whole team designs filters, adds noise, and saves records the same way. `filters.py` checks a filter spec, designs the filter, applies it, and measures its response. `degrade.py` adds noise at an exact signal-to-noise ratio and rounds audio to a bit depth like a recorder would. `records.py` reads settings files, gives each example its own seed, and builds and saves the record that says exactly what was done to an example. `signals.py` makes simple test sounds (a tone, a thump, and noise) for checking code before real recordings are in.

The line at the bottom of each box says which parts use it. The one dashed arrow is a real code link: `records.py` takes the list of noise stages from `degrade.py`, so a record can only say "before the filter" or "after the filter."

## 3. Making the test data

![Making the test data](3-making-the-test-data.png)

The test data generator makes one example at a time. It cuts a clip from a source recording and scales it to a set level. Then it picks a random filter from the ranges in `settings.json`: a Butterworth high-pass, low-pass, or band-pass, at order 1, 2, or 4. Next it decides whether the noise goes in before or after the filter. That's one of the comparisons the proposal asks for, since noise added before the filter gets filtered too, and noise added after it covers even the parts the filter cut. Either way, the clip then gets rounded to 24-bit like a real recording.

Last, it saves the clip with a record of everything that was done: the source recording it came from, the exact filter, the noise, the rounding, and the events from the labels. That record is where every part gets its labels from, so it has to match what really happened to the audio. The green steps already exist in the shared pipeline, so the generator mostly strings them together.

## 4. Undoing the filter

![Undoing the filter](4-undoing-the-filter.png)

The compensation filter can start from either of two specs. The predicted spec from the filter ID model is the real system. The true filter from the example's record is the best case, and comparing the two shows how much the filter ID model's mistakes cost. Either way, `design_filter` rebuilds the filter, and the next step builds its inverse, which boosts what the filter cut. The boost needs a cap, because a straight inverse would boost the noise along with everything else, and anything pushed below the noise floor can't come back.

The inverse is then applied to the filtered audio. Following the role sheet, the output is the compensated audio with the full content plus the recovery, and a separate file for each recovered band.

## 5. Finding the events

![Finding the events](5-finding-the-events.png)

The event model, which is the second AI layer, works on the compensated audio. It turns the audio into spectrogram features with librosa or torchaudio, and an event classifier built with PyTorch or scikit-learn finds the events in them. The proposal names three kinds of events: low-frequency rumble, broadband hiss, and short transients. Each detected event comes out with its type, its start and end time, and how confident the model is. The dashed arrow shows the event labels from the source recordings, which the classifier learns from during training.

## 6. The offline app

![The offline app](6-offline-app.png)

The offline app is the main deliverable: a Windows or Linux program that analyzes one recording at a time. It loads a WAV file and checks that it's mono and at 192 kHz, and if it isn't, it stops with a message that says what's wrong. Otherwise it runs the three parts in order: identify the filter, undo it, and find the events. Then it writes the report and the compensated audio, and shows the results. Each step only needs the one before it, so the parts can be built and tested on their own before the app ties them together.

## 7. Measuring results

![Measuring results](7-measuring-results.png)

The evaluation answers the main question in the proposal: does undoing the filter help find the events? It only uses held-out recordings that no model has trained on. The same examples go through four versions: the filtered audio as-is, undone with the true filter from the record, undone with the AI's estimate, and the original clean audio as the best possible case. The event model runs on each version, and its detections are compared with the event labels to get precision and recall for each run.

The results get split by whether the noise went in before or after the filter. If undoing the filter doesn't improve detection, that's still a valid result, and we report it.

## 8. MicroBlaze

![MicroBlaze](8-microblaze.png)

This tab shows the on-board path in detail. The capstone board's 64-channel ADC feeds the FPGA, which turns the audio into a TDM stream. That stream already goes over gigabit Ethernet to a DAW, which works today, so it's drawn solid. The new part is the MicroBlaze. On the PC, the trained filter ID model gets exported for C as INT8 weights, along with the band layout the features use and any filter coefficients, and built into the firmware with Vitis. On the board, the MicroBlaze works out the same band levels from the TDM stream in C, using the DDR3 memory for audio buffers, and runs the small model on them. Its results go to the PC over UART, where a reader in `host/` picks them up. The current tree model is far too big for the board (its saved file is about 12 MB), so the board needs a smaller model, like a small neural net, trained to give the same answers.

The last step is checking that the C version gives the same answers as Python on the same clips. The board does its math with 32-bit floats or 8-bit integers instead of Python's 64-bit floats, so small differences are expected, and this check shows whether they change any predictions. The `hardware/`, `firmware/`, and `host/` folders in the repo are where this work goes.

## Updating the diagrams

Edit `project-map.drawio` in app.diagrams.net or VS Code, then export each tab again as a PNG with the same name. In the draw.io desktop app, that's File, Export as, PNG, with a zoom of 200% and a border of 24. From a terminal with draw.io installed, this exports the first tab, and changing the page index and file name does the others:

```
draw.io --export --format png --page-index 1 --scale 2 --border 24 --output docs/project-map/1-whole-project.png docs/project-map/project-map.drawio
```
