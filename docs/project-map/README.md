# Project Map

Blue is built code, green is the shared pipeline, orange is a file or data, purple is hardware, and a dashed box isn't built yet. A solid arrow is a hand-off or call, and a dashed arrow is something done ahead of time, like training or exporting a model, or a link that isn't a call. The filter ID model has its own map in `docs/code-map/`, and the event model has one in `docs/event-model-map/`.

## 1. The whole project

![The whole project](1-whole-project.png)

The dashed arrow from the test data generator is training, done ahead of time. The dashed arrow down to the MicroBlaze is a board-sized version of the filter ID model, made and exported on the PC (tab 8). The board runs C, so it gets its own version of the steps it needs instead of the shared Python code.

## 2. Shared pipeline

![Shared pipeline](2-shared-pipeline.png)

The dashed arrow is an import. `records.py` takes `STAGES` from `degrade.py`, so a record can only say the noise went in before or after the filter. `add_noise` and `quantize` need `stage` passed by name.

## 3. Making the test data

![Making the test data](3-making-the-test-data.png)

Noise added before the filter gets filtered too, while noise added after it covers even what the filter cut, and that's one of the comparisons in the proposal. A band-pass is a high-pass followed by a low-pass, like the C compensation filter, and its order is the total for both sides. The event label times shift to match where the clip starts. Every part gets its labels from the record, so the record has to match what really happened to the audio.

## 4. Undoing the filter

![Undoing the filter](4-undoing-the-filter.png)

The predicted spec is the real system. The true filter is a reference, and comparing the two shows how much the filter ID model's mistakes cost. For the true filter, use the saved `sos` coefficients, since the record's filter entry has extra fields `design_filter` won't take. The boost needs a cap because a straight inverse boosts the noise too, and anything pushed below the noise floor can't come back. The first version is C code in `pipeline/reversable_filters/`. It builds the filter itself, and its order counts each side, so a band-pass gets half of ours.

## 5. Finding the events

![Finding the events](5-finding-the-events.png)

The dashed arrow is the event model learning from the event labels ahead of time.

## 6. The offline app

![The offline app](6-offline-app.png)

The app has to hold on to things between steps. Undoing the filter needs the original WAV and the spec, and the outputs need the predicted filter, the events, and the compensated audio.

## 7. Measuring results

![Measuring results](7-measuring-results.png)

The known filter and the clean clip are references to compare against, and the test data only uses recordings that no model trained on. If undoing the filter doesn't improve detection, that's still a valid result, and we report it.

## 8. MicroBlaze

![MicroBlaze](8-microblaze.png)

The DAW path over Ethernet works today, so it's drawn solid. The board-sized model is either a small model trained to give the same answers or the trees ported to C. The C features have to follow every step in `features.py` (the averaging, bands, dB, median, and slopes), not just the band layout. The board does its math with 32-bit floats or 8-bit integers instead of Python's 64-bit floats, so the last check shows whether those small differences change any predictions.
