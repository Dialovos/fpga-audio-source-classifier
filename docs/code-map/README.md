# Filter ID Code Map

These five diagrams show how the code for the first AI layer (the filter ID model) fits together, from the files and who calls whom, down to what happens inside the model. The source is `filter-id-code-map.drawio`, with one tab per diagram. You can open and edit it at app.diagrams.net, or in VS Code with the Draw.io Integration extension. The colors mean the same thing in every diagram. Blue is first-layer code, yellow is the `FilterIdModel` class, green is the shared pipeline code, orange is a file, and a dashed gray box is a part that isn't built yet. A solid arrow means one part calls or feeds another, and a dashed arrow means it uses or matches something without calling it directly. `dataset.py` is left out on purpose, since it's a temporary stand-in that makes synthetic training data until the real test data generator is ready.

## 1. Code structure

![Code structure](1-code-structure.png)

The first layer lives in `pipeline/filter_id/` and comes down to three files. `__main__.py` is the command line, and its `train` and `predict` commands are the only entry points. To turn audio into numbers, it calls `spectrum_features` in `features.py`, which gives back 96 band levels per clip. To train and score, it calls `split_by_recording`, `evaluate`, `save_model`, and `load_model` in `model.py`, and it calls `fit` and `predict` on the `FilterIdModel` class, which `model.py` defines (that's the diamond). The long arrow down the left side is the command line calling `load_settings` in `records.py` to read the settings file.

The bottom row is the shared pipeline code that every part of the project uses. `records.py` borrows the list of noise stages (`STAGES`) from `degrade.py`, which is the dashed arrow between them. The dashed arrow from `FilterIdModel` to `filters.py` isn't a call. It means the spec that `predict` returns is in exactly the format `design_filter` takes, so the compensation filter can rebuild the predicted filter straight from it.

## 2. How the parts connect

![How the parts connect](2-how-the-parts-connect.png)

This is the whole project from a distance. The test data generator, which is being built, makes a WAV file and a record for every example, and the filter ID model trains on those. Everything inside the offline app box runs when someone analyzes a recording. The recording goes into the filter ID model, which predicts the filter's type, order, and cutoffs. The compensation filter uses that prediction to undo the filter, and the event model then looks for events in the compensated audio. What comes out is the predicted filter, the detected events, and the compensated WAV file.

The dashed arrows going down show which shared code each part leans on. The test data generator uses all three shared files, since it applies filters, adds noise and rounding, and writes a record for each example. The filter ID model uses `records.py` to read its settings, and the compensation filter uses `filters.py` to rebuild the predicted filter. The test data generator only runs before training, so it sits outside the app.

## 3. Training

![Training](3-training.png)

A training run starts by reading `settings.json` with `load_settings`, which hands the feature settings to `spectrum_features`. Each example in the labeled test data gives two things. Its audio goes to `spectrum_features` and becomes 96 band levels, and its record gives the labels: the type, order, and cutoff of the filter that was really applied. The labels always come from the saved record, never from what someone meant to apply.

`split_by_recording` then puts each whole recording on either the training side or the test side, never both, so the scores show how the model does on audio it has never heard. The training rows go to `FilterIdModel.fit`, and the trained model and the test rows go to `evaluate`. The results folder gets the saved model (`model.joblib`), the scores (`metrics.json`), a copy of the settings, and every example's record. On the current synthetic data, the baseline gets the type right 95% of the time and the type and order both right 86% of the time on 40 held-out sources, with a median cutoff error of 0.25 to 0.42 octaves.

## 4. Predicting

![Predicting](4-predicting.png)

Predicting works on one mono WAV file at a time. `_read_mono_wav` reads the file and stops with a clear message if it isn't mono or isn't at the sample rate the model was trained on (192 kHz). `spectrum_features` turns it into band levels using the same settings that were saved with the model, since the model only understands features made the same way it was trained. `FilterIdModel.predict` then returns a spec with the type, family, order, and cutoff, plus how confident it is in the type and the order. The last two boxes are dashed because they aren't built yet. The compensation filter will take the spec and rebuild the filter with `design_filter`, then pass the compensated audio to the event model.

## 5. Inside FilterIdModel

![Inside FilterIdModel](5-inside-filteridmodel.png)

Inside the model, the 96 band levels first go through `_with_slopes`, which adds the slope between each pair of neighboring bands, for 191 numbers in total. A filter's knee is a change in slope, and the tree models can't easily subtract one number from another on their own, so handing them the slopes directly took the type and order accuracy from 0.74 to 0.86. The type classifier then picks high-pass, low-pass, or band-pass. The picked type decides which order classifier and which cutoff regressors run next, since each type has its own. The cutoff regressors work in octaves, so a 10% miss counts the same at 100 Hz as at 10 kHz, and their answers are kept inside the range seen in training so they're always valid filter settings. The order and the cutoffs together make the spec.

## Updating the diagrams

Edit `filter-id-code-map.drawio` in app.diagrams.net or VS Code, then export each tab again as a PNG with the same name. In the draw.io desktop app, that's File, Export as, PNG, with a zoom of 200% and a border of 24. From a terminal with draw.io installed, this exports the first tab, and changing the page index and file name does the others:

```
draw.io --export --format png --page-index 1 --scale 2 --border 24 --output docs/code-map/1-code-structure.png docs/code-map/filter-id-code-map.drawio
```
