# Starting Kit

This competition uses code submission. Submit a zip file with this exact entry point at the root:

```text
model.py
```

`model.py` must define:

```python
def predict(input_dir, output_dir):
    ...
```

At evaluation time, Codabench runs your code on MER-PS test data. The test input directory contains a `sample_ids.csv` file and raw EEG/fNIRS `.mat` files under `data/<prediction_subject_id>/`. Do not hard-code the number of evaluation subjects; use `sample_ids.csv`.

Your code must write this exact output file:

```text
predictions.csv
```

The prediction CSV must contain:

```csv
sample_id,valence,arousal
```

Values must be integers on the original MER-PS label scale `[1, 255]`.

`sample_code_submission.zip` is a valid baseline submission. It contains an ASAC-style model checkpoint named `best_model.pt`, loads the test EEG/fNIRS `.mat` files, runs inference, and writes integer `[1, 255]` valence/arousal predictions.

The training script used for the baseline is included as `train_baseline.py`. By default it trains on the full public training split (`test_1` through `test_20`) and validates on `test_21` through `test_24`.
