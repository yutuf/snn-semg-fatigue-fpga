# Dataset

## Source (cite this)
**Cerqueira, S., Vilas-Boas, R., Figueiredo, J., Santos, C. (2024).**
*A Dataset of sEMG and Self-Perceived Fatigue Levels for Muscle Fatigue Analysis.*
University of Minho.
- **Zenodo record:** https://zenodo.org/records/13937111
- **DOI:** 10.5281/zenodo.14182446
- **License:** CC-BY-4.0 (free to use with attribution)

## Why the raw data is NOT physically in this zip
The raw EMG is **3.07 GB** — too large to bundle, and it's a permanent, public,
citable dataset. Bundling it would just be a stale copy of a DOI. Instead this
folder gives you the exact, reproducible path to get and preprocess it.

Run these two, in order, on a normal machine:
```
python download_dataset.py      # pulls from Zenodo into ./raw_data  (use --small to skip the 3 GB)
python preprocess.py            # raw CSV -> spike tensors -> preprocessed.pt
```
`preprocess.py` is the **same** code that produced the training/LOSO inputs, not a
re-implementation.

## File manifest (Zenodo record 13937111)
| File | Size | What it is |
|---|---|---|
| `sEMG_data.zip` | 3.07 GB | raw 4-channel EMG, per subject / per trial CSVs |
| `self_perceived_fatigue_index.zip` | 4.9 MB | per-sample fatigue labels (0=Rested … 2=Fatigued) |
| `protocol.xlsx` | 1.0 MB | experimental protocol |
| `Metadata.xlsx` | 11 KB | subject metadata |
| `code.ipynb` | 21 KB | dataset authors' own reference notebook |

The small companion files land in `companion_files_bundled/` when you run the
downloader. (They were not pre-bundled because Zenodo rate-limited the machine
this package was assembled on; your own connection will be fine.)

## How we use it (summary — full detail in ../README_HANDOFF.md §1)
- 13 subjects, 4 EMG channels, **FS = 1259 Hz**.
- 5.0 s windows, 50% overlap → STFT magnitude (n_fft=64, hop=32) → 4×33×T spectrogram.
- Keep only the two extreme classes → **binary** Rested (0) vs Fatigued (1).
- Delta (temporal-difference) encode → boolean spike tensor fed to the SNN.
- Split **subject-independently** (Leave-One-Subject-Out) — no window from a test
  subject is ever seen in training.

## Expected directory layout after download
```
raw_data/
  sEMG_data/
    subject_01/ ... _trial_1.csv, _trial_2.csv, ...
    subject_02/ ...
    ...
  fatigue_labels/
    ..._trial_1.csv, ...
```
Filenames must contain `subject_N` and `trial_M` — the loader matches them by that
pattern (see `preprocess.py: subject_trial()`). `*mvc*` files are intentionally skipped.
