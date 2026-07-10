#!/usr/bin/env python3
"""
Exact preprocessing pipeline: raw sEMG CSV  ->  spike tensors the SNN consumes.

This is the SAME code used to produce the training/LOSO inputs, extracted from
the Colab training cell into a standalone script. Run it after download_dataset.py.

Pipeline (per subject/trial):
  1. Load 4 EMG channels (columns containing '[V]'), z-normalize per channel.
  2. Attach the per-sample fatigue label (nearest-time lookup).
  3. Slide a 5.0 s window (WINDOW = 5*1259 samples), 50% overlap.
  4. Per window: STFT magnitude (n_fft=64, hop=32, Hann) -> 4 x 33 x T spectrogram.
  5. Global max-normalize, keep only the two extreme classes (0=Rested, 2=Fatigued).
  6. Delta (temporal-difference) encode across time -> boolean spike tensor.

Output: X_all  (N, T, 4, 33) bool spikes  and  y_all (N,) in {0,1}, plus subj_arr
for subject-independent (LOSO) splitting.

Requires: numpy, pandas, torch, tqdm. FS=1259 Hz is fixed by the dataset.
"""
import os, glob, re, gc, warnings
import numpy as np, pandas as pd, torch
from tqdm import tqdm
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
SEMG_DIR = os.path.join(BASE, "sEMG_data")
LABEL_DIR = os.path.join(BASE, "fatigue_labels")

FS = 1259
WINDOW = int(5.0 * FS)
STEP = WINDOW // 2

def subject_trial(fp):
    s = re.search(r"subject_(\d+)", os.path.dirname(fp).lower())
    t = re.search(r"trial_(\d+)", os.path.basename(fp).lower())
    return (int(s.group(1)), int(t.group(1))) if s and t else (None, None)

def load_raw(sp, lp):
    df = pd.read_csv(sp)
    cols = [c for c in df.columns if "[V]" in c and not c.strip().startswith("X")]
    if not cols:
        return None, None
    raw = np.nan_to_num(df[cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    tc = df.iloc[:, 0].values
    dl = pd.read_csv(lp)
    lt, ll = dl.iloc[:, 0].values, dl.iloc[:, 1].values
    idx = np.clip(np.searchsorted(lt, tc), 0, len(ll) - 1)
    al = ll[idx].astype(int)
    m, s = raw.mean(0, keepdims=True), raw.std(0, keepdims=True)
    s[s < 1e-6] = 1.0
    return np.nan_to_num((raw - m) / s, nan=0, posinf=0, neginf=0), al

def build():
    semg_files = sorted(f for f in glob.glob(os.path.join(SEMG_DIR, "**", "*.csv"), recursive=True)
                        if "mvc" not in f.lower())
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "**", "*.csv"), recursive=True))
    semg = {subject_trial(f): f for f in semg_files if subject_trial(f)[0]}
    label = {subject_trial(f): f for f in label_files if subject_trial(f)[0]}
    keys = sorted(set(semg) & set(label))
    print(f"{len(keys)} subject/trial pairs found.")

    window_fn = torch.hann_window(64)
    spec_list, ys, subjs = [], [], []
    for (sj, tr) in tqdm(keys, desc="load + STFT"):
        d, lb = load_raw(semg[(sj, tr)], label[(sj, tr)])
        if d is None or len(d) <= WINDOW:
            continue
        starts = list(range(0, len(d) - WINDOW, STEP)); n = len(starts)
        ch = np.stack([d[i:i + WINDOW] for i in starts]).astype(np.float32)
        lls = [int(np.bincount(lb[i:i + WINDOW], minlength=3).argmax()) for i in starts]
        xt = torch.from_numpy(ch).transpose(1, 2).reshape(n * 4, WINDOW)
        st = torch.abs(torch.stft(xt, n_fft=64, hop_length=32, window=window_fn, return_complex=True))
        st = st.view(n, 4, 33, -1).permute(0, 3, 1, 2).contiguous()
        spec_list.append(st); ys += lls; subjs += [sj] * n
        del d, ch, xt, st

    spec_all = torch.cat(spec_list, 0); del spec_list; gc.collect()
    spec_all = torch.nan_to_num(spec_all, nan=0, posinf=0, neginf=0)
    spec_all = spec_all / (spec_all.max() + 1e-8)
    raw_y = np.array(ys); subj_arr = np.array(subjs)

    keep = np.isin(raw_y, [0, 2]); kt = torch.from_numpy(keep)
    spec_all = spec_all[kt]; raw_y = raw_y[keep]; subj_arr = subj_arr[keep]
    y_all = torch.tensor((raw_y == 2).astype(int), dtype=torch.long)   # 0=Rested, 1=Fatigued

    thr = 0.5 * spec_all.std()
    dx = (spec_all[:, 1:] - spec_all[:, :-1]).abs()
    sp = (dx >= thr); del dx; gc.collect()
    X_all = torch.cat([torch.zeros_like(spec_all[:, :1], dtype=torch.bool), sp], dim=1)
    del spec_all, sp; gc.collect()

    print("Total windows:", len(y_all), "| Subjects:", len(np.unique(subj_arr)))
    return X_all, y_all, subj_arr

if __name__ == "__main__":
    X_all, y_all, subj_arr = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed.pt")
    torch.save({"X": X_all, "y": y_all, "subj": subj_arr}, out)
    print("Saved:", out)
