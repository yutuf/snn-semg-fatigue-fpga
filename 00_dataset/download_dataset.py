#!/usr/bin/env python3
"""
Download the sEMG muscle-fatigue dataset used in this project.

Source: Cerqueira et al. (2024), "A Dataset of sEMG and Self-Perceived
Fatigue Levels for Muscle Fatigue Analysis", University of Minho.
Zenodo record 13937111 | DOI 10.5281/zenodo.14182446 | License CC-BY-4.0

Run this on your OWN machine (a normal internet connection — Zenodo does not
block it). It downloads the two zips needed to reproduce training/eval:
  - sEMG_data.zip                     (~3.07 GB)  raw 4-channel EMG per subject/trial
  - self_perceived_fatigue_index.zip  (~4.9 MB)   per-sample fatigue labels
plus the small companion files (protocol, metadata, authors' own code.ipynb).

After it finishes you'll have:
  raw_data/sEMG_data/subject_XX/..._trial_Y.csv
  raw_data/fatigue_labels/..._trial_Y.csv
which is exactly what preprocess.py expects.

Usage:
  python download_dataset.py            # everything, into ./raw_data
  python download_dataset.py --small    # skip the 3 GB raw EMG (labels+meta only)
"""
import os, sys, json, zipfile, glob, urllib.request, shutil

RECORD = "13937111"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
COMPANION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companion_files_bundled")
UA = {"User-Agent": "Mozilla/5.0 (research handoff downloader)"}

def fetch_manifest():
    req = urllib.request.Request(f"https://zenodo.org/api/records/{RECORD}", headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def dl(url, dest):
    print(f"  -> {os.path.basename(dest)} ...", flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

def main():
    small = "--small" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(COMPANION, exist_ok=True)
    rec = fetch_manifest()
    files = {f["key"]: f["links"]["self"] for f in rec["files"]}

    # 1) labels (small, always)
    lbl_key = next(k for k in files if "fatigue" in k.lower())
    lbl_zip = os.path.join(OUT, lbl_key)
    dl(files[lbl_key], lbl_zip)
    lbl_dir = os.path.join(OUT, "fatigue_labels")
    os.makedirs(lbl_dir, exist_ok=True)
    with zipfile.ZipFile(lbl_zip) as z: z.extractall(lbl_dir)
    for nz in glob.glob(os.path.join(lbl_dir, "**", "*.zip"), recursive=True):
        with zipfile.ZipFile(nz) as z: z.extractall(os.path.dirname(nz))
    os.remove(lbl_zip)

    # 2) small companion files, bundled next to the scripts
    for key in ("protocol.xlsx", "Metadata.xlsx", "code.ipynb"):
        if key in files:
            dl(files[key], os.path.join(COMPANION, key))

    # 3) the big raw EMG (skip with --small)
    if not small:
        raw_key = next(k for k in files if k.lower() == "semg_data.zip")
        raw_zip = os.path.join(OUT, raw_key)
        print("  Downloading raw EMG (~3 GB, this takes a while)...")
        dl(files[raw_key], raw_zip)
        raw_dir = os.path.join(OUT, "sEMG_data")
        os.makedirs(raw_dir, exist_ok=True)
        with zipfile.ZipFile(raw_zip) as z: z.extractall(raw_dir)
        for nz in glob.glob(os.path.join(raw_dir, "**", "*.zip"), recursive=True):
            with zipfile.ZipFile(nz) as z: z.extractall(os.path.dirname(nz))
            os.remove(nz)
        os.remove(raw_zip)
    else:
        print("  --small: skipped the 3 GB raw EMG.")

    print("\nDone. Raw data is in:", OUT)
    print("Now run:  python preprocess.py")

if __name__ == "__main__":
    main()
