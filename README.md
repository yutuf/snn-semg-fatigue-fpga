# Energy-Efficient SNN for Wearable sEMG Muscle-Fatigue Detection

Reproducibility artifact for the paper. A spiking neural network (SNN) for surface-EMG
muscle-fatigue detection, taken from raw data all the way to **bit-exact-verified FPGA RTL**.

<!-- After you mint a Zenodo DOI (see below), add its badge here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

## Result at a glance
- **Task:** binary muscle-fatigue detection (Rested vs Fatigued), 4-channel sEMG.
- **Accuracy:** SNN **78.3%** vs iso-architecture CNN **78.0%** macro-F1, pooled 13-fold
  leave-one-subject-out; McNemar p = 0.354 (statistically indistinguishable).
- **Energy:** SNN **~28× lower** than the CNN (analytical Horowitz 45 nm model).
- **Hardware:** full network (conv1→fc1→fc2) implemented in Verilog, **bit-exact verified**
  per-layer and end-to-end, synthesized with an open-source flow (Yosys/nextpnr, iCE40).

## Where to start
| You are… | Open |
|---|---|
| a human reader | [`README_HANDOFF.md`](README_HANDOFF.md) — full guide |
| looking for runnable code | [`08_notebooks/`](08_notebooks/) — Colab notebooks |
| planning the paper | [`07_paper/`](07_paper/) + [`ROADMAP_TO_HPEC.md`](ROADMAP_TO_HPEC.md) |
| everything, indexed | [`MANIFEST.json`](MANIFEST.json) |

## Reproduce
```bash
# software result (Colab, GPU): open 08_notebooks/01_SNN_fatigue_pipeline.ipynb and run
# FPGA verification (needs oss-cad-suite on PATH):
python run_all_tests.py          # -> SUMMARY: 4/4 checks OK (all bit-exact)
```

## Dataset
Cerqueira et al. (2024), *A Dataset of sEMG and Self-Perceived Fatigue Levels for Muscle
Fatigue Analysis*, Zenodo **DOI 10.5281/zenodo.14182446**, CC-BY-4.0. Not bundled (3 GB);
`00_dataset/download_dataset.py` fetches it.

## How to cite
See [`CITATION.cff`](CITATION.cff). Once this repo has a Zenodo DOI (below), cite that.

## Getting a citable DOI (for the paper)
1. Push this repo to GitHub (public).
2. Log into **[zenodo.org](https://zenodo.org)** with your GitHub account → **Settings →
   GitHub** → flip the switch **ON** for this repository.
3. On GitHub, create a **Release** (e.g. tag `v1.0.0`). Zenodo automatically archives it and
   mints a DOI.
4. Copy the DOI into `CITATION.cff`, this README's badge, and the paper.

## License
Code: MIT (see [`LICENSE`](LICENSE)). Dataset: CC-BY-4.0, © its original authors.
