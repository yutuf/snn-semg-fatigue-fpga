# HPEC Paper — Outline & Skeleton

A results-first distillation of `../06_results_and_reports/training_journey_report.md` into
IEEE conference structure. **All numbers below are real** unless tagged `[TODO]`. Expand each
section into prose against the confirmed venue's template and page limit.

**Working title:**
*Energy-Efficient Spiking Neural Networks for Wearable sEMG Muscle-Fatigue Detection: From Leakage-Free Evaluation to Verified FPGA RTL*

---

## Abstract
See `ABSTRACT.md` for a paste-ready draft. Core claims: first SNN for sEMG *fatigue*;
matches an iso-architecture CNN (78.3% vs 78.0%, McNemar p=0.354) at **~28× lower energy**;
carried to bit-exact-verified FPGA RTL with real synthesis — unlike prior SNN-sEMG work that
stops at simulation.

## I. Introduction
- Wearable muscle-fatigue monitoring matters (injury risk, rehab, ergonomics); must be
  ultra-low-power to run on-body.
- SNNs promise event-driven, low-energy inference; neuromorphic sEMG work exists **but only
  for gesture recognition** — nobody has applied SNNs to *fatigue*.
- Contributions: (1) first SNN fatigue classifier on the Cerqueira 2024 dataset; (2) rigorous
  leakage-free LOSO showing accuracy parity with a CNN at ~28× less energy; (3) a complete,
  **bit-exact-verified** FPGA RTL implementation with real synthesis + measured latency —
  closing the "FPGA is future work" gap the companion studies share.

## II. Related Work
Three buckets (source: report §7):
- **Same dataset:** Cerqueira et al. 2024 provides only classical spectral fatigue metrics,
  no ML classifier → we are the first learned classifier on it.
- **SNN + sEMG:** all prior work is gesture/movement (incl. the 5K-LUT FPGA gesture SNN, 83%
  on NinaPro — our methodological template). None target fatigue.
- **Companion lab papers (same program):** apnea-ECG SNN and STDP gesture SNN — same energy
  methodology, both leave FPGA as future work. Explicit differentiation table (report §7.6):
  our work uses fatigue as the *actual label* on *real* data and is the only one with verified
  hardware.

## III. Dataset & Preprocessing
- Cerqueira 2024, Zenodo DOI 10.5281/zenodo.14182446, 13 subjects, 4-ch sEMG, 1259 Hz, CC-BY.
- 5.0 s windows, 50% overlap → STFT (n_fft=64, hop=32) → 4×33×197 spectrogram → delta
  (temporal-difference) spike encoding.
- Binary reframing: keep the two extreme classes (Rested/Fatigued), drop the ambiguous middle
  — justified by the 3-class ~50% ceiling being *label noise*, not model capacity (report §3.2).

## IV. SNN Model & Training
- Architecture: Conv1d(4→16, k5, pad2)+LIF → FC(528→256)+LIF → FC(256→2)+LIF; classification
  from the output LIF membrane averaged over 197 timesteps.
- Surrogate-gradient training (snnTorch, fast-sigmoid); learned β={0.8216,0.9108,0.8853}, thr 0.30.
- Delta encoding + L1 spike-rate regularization drive the energy advantage; K=5 temporal
  smoothing recovers accuracy at zero energy cost (report §4.3).
- Iso-architecture CNN baseline (same shapes, ReLU instead of LIF) for a fair energy comparison.

## V. Hardware Implementation (the differentiator)
- Time-multiplexed spiking datapath, Q6.10 fixed-point, one shared MAC + LIF core per layer,
  membranes in BRAM.
- **Bit-exact verification** against a Python Q6.10 reference — per layer AND full chain
  (methodology + exact spike/membrane match counts, report §8.3).
- Resource (nextpnr, iCE40 UP5K): conv1 33% LUT/4 BRAM/1 DSP; fc1 21%/2/2; fc2 9%/0/1. Fmax ~16–17 MHz.
- Latency (measured in sim): 305,567 cyc/timestep, 60.2M cyc/inference; report as cycle counts,
  frame wall-clock as an optimization axis (parallelism, event-driven sparsity, clock).
- Memory finding: fc1's 16-bit weight table needs 539 BRAM vs 30 available → int8 or external
  SPRAM/flash. Quantified by a real synthesis attempt, not estimated (report §8.3, finding 6).

## VI. Energy Analysis
- Horowitz 45 nm model: MAC = 4.6 pJ, AC = 0.9 pJ (same as companion papers).
- 28.81 M ops/inference; CNN dense ≈ 132 µJ; SNN spike-driven + sparse → **~28× lower**
  (stage-by-stage SynOps 9.9×→28.8×, report §4.3).
- `[TODO: measured power]` from Vivado `report_power` → absolute joules/inference. This is the
  one number that upgrades the energy claim from analytical to silicon-grounded.

## VII. Results
- Table: pooled 13-fold LOSO — SNN 78.3% / CNN 78.0% macro-F1; McNemar p=0.354 (n.s.).
- Per-fold range 64.9%–89.5% (genuine inter-subject variability, 13-subject cohort).
- Sensitivity/specificity table (report §4.4). Interpretation: *parity accuracy, fraction of
  the energy* — the reviewer-proof framing, not an accuracy-win overclaim.

## VIII. Limitations & Future Work
- Window-level (not per-session clinical) decisions; small cohort; self-reported labels.
- Energy is analytical pending `[TODO: measured power]`.
- fc1 external-memory streaming loader (SPI→SPRAM) specified but not built; latency not yet
  optimized (event-driven core is the biggest available win). Report §10.

## IX. Conclusion
First SNN for sEMG fatigue; CNN-parity accuracy at ~28× less energy; and — uniquely among the
companion studies — carried to verified FPGA RTL with real synthesis, positioning it as a
concrete step toward on-body neuromorphic fatigue monitoring.

---

### Source-material map (where each section's raw content lives)
| Paper section | Source in report | Data files |
|---|---|---|
| III Dataset | §2 | `00_dataset/DATASET.md` |
| IV Model | §3, §8.1 | `01_model/` |
| V Hardware | §8.3 | `03_rtl/`, `04_synthesis_results/`, `05_verification/` |
| VI Energy | §4.1–4.3 | `04_synthesis_results/LATENCY_AND_ENERGY.md` |
| VII Results | §4.4 | `06_results_and_reports/loso_results.json` |
| II Related work | §7 | report §7.1–7.6 |
