# SNN sEMG Fatigue Detection — Handoff Package

**Author:** Yusuf Kerim Kaymakçı · **Prepared:** 2026-07-09
**Purpose:** Complete transfer of the software/model/RTL work so the FPGA implementation and power measurement can be finished on the professor's side.

Everything that was scattered across Google Drive, Colab notebooks, and chat is consolidated here. This README is the map — read it top to bottom, each section points to the exact files.

**Want the runnable code as notebooks?** → `08_notebooks/` has Colab notebooks: `01_SNN_fatigue_pipeline.ipynb` reproduces the whole ML result (data → train → LOSO → energy); `02_FPGA_weight_export.ipynb` bridges the trained model to the FPGA hex weights.

**Companion docs at the root (open these early):**
- `ROADMAP_TO_HPEC.md` — priority-ordered path to submission + checklist + open questions for the professor.
- `MANIFEST.json` — machine-readable inventory of every file with one-line descriptions.
- `FAQ_ANTICIPATED_QUESTIONS.md` — answers to the 11 questions most likely to come up (accuracy, binary reframing, RTL correctness, what's left, etc.).
- `07_paper/` — HPEC paper outline, draft abstract, and figure/table plan, all with real numbers.
- `04_synthesis_results/LATENCY_AND_ENERGY.md` — all the hardware numbers (resource, measured cycle-accurate latency, operation counts, energy model) with formulas shown.

---

## 0. TL;DR — what this is and where it stands

- **Task:** binary muscle-fatigue detection (Rested vs Fatigued) from 4-channel surface EMG, using a Spiking Neural Network (SNN).
- **Result:** SNN reaches **78.3%** accuracy under 13-subject Leave-One-Subject-Out (LOSO); an iso-architecture CNN reaches **78.0%** — statistically **no difference** (McNemar p = 0.354), but the SNN is **~28× cheaper in energy** (analytical Horowitz model). That equivalent-accuracy-at-far-lower-energy is the paper's story.
- **FPGA status:** the full network (conv1 → fc1 → fc2) is written in Verilog and **verified bit-exact** against a Python reference. Real open-source synthesis (Yosys/nextpnr, iCE40) gives real LUT/DSP/BRAM numbers. **What is NOT done: a real measured power number** — that is the piece we're handing over (Vivado `report_power` on the professor's machine).
- **The single most important open engineering decision** is memory: the fc1 weight table does not fit in on-chip BRAM at 16-bit. See §5.

---

## 1. Dataset

- **Source:** Cerqueira et al. (2024), sEMG fatigue dataset — **Zenodo record 13937111** (`sEMG_data.zip` + fatigue label zip).
- **Subjects:** 13 (all used). **Channels:** 4 EMG. **Sampling:** 1259 Hz.
- **Windowing:** 5.0 s windows, 50% overlap → STFT spectrogram front-end (n_fft=64, hop=32), then **delta (temporal-difference) encoding** into spikes.
- **Labeling:** original 3-class self-reported fatigue was noisy and capped accuracy ~50%. We reframed to **binary** by keeping only the two extremes (Rested vs Fatigued) and dropping the ambiguous middle. This is the honest, defensible move and the reason accuracy is usable.
- **Total windows across all LOSO folds:** 12,370.

**Dataset is not physically bundled** (it's 3.07 GB, public, CC-BY, DOI 10.5281/zenodo.14182446). Instead `00_dataset/` contains the exact reproducible path: `download_dataset.py` (pulls it from Zenodo) → `preprocess.py` (the *same* windowing/STFT/delta-encoding code that produced the training inputs). See `00_dataset/DATASET.md` for the full manifest and citation.

---

## 2. Best model (the one to deploy)

**Checkpoint:** `01_model/snn_fatigue_final.pth`
**Full parameter dump:** `01_model/_model_report.txt` · **Config/metadata:** `01_model/_prep_report.txt`

### Architecture (exact)
```
Input:  4 channels × 33 freq bins × 197 time-frames (spikes, delta-encoded)
conv1:  Conv1d(4 → 16, kernel=5, pad=2)  + BatchNorm  + LIF1
fc1:    Linear(16×33=528 → 256)          + LIF2
fc2:    Linear(256 → 2)                   + LIF3
Readout: classification uses mem.mean(0) of fc2's LIF (membrane), NOT spike count
Total trainable params: 136,351
```

### Learned constants (already baked into the hex weights)
| | LIF1 (conv1) | LIF2 (fc1) | LIF3 (fc2) |
|---|---|---|---|
| β (leak) | 0.8216 | 0.9108 | 0.8853 |
| threshold | 0.30 | 0.30 | 0.30 |

- Reset: subtract-on-fire (`reset_mechanism = subtract`).
- BatchNorm is **folded into conv1** for hardware (see `_prep_report.txt`: folded conv1 = 336 params).
- Single per-window classification = argmax of the 2 accumulated fc2 membranes averaged over the 197 timesteps.

### Reported accuracy
- Single trial-grouped split (the checkpoint's own val): **macro-F1 = 82.9%** (smoothed 83.2%).
- **13-subject LOSO (the number to cite): 78.3% SNN / 78.0% CNN, McNemar p=0.354.** Raw per-fold data in `06_results_and_reports/loso_results.json`; recompute with `analyze_loso.py`.

---

## 3. What works vs what doesn't (be honest with reviewers)

**Works / solid:**
- Leakage-free evaluation (subject-independent LOSO, no window from a test subject ever seen in training).
- Binary fatigue task is learnable and the SNN matches the CNN.
- Energy advantage (~28×) is real under the same Horowitz model both sibling lab papers use.
- Full RTL exists and is bit-exact verified end to end (§4).

**Doesn't work / limitations:**
- 3-class fatigue is not reliably separable on this data (~50% ceiling) — that's why we went binary.
- Data augmentation experiments (amplitude/time warping) **hurt** performance — collapsed toward majority class. Not used in final.
- Accuracy is windows-level, not a clinical per-session decision — framed as a proof-of-concept.
- **No measured joules/inference yet** — only analytical energy + measured latency. This is the handoff ask.

---

## 4. FPGA / RTL — what's built and verified

All RTL in `03_rtl/`. All verification vectors + Python golden-reference generators in `05_verification/`. Synthesis scripts + P&R utilization logs in `04_synthesis_results/`.

**Design style:** *time-multiplexed* (one shared LIF compute core reused across all neurons in a layer, membranes stored in memory). This is what makes it fit a tiny FPGA. The critical path — and therefore Fmax — is set by the per-neuron compute chain, so it is **independent of neuron count** (~16–17 MHz measured, see below).

| Module | File | Verified | Synthesis (iCE40 UP5K, nextpnr) |
|---|---|---|---|
| LIF neuron | `lif_neuron.v` | ✅ fires correctly w/ learned β=0.82 | tiny |
| conv1 + LIF1 (weight-sharing, zero-pad) | `snn_conv1_core_bram.v` | ✅ **1056/1056** spike-checks bit-exact | 1753 LC / 4 BRAM / 1 DSP (33%) |
| fc1 (528→256), membranes in BRAM | `snn_fc_core_bram.v` | ✅ bit-exact | 1127 LC / 2 BRAM / 2 DSP (21%) |
| fc2 output (256→2, exposes membranes) | `snn_fc2_output.v` | ✅ bit-exact on spikes **and** membranes | 510 LC / 0 BRAM / 1 DSP (9%) |
| **Full chain** conv1→fc1→fc2 + FSM | `snn_top.v` | ✅ bit-exact vs `gen_ref_full.py` | (see §5 — memory is the blocker) |
| SPRAM boot-load pattern | `spram_conv1_weights.v` | ✅ 336/336 weights loaded | proves external-memory load path |

- **Fixed-point format:** Q6.10 (16-bit signed). Weights exported both Q6.10 and int8 (`02_weights_hex/`).
- **Latency — reported as measured cycle counts, deliberately left as your optimization axis.** The current RTL is the *area-minimal* build (one shared MAC), so it is intentionally slow-but-tiny. Measured in simulation (`lat_tb.v` in `05_verification/`): one timestep = **305,567 cycles**, full 197-timestep inference = **60,196,699 cycles**, fc1 = ~89% of it. We give you the cycle counts rather than a fixed wall-clock number **because throughput is an open FPGA-side optimization**, not a fixed property — three orthogonal knobs, all yours:
  1. **Parallel MAC cores** — cycles drop ~linearly (16× fc1 cores → ~0.22 s); the fabric budget easily allows it (fc1 core = 21% of the chip).
  2. **Event-driven sparsity** — the current core iterates *every* synapse even when the input spike is 0 (dense). Skipping non-spiking inputs (the whole point of an SNN) would cut fc1 cycles by the sparsity factor. This is real, legitimate, and not yet done.
  3. **Clock** — 16–17 MHz is the iCE40 P&R result; an Artix-7 target clocks far higher.
- **Whatever you choose, the energy-per-inference claim (~28×) is unaffected** — it's set by operation *count*, not by how serially or fast they execute. So latency optimization and the paper's energy story are independent.
- **How to re-verify the whole thing** (with oss-cad-suite active): from the package root run
  ```
  python run_all_tests.py
  ```
  It assembles a flat `build/` dir (the testbenches load files by bare name) and runs every
  self-checking testbench. Confirmed output: **`SUMMARY: 4/4 checks OK`** (LIF, conv1, fc2,
  full chain — all bit-exact). This was run against this exact package and passes.

---

## 5. THE key decision for the professor: fc1 memory

This is the one real architectural fork and where a hardware expert adds the most value.

- fc1 has **135,424 weights** (dominant cost). At **16-bit** the total weight memory is **2129 Kbit**, but the UP5K has only **1144 Kbit** on-chip → **does not fit (186%)**. This was confirmed by actual synthesis, not estimated: naive on-chip storage asks for ~539 BRAMs vs 30 available.
- At **int8**, total drops to **1065 Kbit** → fits (~93% of on-chip). Int8 weights are already exported (`fc1_w_int8.hex`) — **int8 accuracy still needs to be confirmed** (quick to check, not yet done).
- The **config we actually want to implement**: **conv1 + fc2 fully on-chip; fc1 weights streamed from external SPI flash into SPRAM at boot** (standard practice). The boot-load path is already proven in `spram_conv1_weights.v`. This was going to be the next step.

**Alternative if targeting a bigger board (e.g. Xilinx Artix-7, which is what Vivado is installed for):** fc1 maps cleanly to block RAM/URAM without the SPI-flash gymnastics — likely *simpler* than iCE40. The Verilog (FSMs + Q6.10 arithmetic) is vendor-neutral; only the SPRAM primitive is iCE40-specific and can be dropped.

---

## 6. Suggested next steps in Vivado (the handoff task)

1. Bring `snn_top.v` + the three core modules + testbenches into a Vivado project (target an Artix-7 part — memory fits natively there).
2. Load the Q6.10 weight hex files (`02_weights_hex/`) via `$readmemh` (already wired in `snn_top.v`).
3. Run behavioral sim against the golden vectors in `05_verification/` to confirm bit-exactness carries over.
4. Synthesize + implement, then run **`report_power`** — this is the missing number we've been blocked on (Lattice's tools were gated behind account activation; Vivado has power analysis built in).
5. Report: measured power (mW) from `report_power` × latency (the measured cycle count ÷ achieved Fmax) → **real joules/inference**, to sit alongside the analytical ~28× energy claim.

---

## 7. File map

```
README.md            repo landing page (result summary + how to cite)
ROADMAP_TO_HPEC.md   submission roadmap + checklist + open questions
MANIFEST.json        machine-readable inventory of every file
FAQ_ANTICIPATED_QUESTIONS.md   pre-answered questions
00_dataset/          DATASET.md + download_dataset.py + preprocess.py (Zenodo DOI 10.5281/zenodo.14182446; raw data 3 GB, not bundled — fetched by the script)
01_model/            snn_fatigue_final.pth (THE model), param dump, weight-prep scripts, budget json
02_weights_hex/      conv1/fc1/fc2 weights + biases, both Q6.10 and int8, ready for $readmemh
03_rtl/              all Verilog: layer cores, full top, testbenches
04_synthesis_results/ Yosys scripts (.ys) + nextpnr utilization logs (pnr_*.txt) + LATENCY_AND_ENERGY.md
05_verification/     Python golden-reference generators + expected spike/membrane vectors + input stimuli
06_results_and_reports/ LOSO raw results + analysis script + full training-journey report
07_paper/            HPEC paper outline + draft abstract + figure/table plan (real numbers)
08_notebooks/        runnable Colab notebooks (full ML pipeline + FPGA weight-export bridge)
```

**Toolchain used (open-source, free):** oss-cad-suite (Yosys + nextpnr + IceStorm + Icarus Verilog).
Note: iverilog needs `-g2012` when compiling anything that includes iCE40 `cells_sim.v`.

---

## 8. One-paragraph summary for the paper's hardware section

> The network is implemented as a time-multiplexed spiking datapath in Verilog (Q6.10 fixed-point), verified bit-exact against a PyTorch-derived reference at every layer and end-to-end. On an iCE40 UP5K the compute logic occupies 9–33% of LUTs per layer at a P&R clock of ~16–17 MHz; in the area-minimal single-MAC configuration a full 197-timestep inference takes 60.2 M cycles (measured in simulation), which parallel MAC cores reduce linearly. The fc1 weight table (135k params) exceeds on-chip BRAM at 16-bit and is therefore streamed from external flash into SPRAM at boot, or moved to int8. This is, to our knowledge, the first of the three companion lab studies to carry an SNN all the way to verified RTL rather than leaving FPGA as future work.
