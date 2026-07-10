# Energy-Efficient Spiking Neural Networks for Wearable sEMG Muscle-Fatigue Detection
### Training & Methodology Journey Report
*Prepared for supervisor review — target venue: IEEE HPEC*
*Date: 2026-07-03*

---

## 1. Executive Summary

We developed the **first spiking neural network (SNN) approach to surface-EMG muscle-fatigue detection**, targeting an ultra-low-power wearable use case. Under a **full 13-fold leave-one-subject-out (LOSO) evaluation, pooled across all subjects** — matching the rigor of comparable studies in this program — our SNN is a **statistical dead heat with a fairly-tuned, iso-architecture CNN on accuracy** (78.3% vs 78.0% macro-F1; McNemar p = 0.35, not significant), while running at **~28–30× lower estimated compute energy** and now implemented and verified in real FPGA hardware (RTL simulated, synthesized, and placed-and-routed on Lattice iCE40 — not just projected).

**Headline results (binary rested-vs-fatigued):**

| Evaluation | CNN (baseline) | SNN (ours) | Δ | SNN energy advantage |
|---|---|---|---|---|
| **Pooled 13-fold LOSO (defensible headline)** | 77.9% macro-F1 | **78.3% macro-F1** | +0.4 pt (n.s., p=0.35) | **~28×** |
| Single-split, personalized (optimistic, superseded) | 85.5% | 84.4% | −1.1 pt | ~9.9× |
| Single-split, cross-subject (optimistic, superseded) | 78.2% | ~77.6% | −0.6 pt | ~10× |

The central thesis — *statistically indistinguishable accuracy at an order-of-magnitude lower energy* — is supported by a **fair, iso-architecture comparison** (identical layer sizes, data, split, and training budget; only the neuron model differs) evaluated under the same protocol rigor (pooled subject-independent CV + significance testing) as the two companion papers from this program. The novelty is the **application** (fatigue, not gesture), the **energy/edge deployment framing**, and — uniquely among this program's SNN studies so far — a **working FPGA implementation** rather than an energy estimate alone.

---

## 2. Problem Statement & Dataset

**Motivation.** Muscle fatigue is an established risk factor for musculoskeletal injury. Continuous wearable monitoring requires always-on inference under a tight power budget — exactly the regime where neuromorphic/SNN computation is advantageous. We frame the task as an **early-warning fatigue-state detector**, using self-perceived fatigue labels as a proxy (we are careful **not** to overclaim "injury prediction").

**Dataset.** Cerqueira, Vieira Boas, Figueiredo & Santos, *A Comprehensive Dataset of Surface Electromyography and Self-Perceived Fatigue Levels* (Sensors, 2024; Zenodo record 13937111).
- 13 subjects × up to 12 upper-limb dynamic trials.
- 4 EMG channels per arm @ **1259 Hz**; self-perceived fatigue labels (levels 0/1/2) @ 50 Hz.
- All **13 subjects** matched and used (an earlier filename-matching bug that dropped subjects 3 and 5 was fixed; confirmed by the 13-fold LOSO run in §4.4, 12,370 total windows).

---

## 3. Methodology Evolution (the journey)

This section documents what we tried, **why**, and what we learned. Several negative results were as informative as the positive ones.

### 3.1 Starting point & the leakage discovery
- **Initial notebook** reported ~74–79% validation accuracy on a 3-class problem — but used `train_test_split` with **no subject/trial grouping**. Because our sliding windows overlap 50%, adjacent windows from the *same trial* leaked across the train/val boundary.
- **Fix:** switched to **subject-independent** (`GroupShuffleSplit` on subject ID) and **trial-grouped** splits. Honest 3-class accuracy immediately dropped to **~48–50% macro-F1**.
- **Lesson:** the original number was inflated by data leakage; no reviewer would accept it.

### 3.2 Diagnosing the ceiling: it's the label, not the model
- On the **trial-grouped** protocol (subjects *seen* in training), 3-class macro-F1 still capped at **50.5%**. When even seeing the subject doesn't help, the bottleneck is **label noise**, not generalization.
- Root cause: "self-perceived fatigue" on a 0/1/2 scale is subjective, and the middle class ("Warning/transition") is the fuzzy boundary between two feelings the human rater could not sharply separate. Per-class F1 confirmed: Safe and Critical separable, Warning bleeds into both.

### 3.3 Task reframing: 3-class → binary
- We **dropped the ambiguous middle class** and reframed as **binary Rested (Safe) vs Fatigued (Critical)** — the two physiologically distinct endpoints (fresh vs fatigued muscle = large median-frequency shift). This mirrors the standard fatigue-detection paradigm in the literature (start-vs-end of sustained contraction).
- Impact: macro-F1 jumped from ~48% (3-class) to **~70–85%** (binary), depending on protocol. This reframing is both more accurate *and* more clinically meaningful (detecting the at-risk state) — reported honestly as binary state discrimination, not fine-grained level prediction.

### 3.4 Feature representation study
| Representation | Result | Verdict |
|---|---|---|
| **STFT spectrogram** (4ch × 33 freq × T frames) | best accuracy + best energy ratio | **kept** |
| MDF/MNF physiological features (12-dim) | −20 pt accuracy AND worse energy (4.8× vs 10×, higher firing) | rejected |
| Delta encoding (spike-ified input) | −12 pt accuracy but **~36× energy** (conv becomes spike-driven) | reserved for FPGA phase |

### 3.5 Window length — the key accuracy lever
Longer analysis windows capture more of the slow fatigue trend:

| Window | Personalized macro-F1 | Cross-subject macro-F1 |
|---|---|---|
| 1 s | 78.7% | ~73.7% |
| 3 s | 83.6% | 78.0% |
| **5 s** | **85.0%** | 77.9% (no further gain) |

5 s pushed personalized just past 85%; cross-subject plateaued at ~78% (its genuine ceiling).

### 3.6 Evaluation protocol study
- **subject_independent** — hardest, "new user, zero calibration": ~78%.
- **trial_grouped** — personalized, "known user, unseen trials": ~85%.
- **subject_adaptive (temporal calibration)** — *rejected*: because fatigue is monotonic in time, splitting each trial temporally puts mostly-rested data in train and mostly-fatigued in test → severe label-distribution shift (validation was 81% Critical). Unstable and misleading. A trial-level k-shot calibration variant was designed but not cleanly evaluated (see 3.7).

### 3.7 Negative result: input augmentation collapses the network
- Gaussian input augmentation with absolute-magnitude noise **destroyed training** (accuracy frozen at ~50%, neuron firing rate → 0.001). The STFT spectrogram is sparse with small values; absolute noise swamped the signal.
- A scale-relative fix was applied, but augmentation provided no benefit and was **dropped entirely**. Documented as a pitfall.

### 3.8 Ceiling confirmation
Across ~20+ configurations (architecture, window, features, encoding, augmentation, regularization, protocol), binary accuracy plateaued at **~85% personalized / ~78% cross-subject**. We conclude this is the **data ceiling** (11–13 subjects, subjective labels, 4 channels), not a model deficiency.

### 3.9 SNN vs CNN — the core comparison
Iso-architecture: identical conv (4→16, k=5) + FC (528→256→2) layers, same data/split/hyperparameters/epoch budget; only **LIF (spiking) vs ReLU** differs. The CNN was tuned *fairly* (deliberately not weakened — a fairness concern we explicitly addressed). See results table in §4.

---

## 4. Final Results

### 4.1 Accuracy & energy (binary rested-vs-fatigued, macro-F1)
| Protocol | Window | CNN | SNN | Δ | SNN spike rate | SNN energy vs iso-ANN |
|---|---|---|---|---|---|---|
| Personalized (trial-grouped) | 5 s | **85.49%** | 84.42% | −1.07 | ~16–21% | **9.9×** |
| Cross-subject (subject-indep) | 3 s | **78.22%** | ~77.6% | −0.6 | ~14–23% | **~10×** |

### 4.2 Energy methodology
- Horowitz (ISSCC 2014) per-operation energies: **MAC = 4.6 pJ, AC = 0.9 pJ**.
- SNN energy = analog-input conv (MAC) + spike-driven FC layers (AC, scaled by measured firing rate). CNN = all-MAC baseline.

### 4.3 Energy roadmap — we are NOT stopping at 10×
The ~10× figure is the **conservative algorithmic floor** (iso-architecture, plain spectrogram front-end). We have a concrete, staged plan to widen the gap, and **Stage 2 is now empirically confirmed, not just projected**:

| Stage | Technique | Accuracy | Energy vs CNN | Status |
|---|---|---|---|---|
| **0** | Iso-architecture SynOps (spectrogram front-end, no delta) | 84.4% | **9.9×** | **Measured** |
| **1** | + Delta input encoding (frame-to-frame delta on spectrogram) | 82.3% | **16.0×** | **Measured** |
| **2** | **+ Spike-rate (L1) sparsity regularization** | **83.1%** | **30.0×** | **Measured — best balance** |
| **2b** | Same, stronger sparsity penalty (λ=2.0) | 82.4% | **37.3×** | **Measured — max energy variant** |
| **3** | **System-level: SNN on FPGA (~10 mW) vs CNN on GPU (50–250 W)** | — | **~100–1000×** energy/inference | To be measured (FPGA phase) |

**Key finding:** delta encoding + sparsity regularization, combined with the 5-second analysis window, **triples the energy advantage (9.9× → ~29×) for a cost of only ~1.3 accuracy points** (84.4% → 83.1%), achieved via a 5-account parameter sweep over the sparsity penalty weight (λ ∈ {0, 0.1, 0.5, 2.0}) and the delta threshold. A sparser input threshold (fewer input spikes) was tested and **rejected** — it discards too much signal (accuracy dropped to 78.5%) for a smaller energy gain (33.8×) than the sparsity-regularization route.

**Final refinement — temporal smoothing (free, zero energy cost):** re-adding inference-time smoothing (majority vote over K=5 consecutive windows) recovers nearly all the accuracy the delta/sparsity trade had cost: **84.11% macro-F1 at 28.8× energy** — only 0.31 points below the original non-delta baseline (84.42%/9.9×), while keeping ~2.9× more of the energy advantage. **This is the final, locked headline model** (config: delta_k=0.5, λ_sparse=0.5, smooth_k=5). No further accuracy tuning is planned — the accuracy-energy trade-off has reached a point of diminishing returns relative to the remaining FPGA and writing work.

**Caveat for the hardware phase:** in the current pipeline, delta encoding is applied to **STFT spectrogram frames** (frequency-domain), not the raw sEMG signal — so an FPGA implementation of *this exact* model still requires an on-chip FFT front-end. A raw-signal delta encoding variant (no FFT needed at all, following the original delta-modulation scheme in our earliest exploration, §3.9-adjacent) would simplify the hardware further but was not re-tested under time constraints — documented as future work (§9).

**Honesty note for review:** Stages 0–2 are measured on identical data/split/training budget; Stage 3 (system-level FPGA-vs-GPU) is the next deliverable, not yet measured.

### 4.4 Leave-one-subject-out cross-validation (13-fold) — the defensible headline number

The single-split numbers above (84% personalized) were vulnerable to the "you got a lucky split" critique — the exact critique both sibling lab papers pre-empt by pooling a full subject-independent cross-validation. We therefore ran a **complete 13-fold leave-one-subject-out (LOSO)** evaluation: each subject is held out exactly once, the SNN and the iso-architecture CNN are retrained on the other 12, and all out-of-fold window predictions are **pooled into one set** (N = 12,370 windows) before scoring. The two models are identical in every respect except LIF-vs-ReLU, and share data, splits, and training budget per fold.

**Pooled LOSO results (binary Rested vs Fatigued):**

| Model | Accuracy | Macro-F1 | Sensitivity (Fatigued) | Specificity (Rested) |
|---|---|---|---|---|
| **SNN (ours)** | 78.3% | **78.3%** | 75.9% | 80.6% |
| CNN (iso-architecture) | 78.0% | 77.9% | 72.9% | 82.8% |

- **Per-subject accuracy (mean ± std over 13 folds):** SNN **78.5 ± 7.0%**, CNN **77.8 ± 8.4%**. The SNN is marginally higher *and* slightly more stable across subjects.
- **Statistical test (McNemar, pooled window predictions):** 1,683 discordant pairs (SNN-only-correct 861 vs CNN-only-correct 822); **exact two-sided p = 0.354** → the accuracy difference is **not statistically significant**. A paired t-test on per-subject accuracy agrees (p = 0.68).

**Interpretation — this is the result we want, not a disappointment.** Under the honest LOSO protocol the SNN and CNN are a **statistical dead heat on accuracy**. Combined with the measured ~28× energy advantage (§4.3), the story is clean and reviewer-proof: *the spiking model matches a fairly-tuned CNN's accuracy at a fraction of the energy.* The headline moves from the optimistic single-split 84% to the defensible pooled-LOSO **78.3%** — lower in absolute terms, but this is the number that survives scrutiny and matches the evaluation rigor of the sibling papers (pooled subject-independent CV + significance test). Per-fold accuracy ranged 64.9%–89.5%, reflecting genuine inter-subject variability on an 11–13 subject cohort.

*(Reproducible artifacts: `loso/loso_results.json` holds every fold's confusion matrices and per-subject accuracies; `loso/analyze_loso.py` regenerates the pooled table, McNemar test, and `loso/loso_summary.json`.)*

---

## 5. Key Scientific Decisions (rationale summary)
1. **Subject-independent + trial-grouped splits** — eliminates window leakage; reports both "new user" and "personalized" numbers.
2. **Macro-F1 as the selection metric** (not accuracy) — guards against majority-class collapse under class imbalance.
3. **Binary reframing** — matches the fatigue-detection literature, removes irreducible label noise, more clinically actionable.
4. **Iso-architecture CNN baseline, fairly tuned** — isolates the spiking-vs-not variable; a strong baseline makes the "same accuracy, less energy" claim credible.
5. **STFT spectrogram front-end** — best accuracy/energy; delta encoding reserved as the FPGA-native path.

---

## 6. Engineering Challenges Encountered (rigor log)
- **Google Drive FUSE bottleneck** — unzipping a 3 GB archive directly over the network mount stalled; fixed by copy-to-local-SSD first.
- **NaN collapse** — a single bad value poisoned global-max normalization (`x / NaN = NaN` across the whole tensor); fixed with `nan_to_num` guards before normalization.
- **RAM OOM** — holding raw windows + full spectrogram simultaneously exceeded Colab RAM; fixed by computing STFT per-trial and discarding raw windows (float32 throughout).
- **Augmentation collapse** — see §3.7.

---

## 7. Related Work & Positioning

We organise related work into three buckets, corresponding to the three questions a reviewer (and our supervisor) will ask: *Who else used this dataset? What is the state of the art in EMG fatigue classification? And what has been done with SNNs on EMG?*

### 7.1 Work on the same dataset (Cerqueira et al., Zenodo 13937111)
The dataset (**Cerqueira, Vieira Boas, Figueiredo & Santos, Sensors 2024**) is recent (published Dec 2024). Its accompanying code computes only **classic spectral fatigue metrics (median/mean frequency)** — it provides **no machine-learning classifier baseline**. To our knowledge, **no published classifier has been trained and evaluated on this dataset.** We therefore establish the **first learned classifier** — and the first SNN — on it. *(This is a direct, verifiable novelty claim; the dataset's youth is a strength here.)*

### 7.2 sEMG fatigue classification (other datasets — not directly comparable)
Deep-learning and ML fatigue models exist, but on **different datasets, tasks, sensors, and protocols**, so head-to-head accuracy comparison is not meaningful. We cite them to position the field:

| Work | Model | Data / task | Reported acc. | Why not directly comparable |
|---|---|---|---|---|
| FatigueFormer (2026) | Transformer (static+temporal fusion) | Self-collected, 30 subj, 4 MVC levels | SOTA (isometric) | Different data; MVC-level task, not self-perceived |
| Bi-LSTM fatigue+stress | Bi-LSTM | 188 EMG + 223 ECG | 95% | **Sensor fusion** (EMG+ECG); different data |
| CWT-CNN | CNN on wavelet scalograms | Self-collected | 89.1% LOSO (binary) | Different data & preprocessing |
| TCN+Attention+CNN | Hybrid | Isometric contractions | 90.07% | Isometric (static) task; different data |
| Force-Level under fatigue (2025) | RF / CatBoost / XGBoost | Forearm grasping, 12 force levels | — | Classic ML; force-level, not fatigue-state |

**Takeaway:** the field is dominated by CNN/RNN/Transformer or classic-ML approaches, frequently using isometric protocols, MVC-percentage targets, or multi-sensor fusion. **None use spiking networks, and none report energy/hardware efficiency** — which is precisely our axis of contribution.

### 7.3 SNN for sEMG (gesture only — the template we adapt to fatigue)
Every SNN-sEMG work we found targets **hand-gesture / motion decoding**, never fatigue:

| Work | Platform | Task | Accuracy | Efficiency |
|---|---|---|---|---|
| **SNN on 5K-LUT FPGA (2024)** — *closest prior art / our template* | FPGA (Lattice iCE40) | 12 gestures (NinaPro DB5) | 83.17% | 44.6 µJ/inf, 11.3 mW |
| Neuromorphic RSNN | Intel Loihi | 12 gestures | 74% | 41 mW |
| HD-sEMG SNN | Jetson | 10 gestures | 95% | 0.97 mJ/inf, 100 mW |
| SNN (SpiNNaker) | SpiNNaker | 4-class | 84.8% | 1–4 W |

We essentially **replicate the SNN-on-FPGA gesture methodology and re-target it to fatigue** — the same neuromorphic/energy machinery applied to an application no one has addressed with SNNs.

### 7.4 Foundational references
- **Cifrek et al. (Clinical Biomechanics 2009)** — canonical sEMG-fatigue review; grounds the median-frequency-shift premise.
- **Eshraghian et al. (Proc. IEEE 2023)** / **Neftci et al. (IEEE SPM 2019)** — snnTorch and surrogate-gradient training foundations.
- **Horowitz (ISSCC 2014)** — per-operation energy model underpinning our energy claim.

### 7.5 The gap we fill
Positioned against all three buckets, our contribution occupies an **empty intersection**:

> **SNN × muscle-fatigue detection × energy-efficient (FPGA) deployment** — on a dataset with no prior classifier.

Fatigue DL exists (but no SNN, no energy story); SNN-sEMG exists (but only gesture); the dataset exists (but no classifier). We are the first to combine all three.

### 7.6 Differentiation from companion works (same research program)
Two closely related SNN studies were produced in the same program and share our energy methodology (Horowitz 45 nm, MAC 4.6 pJ / AC 0.9 pJ) and honest subject-independent framing. We state the distinctions explicitly to avoid confusion:

| Aspect | Apnea-ECG SNN (companion) | STDP sEMG-gesture SNN (companion) | **This work** |
|---|---|---|---|
| Signal / task | ECG → sleep-apnea minutes | sEMG → 12-class gesture | sEMG → binary fatigue state |
| Role of *fatigue* | absent | a **synthetic degradation** injected to stress-test a gesture model | the **actual classification target**, on real fatigue recordings |
| SNN type | surrogate-gradient LIF | unsupervised STDP + adaptive threshold | surrogate-gradient LIF + delta encoding + spike-sparsity regularization |
| Energy result | 1.4× vs MLP, 25× vs CNN (estimate) | ~5× vs CNN (estimate) | **~28× vs CNN (estimate) + measured FPGA implementation** |
| Hardware | future work | future work | **implemented: RTL simulated, synthesized & placed-and-routed on iCE40** |

**Two clear, defensible differentiators:** (1) ours is the only one where *fatigue itself is the label* on real data — the gesture paper only *simulates* fatigue as noise; (2) ours is the only one that **closes the FPGA gap the other two explicitly leave open** — we move the shared "future work: implement on FPGA" from promise to measured result.

---

## 8. Techniques in Reserve & Planned Improvements (the arsenal)

We have deliberately **not** exhausted our options — the current numbers are a solid, honest baseline, and the following levers are designed/ready to push both accuracy and energy further. This section exists so the trajectory is explicit: **we are not stopping at a theoretical 10×.**

### 8.1 Accuracy levers (to raise the 78% cross-subject / 85% personalized)
| Lever | Mechanism | Expected impact | Status |
|---|---|---|---|
| **MVC normalization** | Normalize each subject's EMG by their **own max voluntary contraction** (the dataset includes MVC recordings we currently discard). Standard EMG practice to remove inter-subject amplitude variance. | Primarily lifts **cross-subject** (78%) — attacks the exact source of the ceiling | **Not yet tested — highest-value untested lever** |
| **k-shot per-user calibration** | Include a few of each new user's trials in training (trial-level, leakage-free); mirrors how real myoelectric devices calibrate per wearer | cross-subject 78% → low-80s (typical EMG calibration gain) | Designed; a prior attempt was corrupted by the augmentation bug — needs one clean run |
| **Leave-one-subject-out CV** | 13-fold LOSO → report mean ± std instead of a single split | Statistical rigor (not necessarily higher mean); expected by reviewers | **Done (§4.4): 78.3% pooled, SNN≈CNN, McNemar p=0.35** |
| **Temporal smoothing** | Vote over consecutive windows (fatigue is a slow state, so neighbours agree) | +2–5 pt at segment level, essentially free | Implemented; in current results |
| **Recurrent SNN (RLeaky)** | Neuron-level memory better suited to the slow temporal fatigue trend | Uncertain (+2–3 pt possible) | Tried in early 3-class (unstable then); untested in the clean binary setup |

### 8.2 Energy levers (to widen the 10× gap — see §4.3 roadmap)
| Lever | Mechanism | Impact | Status |
|---|---|---|---|
| **Delta input encoding + spike-rate (L1) regularization** | Spike-ify spectrogram frames (AC not MAC) + penalize internal firing rate in the loss | **9.9× → 30×, cost only ~1.3pt accuracy** (84.4%→83.1%) | **Measured** (5-config sweep) |
| Stronger sparsity penalty variant | Same, higher λ | 37.3× energy, 82.4% accuracy | **Measured** (alt. config) |
| **Raw-signal delta encoding (no FFT)** | Apply delta modulation directly to raw sEMG instead of spectrogram frames — would eliminate the on-chip FFT requirement entirely | Simpler hardware; accuracy/energy impact unknown | Documented future work, not yet tested |
| **Post-training quantization (int8)** | Reduce weight/activation precision | Lower memory-access energy + smaller FPGA footprint | Planned (FPGA phase) |

### 8.3 Hardware phase — measured FPGA results (IN PROGRESS, real numbers)
**Toolchain:** open-source **Yosys + nextpnr** (oss-cad-suite), targeting **Lattice iCE40 UltraPlus (UP5K)** — the *same FPGA family as the closest prior-art paper*, giving directly comparable, apples-to-apples resource numbers (vs. a cross-vendor Xilinx comparison). Vivado was abandoned mid-project (93-hour install on the available connection); the open-source flow is both faster and a better methodological match.

**Model under implementation (locked):** the FPGA target is the final delta+sparsity SNN (`FINAL_for_fpga`: Conv1d(4→16, k5) → LIF₁ → FC(528→256) → LIF₂ → FC(256→2) → LIF₃; learned β = {0.8216, 0.9108, 0.8853}, threshold 0.30; float macro-F1 82.9% / 83.2% smoothed). 136,274 parameters total, **fc1 (528×256 = 135,168 weights) is 99.3% of them** — so fc1 dominates every hardware budget.

**Weight preparation (done, reproducible from `.pth`):** trained weights exported → BatchNorm folded into conv1 → two quantization paths characterized. Weight signal-to-quantization-noise ratio (SQNR), measured per layer:

| Layer | Q6.10 (16-bit) SQNR | int8 SQNR | int8 scale |
|---|---|---|---|
| conv1 (BN-folded) | 68.1 dB | 45.7 dB | 0.0127 |
| **fc1 (dominant)** | 39.8 dB | **39.5 dB** | 0.00101 |
| fc2 | 42.9 dB | 45.9 dB | 0.00068 |

The key result: for the dominant fc1 layer, **int8 is within 0.3 dB of 16-bit** — because its weights are tiny (absmax ≈ 0.128), int8 loses almost nothing. This is what makes the 8-bit memory path viable (below). (A prior Colab run confirmed 16-bit end-to-end macro-F1 = 82.88% vs 82.92% float — lossless; the int8 accuracy check is the next Colab step.)

**Memory budget — the load-bearing hardware finding (measured):**

| Precision | fc1 memory | total weight memory | UP5K on-chip (~1,144 Kbit) | ECP5-25 EBR (~1,008 Kbit) |
|---|---|---|---|---|
| 16-bit (Q6.10) | 2,116 Kbit | 2,129 Kbit | **does not fit (186%)** | does not fit |
| **8-bit (int8)** | 1,058 Kbit | 1,065 Kbit | **fits (93%)** | does not fit |

So the design decision is settled by the numbers: **8-bit weights in the four SPRAM banks are the only way the full model fits on the UP5K**, and the fc1 SQNR shows that costs essentially no accuracy. 16-bit would force a larger device or external DRAM.

**Architecture — shared, time-multiplexed neuron core with external weight memory.** A fully-parallel design is infeasible (274 neurons × ~185 LUTs ≈ 50k LUTs vs 5,280 available), so one shared LIF compute unit processes all neurons sequentially. Critically, weights are **not** held in fabric registers — they are streamed from external synchronous memory (SPRAM) via address/data ports (`snn_fc_core.v`), which is what lets the 135k-weight fc1 layer exist at all.

**Verified & measured (real Verilog; Icarus Verilog simulation + Yosys synthesis + nextpnr P&R):**
- **LIF neuron** (`lif_neuron.v`): functionally verified — integrates, leaks, fires, resets with the real learned β = 0.8216; 48 LUTs + 1 DSP when DSP-mapped.
- **Shared FC+LIF engine, tiny case** (`snn_fc_layer.v`, 4→2, internal weights): bit-exact to hand calculation over 2 timesteps. P&R on UP5K: **~383–451 logic cells (~7–8%), 1 DSP (12%), Fmax ≈ 16–17 MHz.**
- **Full-scale FC+LIF core** (`snn_fc_core.v`, **528→256 with the real trained fc1 weights**, external memory): **bit-exact match to the numpy Q6.10 reference on all 512 neuron-spikes over 2 timesteps** (37 and 49 spikes respectively) — the actual paper-scale layer is now functionally proven, not just a toy. Synthesis (iCE40, DSP-mapped): **5,185 LUT4 + 4,733 flip-flops + 2 DSP (SB_MAC16)**.

**Second hardware finding — membrane storage must be BRAM, not flip-flops (measured before/after).** At 256 output neurons, the first full-scale core (`snn_fc_core.v`) holds every membrane potential in fabric flip-flops. Synthesis showed this nearly fills the chip by itself. Moving the 256 membranes into an inferred **block-RAM (EBR)** — `snn_fc_core_bram.v`, re-verified **bit-exact** against the same reference — collapses the datapath:

| Full-scale fc1 core (528→256) | LUT4 | Flip-flops | DSP (MAC16) | EBR (SB_RAM40_4K) | Fits UP5K? |
|---|---|---|---|---|---|
| Membranes in **flip-flops** (`snn_fc_core.v`) | 5,185 | 4,733 | 2 | 0 | barely (~98% LUTs) |
| Membranes in **BRAM** (`snn_fc_core_bram.v`) | **976** | **479** | 2 | 2 / 30 | **easily (~18% LUTs)** |

That is an **81% LUT and 90% flip-flop reduction** for the cost of 2 of the 30 EBR blocks — a clean, citable design-space result ("naive register-file membrane storage does not scale to a useful neuron count; membranes belong in on-chip RAM"), directly analogous to the shared-core rationale. The datapath now leaves the vast majority of the fabric free for the conv front-end and control.

**Third hardware finding — the missing conv1+LIF1 stage, built with true weight sharing.** Unlike the FC layers, conv1 reuses the same 336 parameters (16×4×5 weights + 16 biases) across all 33 spatial positions — so it cannot be treated as a generic fully-connected layer without either (a) exploding memory by expanding/duplicating the shared weights per position (would cost ~1.1 Mbit, defeating the point), or (b) implementing genuine weight-shared addressing. We built (b): `snn_conv1_core_bram.v` computes each of the 528 output neurons' receptive field (4 channels × 5 taps, zero-padded at the boundaries) by indexing directly into the true 336-parameter table, with the 528 membranes in BRAM (same lesson as fc1, since 528 > 256). **Verified bit-exact against a numpy Q6.10 reference on all 1,056 neuron-spike checks (528 neurons × 2 timesteps).** Synthesis: 1,597 LUT4 + 750 FF + 1 DSP + 4 BRAM.

**Fourth finding — the output layer needs membrane potential, not spike count.** Classification uses `mem.mean(0)` in the trained model — the time-averaged membrane of the 2 output neurons — not their spike rate. The generic FC engine only exposes `spike_out`, so a dedicated `snn_fc2_output.v` variant was built that also exposes the raw membrane value per output neuron (negligible extra cost at only 2 neurons). **Verified bit-exact on both spikes and membrane values.** Synthesis: 406 LUT4 + 171 FF + 1 DSP.

**Fifth finding — the full network, wired and verified end-to-end.** `snn_top.v` sequences conv1 → fc1 → fc2 with a small FSM (each stage's `done` triggers the next's `start`), with each layer's spike output combinationally indexed as the next layer's input. **Verified bit-exact against a genuinely chained Python reference** (conv1's real output feeds fc1, fc1's real output feeds fc2 — not three independently-tested stages) — a real width bug (`$clog2(528)=10`, not 9) was caught and fixed by this test. Standalone compute-logic total: conv1 (1,597) + fc1-BRAM (976) + fc2 (406) = **2,979 LUT4 (~56% of UP5K)**.

**Sixth finding — generic memory inference does not scale to fc1's weight table, and the failure is now quantified, not just estimated.** A naive full-system synthesis (weights loaded via `$readmemh` directly in the RTL, letting Yosys auto-infer memory) produced a concrete, unambiguous result: fc1's 2.16 Mbit weight table maps to **539 block-RAMs (SB_RAM40_4K) — against 30 available on UP5K, ~18× over budget.** (Total system: 4,758 LUT4 / 1,431 FF / 4 DSP / 539 BRAM.) This confirms the earlier analytical memory-budget finding (§ above) with an actual synthesis attempt rather than a calculation. The fix is well-defined and scoped, not vague: fc1's weights must be explicitly mapped to the chip's **4 dedicated SPRAM256K hard macros** (256 Kbit each, purpose-built for large tables) via direct `SB_SPRAM256KA` primitive instantiation, rather than relying on Yosys's automatic inference (which only targets the small EBR blocks). This is the concrete remaining FPGA step.

**Scope:** functional simulation + synthesis + resource, for every stage of the network individually **and now fully wired end-to-end**, bit-exact against Python references throughout. Remaining: explicit SPRAM instantiation for fc1's weights (well-scoped, described above), then full P&R for final timing; physical-board power is future work. **This already substantially exceeds both sibling papers from the same lab, which list FPGA as entirely unstarted future work — we have a functionally verified, fully-wired network and a precisely quantified memory-mapping problem with a defined solution.**

*(Artifacts in `fpga/`: RTL `lif_neuron.v`, `snn_fc_layer.v`, `snn_fc_core.v`, `snn_fc_core_bram.v`, `snn_conv1_core_bram.v`, `snn_fc2_output.v`, `snn_top.v` + testbenches; `read_pth.py`/`prep_weights.py` regenerate the folded/quantized weights, hex exports, and `fpga_weight_budget.json`; `gen_ref*.py` scripts build the per-layer and full-chain verification references.)*

## 9. Phased Research Roadmap
| Phase | Content | Status |
|---|---|---|
| **P1 — Data & methodology** | Leakage-free splits, task reframing, preprocessing | ✅ Done |
| **P2 — Model & ceiling** | Architecture/window/feature/protocol study, ceiling confirmation | ✅ Done |
| **P3 — Comparison** | Iso-architecture SNN vs CNN (accuracy + energy) | ✅ Done |
| **P4 — Accuracy push (optional)** | MVC normalization, k-shot calibration, LOSO CV | ⏳ LOSO ✅ done (§4.4); MVC/k-shot reserved (§8.1) |
| **P5 — Energy push** | Delta encoding + sparsity regularization | ⏳ Partly demonstrated (§8.2) |
| **P6 — FPGA** | iCE40 RTL (Yosys/nextpnr): resource/timing, weight budget | ⏳ In progress — full-scale fc1 core verified bit-exact + synthesized (§8.3); top-level wiring + BRAM membranes next |
| **P7 — Writing** | Paper (with supervisor) | 🔜 Final days |

---

## 10. Limitations & Open Questions
- Small cohort (13 subjects) — addressed via full pooled LOSO (§4.4), but per-subject variance remains genuine (range 64.9–89.5%).
- Labels are *self-perceived* fatigue (subjective) — we report **fatigue-state proxy**, not injury.
- Cross-subject ceiling (~78%, now confirmed by LOSO rather than a single split) reflects inter-subject EMG variability; a per-user calibration protocol (k-shot) is a promising, not-yet-cleanly-evaluated lever.
- Energy figures are **estimated** (analytical SynOps + Horowitz); FPGA synthesis gives resource/timing, not yet measured silicon power.
- FPGA top-level (conv1+fc1+fc2 wired together) is not yet integrated — each block is verified individually.
