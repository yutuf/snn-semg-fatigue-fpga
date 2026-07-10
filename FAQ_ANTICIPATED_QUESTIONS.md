# Anticipated Questions (read before asking)

Answers to the things most likely to come up, with pointers to the exact files/evidence.

---

### Q1. Why is accuracy "only" 78%? Isn't that low?
It's the **honest** number, and it's competitive for this task. 78.3% is the
**pooled 13-subject Leave-One-Subject-Out** figure — every test subject is completely
unseen in training. Easier protocols inflate this: the same model hits 82.9% on a
trial-grouped split and up to 89.5% on individual folds. The sibling lab papers report
subject-independent numbers in the same range or lower (apnea AUC 0.72–0.76; gesture
macro-F1 0.15–0.51 on a 12-class task). The point of the paper is **not** peak accuracy —
it's *matching a CNN's accuracy at ~28× less energy*. Evidence: `06_results_and_reports/loso_results.json`, recompute with `analyze_loso.py`.

### Q2. Why binary (Rested vs Fatigued) instead of the original 3 classes?
The 3-class self-reported fatigue labels are too noisy to separate — under leakage-free
evaluation 3-class capped at **48–50% macro-F1**, and it stayed capped even when the
subject *was* seen in training (trial-grouped), which proves the bottleneck is **label
noise**, not model capacity. Dropping the ambiguous middle class and predicting the two
extremes is both more accurate and more clinically meaningful (detecting the at-risk
state). Full reasoning: `training_journey_report.md` §3.

### Q3. How do I know the Verilog is actually correct, not just "compiles"?
Every layer and the full chain are **bit-exact** against a Python reference that computes
the identical Q6.10 fixed-point arithmetic. Re-run any of them (toolchain active):
```
iverilog -g2012 -o x.sim snn_conv1_core_bram.v snn_conv1_core_bram_tb.v && vvp x.sim   # conv1: 1056/1056 checks
iverilog -g2012 -o x.sim snn_fc2_output.v      snn_fc2_output_tb.v      && vvp x.sim   # fc2: spikes + membranes
iverilog -g2012 -o x.sim snn_top.v snn_conv1_core_bram.v snn_fc_core_bram.v snn_fc2_output.v snn_top_tb.v && vvp x.sim  # full chain
```
The golden vectors and the generators that made them are in `05_verification/`
(`gen_ref_conv1.py`, `gen_ref_fc2.py`, `gen_ref_full.py`).

### Q4. Which exact model gets deployed, and where are its weights?
`01_model/snn_fatigue_final.pth` (the `FINAL_for_fpga` checkpoint). Architecture and the
learned constants (β, thresholds) are in README §2 and `fpga_weight_budget.json`. The
weights are already quantized and exported to hex in `02_weights_hex/` — both Q6.10
(16-bit) and int8 — ready for `$readmemh`. `snn_top.v` already loads them.

### Q5. What's the inference latency / throughput?
We report it as **measured cycle counts** (one timestep = 305,567; full inference =
60.2 M cycles), **not** a fixed wall-clock number — because throughput is an open
FPGA-side optimization, not a fixed property of the design. This is the current
*area-minimal single-MAC* build (9–33% of a $5 chip), deliberately tiny rather than fast.
Three orthogonal knobs are left for the FPGA engineer: **parallel MAC cores** (~linear
speedup, fabric allows 16×+), **event-driven sparsity** (the current core still spends
cycles on zero-spike inputs — skipping them is the biggest available win and the SNN's
whole premise), and **clock** (Artix-7 runs far above iCE40's 16–17 MHz). Crucially, the
**~28× energy claim is independent of all of this** — energy depends on operation count,
not how serially they run. Full breakdown: `04_synthesis_results/LATENCY_AND_ENERGY.md`.

### Q6. Why iCE40 (tiny FPGA)? Why not just use the Artix-7 that Vivado targets?
iCE40 was chosen to match the closest prior art (the 5K-LUT SNN-on-FPGA gesture paper we
adapt) and to make the ultra-low-power wearable story concrete. **But the RTL is
vendor-neutral** — the FSMs and Q6.10 arithmetic port directly; only the iCE40 SPRAM
primitive is device-specific and can be dropped. On Artix-7 the fc1 weight table fits in
block RAM/URAM natively, which is *simpler* than the iCE40 SPRAM path — so targeting
Artix for the power measurement is a perfectly good call.

### Q7. What is actually left to do (so I know the true state)?
1. **Measured power** — run Vivado `report_power` (the one blocked item). This is the ask.
2. **Confirm int8 accuracy** — 16-bit is already verified lossless (82.88% vs 82.92% float);
   int8 weights are exported but the end-to-end int8 accuracy check hasn't been run yet.
3. **fc1 weight sourcing at full scale** — the SPRAM boot-load pattern is proven
   (`spram_conv1_weights.v`), but streaming all 135k weights from external SPI flash at
   boot isn't built (or: use int8 on-chip, or use Artix BRAM). This is the main open
   engineering choice — see README §5.
Everything else (model, preprocessing, LOSO eval, per-layer + full-chain verified RTL,
synthesis, resource/latency numbers) is done.

### Q8. Can I retrain or reproduce the model? Where's the training code?
Yes. `00_dataset/download_dataset.py` + `preprocess.py` reproduce the exact model inputs
from the public dataset (DOI 10.5281/zenodo.14182446). The full training + LOSO harness
(the Colab cell, ready to paste) is reproduced in `06_results_and_reports/` and detailed
in `training_journey_report.md`. `read_pth.py` reads the checkpoint **without** needing
PyTorch installed, if you just want to inspect weights.

### Q9. What's the fixed-point format?
**Q6.10** signed 16-bit (6 integer bits, 10 fractional) throughout the datapath;
accumulators are 32-bit with saturation to ±32767. β and threshold are pre-converted:
β·mem uses `(BETA*mem)>>>10`. Constants baked into the RTL: β = {0.8216, 0.9108, 0.8853}
→ {841, 933, 907} in Q6.10; threshold 0.30 → 307. Weights: see `prep_weights.py` for the
exact quantization (and int8 scales in `fpga_weight_budget.json`).

### Q10. How is this different from the two sibling lab papers (so we don't look redundant)?
Same energy methodology (Horowitz), different application and different maturity:
- **Apnea paper:** ECG, apnea labels, FPGA = future work.
- **Gesture paper:** sEMG *gesture* recognition; uses fatigue only as a *synthetic signal
  degradation*, never classifies fatigue. FPGA = future work.
- **This work:** sEMG, **fatigue is the actual label**, on real data — and it's the **only
  one of the three carried to verified RTL + synthesis**, not left as future work.

### Q11. What toolchain / dependencies do I need?
- **RTL sim/synth:** oss-cad-suite (Yosys + nextpnr + IceStorm + Icarus Verilog), free.
  Note `-g2012` when compiling anything that includes iCE40 `cells_sim.v`. For the power
  number: Vivado (already installed).
- **Python:** numpy, pandas, torch, snntorch, scikit-learn, tqdm. Only needed to
  retrain/re-preprocess; the handoff hardware flow needs none of it.
