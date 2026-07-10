# Latency, Resource & Energy — the hardware numbers

Every number here is either **measured** (simulation cycle count, or Yosys/nextpnr
output) or **derived from an explicit formula shown inline**. Nothing is hand-waved.

---

## 1. Resource utilization (measured — nextpnr, iCE40 UP5K)

Logs: `pnr_conv1.txt`, `pnr_fc1bram.txt`, `pnr_fc2.txt` in this folder.

| Module | Logic cells (of 5280) | BRAM (of 30) | DSP (of 8) |
|---|---|---|---|
| conv1 + LIF1 (`snn_conv1_core_bram`) | 1753 (33%) | 4 (13%) | 1 (12%) |
| fc1 528→256 (`snn_fc_core_bram`) | 1127 (21%) | 2 (6%) | 2 (25%) |
| fc2 256→2 (`snn_fc2_output`) | 510 (9%) | 0 | 1 (12%) |

**Compute logic is tiny** — each layer fits several times over. The constraint is
**weight memory**, not logic (see README §5): fc1's 135k weights need external SPRAM
or int8, not the on-chip BRAM.

**Clock:** P&R reports **Fmax ≈ 16–17 MHz** (`../04_synthesis_results/` and the
original `pnr_out.txt`). The datapath is time-multiplexed, so the critical path
(one multiply-accumulate + LIF update) is the same regardless of layer size — this
Fmax is representative of the full design.

---

## 2. Latency (MEASURED in simulation — reproducible)

Testbench: `05_verification/lat_tb.v`. Run it:
```
iverilog -g2012 -o lat.sim snn_top.v snn_conv1_core_bram.v snn_fc_core_bram.v snn_fc2_output.v lat_tb.v && vvp lat.sim
```
It prints:
```
ONE_TIMESTEP_CYCLES=305567
FULL_INFERENCE_197_CYCLES=60196699
```

### Per-layer cycle breakdown (one timestep)
The FC engine spends `2·N_IN + 3` cycles per output neuron (address → data → fire →
next; see `snn_fc_core_bram.v` FSM), so:

| Layer | Formula | Cycles/timestep | Share |
|---|---|---|---|
| conv1 (16×33 outputs, weight-shared) | measured residual | 33,433 | 11% |
| **fc1** (256 neurons × (2·528+3)) | 256 × 1059 | **271,104** | **89%** |
| fc2 (2 neurons × (2·256+3)) | 2 × 515 | 1,030 | <1% |
| **total/timestep** | | **305,567** | |

### Full inference
- 197 timesteps × 305,567 = **60,196,699 cycles**.
- We deliberately report this as a **cycle count, not a fixed wall-clock latency** —
  the seconds figure depends on clock and parallelism choices that are the FPGA
  engineer's to make. (For reference only: at 17 MHz the un-optimized serial build is
  ~3.5 s; this is a *baseline*, not a result to quote.)

### Latency is an open optimization axis — three orthogonal knobs
This is the **area-minimal operating point**: ONE shared multiply-accumulate unit does
every synapse serially, using just 9–33% of a $5 FPGA. Throughput is intentionally left
open:

1. **Parallelism.** Latency drops ~linearly with parallel MAC cores. 16 parallel fc1
   cores → fc1 ≈ 17k cycles/timestep → whole inference ≈ 0.22 s. Fabric easily allows it
   (fc1 core = 21%).
2. **Event-driven sparsity (not yet exploited).** The current FSM walks *every* input
   synapse and only conditionally accumulates (`if (in_spike) acc <= acc + w_data`) — it
   still spends the cycles on zero-spike inputs. A true event-driven core that iterates
   only *active* spikes would cut fc1's 2·N_IN cost by the input sparsity factor. This is
   the single biggest latency win available and is the SNN's whole premise.
3. **Clock.** 16–17 MHz is the iCE40 P&R ceiling; an Artix-7 target runs much faster.

**Energy is independent of all three.** Energy per inference is set by the *number* of
operations (§3), not how serially/fast they run — so latency optimization does **not**
change the ~28× energy story. The window itself is 5 s of EMG, so even the un-optimized
serial build is within an order of magnitude of real-time for a single window.

---

## 3. Operation counts & energy model

### Dense operation count per inference (both models do this many)
Per timestep:
- conv1: 16 out-ch × 33 pos × (4 in-ch × 5 tap) = **10,560 MAC**
- fc1: 256 × 528 = **135,168 MAC**
- fc2: 2 × 256 = **512 MAC**
- total = 146,240 ops/timestep × 197 = **28.81 M operations / inference**

### Energy model (Horowitz, ISSCC 2014, 45 nm — same as both sibling papers)
- **MAC = 4.6 pJ**, **AC (accumulate) = 0.9 pJ**.
- **CNN baseline:** every op is a full MAC → 28.81 M × 4.6 pJ ≈ **132 µJ / inference** (dense upper bound).
- **SNN:** conv front-end is MAC; the FC layers are **spike-driven accumulates** (AC, 0.9 pJ)
  and only fire on the fraction of neurons that spike. Folding in the measured firing
  sparsity (delta encoding + L1 spike-rate regularization) gives the estimated
  **~28× lower energy vs the iso-architecture CNN** — the paper's headline.
  Full stage-by-stage SynOps derivation (9.9× → 28.8×) is in
  `../06_results_and_reports/training_journey_report.md` §4.1–4.3.

### The one missing number (the handoff ask)
The ~28× is a **relative, analytical** figure. To turn it into an **absolute measured**
one:
> **joules/inference = measured power (Vivado `report_power`, in mW) × latency (cycles ÷ Fmax)**

That measured power is exactly what we could not obtain (Lattice tools gated, see
README §6). With Vivado now installed, `report_power` closes this — the last hardware
number for the paper.

---

## 4. Summary table (paste-ready for the paper's hardware section)

| Quantity | Value | Source |
|---|---|---|
| Logic (per layer) | 9–33% of iCE40 UP5K | nextpnr (measured) |
| DSP | 1–2 per layer | nextpnr (measured) |
| Fmax | ~16–17 MHz | nextpnr (measured) |
| Cycles / timestep | 305,567 | simulation (measured) |
| Cycles / inference (197 steps) | 60,196,699 | simulation (measured) |
| Wall-clock latency | *open FPGA-side optimization* (parallelism / sparsity / clock) | — |
| Ops / inference | 28.81 M | counted |
| CNN energy (dense, Horowitz) | ~132 µJ | derived |
| SNN vs CNN energy | ~28× lower | analytical SynOps (report §4) |
| Absolute SNN joules/inference | **TODO — Vivado report_power** | — |
