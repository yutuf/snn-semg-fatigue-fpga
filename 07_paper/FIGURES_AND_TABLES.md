# Figures & Tables plan (all generable from real data in this package)

Nothing here is a placeholder — each item lists the exact data source so it can be plotted
from real numbers. Generate as vector PDF/EPS for IEEE.

## Tables
| # | Table | Data source |
|---|---|---|
| T1 | Pooled 13-fold LOSO: SNN vs CNN accuracy/macro-F1 + McNemar p | `06_results_and_reports/loso_results.json` (`analyze_loso.py`) |
| T2 | Per-fold accuracy (13 subjects), SNN & CNN | `loso_results.json` (`per_subject_acc`) |
| T3 | Sensitivity/specificity, confusion matrix | `loso_results.json` (confusion matrices) |
| T4 | Energy roadmap Stage 0→3 (SynOps, multiplier) | report §4.3 |
| T5 | FPGA resource per layer (LUT/FF/DSP/BRAM) + Fmax | `04_synthesis_results/pnr_*.txt`, `LATENCY_AND_ENERGY.md` |
| T6 | Latency: cycles per layer / per inference | `LATENCY_AND_ENERGY.md` §2 |
| T7 | Differentiation vs 2 companion lab papers | report §7.6 |

## Figures
| # | Figure | Data source | Notes |
|---|---|---|---|
| F1 | System pipeline: sEMG → STFT → delta-encode → SNN → FPGA datapath | schematic | draw; no data needed |
| F2 | Per-fold accuracy bars, SNN vs CNN, 13 subjects | T2 data | shows parity + inter-subject spread |
| F3 | Accuracy vs energy trade-off (Stage 0→3 + smoothing) | report §4.3 | the core efficiency story |
| F4 | FPGA datapath / FSM of the time-multiplexed core | `03_rtl/snn_fc_core_bram.v` FSM | architecture figure |
| F5 | (optional) membrane-potential trace of a LIF neuron over timesteps | `lif_neuron.v` sim | illustrative |

## Consistency rules (enforce before submission)
- Every number in prose must equal its table cell. Re-run `analyze_loso.py` to regenerate
  T1–T3 rather than hand-copying.
- Energy multipliers (28×, 9.9×, etc.) must match report §4.3 exactly.
- Resource/latency numbers must match the `pnr_*.txt` logs and `lat_tb.v` output — cite the
  measured values, not rounded guesses.
- If the measured-power `[TODO]` is not filled by submission, F3/energy table must clearly
  label the energy axis as *analytical (Horowitz)*, and state measured power as future work.
