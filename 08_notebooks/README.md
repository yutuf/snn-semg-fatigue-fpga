# Colab Notebooks

Runnable notebooks that reproduce the software side of the project. Upload to Google Colab
(**Runtime → Change runtime type → GPU**) or run locally with a Python/PyTorch environment.

| Notebook | What it does |
|---|---|
| `01_SNN_fatigue_pipeline.ipynb` | **The main pipeline.** Download dataset → preprocess to spikes → train SNN + iso-CNN → full 13-fold LOSO + McNemar → Horowitz energy estimate. Reproduces the headline result (SNN 78.3% / CNN 78.0%, p=0.354, ~28× energy). |
| `02_FPGA_weight_export.ipynb` | Bridges the trained checkpoint to hardware: loads `snn_fatigue_final.pth`, folds BatchNorm, quantizes to Q6.10/int8, writes the `.hex` files the Verilog loads. Wraps `01_model/read_pth.py` + `prep_weights.py`. |

## Notes
- The code in `01_...` is the same code (model definitions, LOSO harness, energy
  instrumentation) that produced the cited numbers on Colab — reorganized into clean cells.
  Every cell was JSON- and Python-syntax-validated. It was **not** re-executed end-to-end
  during packaging (needs GPU + the 3 GB dataset + tens of minutes); run it on Colab to
  regenerate results.
- Full 13-fold LOSO in one session is slow. To split it, set `SUBJECTS = [1,2,3]` (etc.) in
  separate runs and sum the printed confusion matrices — `../06_results_and_reports/analyze_loso.py`
  pools them exactly this way (that's how the original result was produced across 5 accounts).
- These notebooks reproduce the **software** results. The FPGA/RTL work is Verilog, verified
  separately via `../run_all_tests.py`.
