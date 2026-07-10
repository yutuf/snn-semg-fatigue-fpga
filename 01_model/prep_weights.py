"""
FPGA weight preparation for the trained SNN fatigue model.

Steps:
  1. Load checkpoint (torch-free) + print metadata.
  2. Fold BatchNorm(bn1) into conv1  ->  conv1_folded.
  3. Report exact on-chip memory budget (16-bit vs 8-bit) vs iCE40 UP5K / ECP5.
  4. Quantize each layer (Q6.10 16-bit, and int8 per-tensor symmetric),
     report quantization SNR per layer.
  5. Export real weight hex files for RTL $readmemh.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_pth import load_state_dict

sys.stdout.reconfigure(encoding="utf-8")

CKPT = r"C:\Users\Asuss\Downloads\oss-cad-suite\snn_fatigue_final.pth"
OUT = r"C:\Users\Asuss\Downloads\snn_fpga"

obj = load_state_dict(CKPT)
sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

# ---- metadata ----
print("=== CHECKPOINT METADATA ===")
for k in ("config", "arch", "betas", "threshold", "delta_thr",
          "val_macro_f1", "smoothed_macro_f1", "classes", "T_frames"):
    if isinstance(obj, dict) and k in obj:
        print(f"  {k}: {obj[k]}")

def arr(name):
    return np.asarray(sd[name], dtype=np.float64)

# ---- 1. BN fold into conv1 ----
w = arr("conv1.weight")            # (16,4,5)
b = arr("conv1.bias")              # (16,)
g = arr("bn1.weight")              # gamma (16,)
beta = arr("bn1.bias")             # (16,)
mean = arr("bn1.running_mean")     # (16,)
var = arr("bn1.running_var")       # (16,)
eps = 1e-5
scale = g / np.sqrt(var + eps)     # (16,)
w_fold = w * scale[:, None, None]  # broadcast over (in,k)
b_fold = (b - mean) * scale + beta
print("\n=== BN FOLD (conv1) ===")
print(f"  conv1_folded.weight absmax={np.abs(w_fold).max():.5f}  (was {np.abs(w).max():.5f})")
print(f"  conv1_folded.bias   absmax={np.abs(b_fold).max():.5f}  (was {np.abs(b).max():.5f})")

fc1_w = arr("fc1.weight"); fc1_b = arr("fc1.bias")
fc2_w = arr("fc2.weight"); fc2_b = arr("fc2.bias")

layers = {
    "conv1": (w_fold, b_fold),
    "fc1":   (fc1_w, fc1_b),
    "fc2":   (fc2_w, fc2_b),
}

# ---- 2. Memory budget ----
n_conv1 = w_fold.size + b_fold.size
n_fc1 = fc1_w.size + fc1_b.size
n_fc2 = fc2_w.size + fc2_b.size
n_tot = n_conv1 + n_fc1 + n_fc2

# UP5K: 4x SPRAM256K (256 Kbit each) = 1024 Kbit + 30x EBR (4 Kbit) = 120 Kbit
UP5K_SPRAM = 4 * 256 * 1024
UP5K_EBR = 30 * 4 * 1024
UP5K_TOTAL = UP5K_SPRAM + UP5K_EBR
# ECP5-25: ~1008 Kbit EBR + 194 Kbit dist; use LUT-RAM+EBR ~ 1.3 Mbit practical
ECP5_25_EBR = 56 * 18 * 1024  # 56 EBR x 18 Kbit

print("\n=== PARAMETER / MEMORY BUDGET ===")
print(f"  conv1(folded): {n_conv1:6d} params")
print(f"  fc1          : {n_fc1:6d} params  (dominant)")
print(f"  fc2          : {n_fc2:6d} params")
print(f"  TOTAL        : {n_tot:6d} params")
for bits in (16, 8):
    tot_bits = n_tot * bits
    fc1_bits = n_fc1 * bits
    print(f"\n  --- {bits}-bit weights ---")
    print(f"    total weight memory : {tot_bits/1024:.1f} Kbit ({tot_bits/1e6:.3f} Mbit)")
    print(f"    fc1 alone           : {fc1_bits/1024:.1f} Kbit ({fc1_bits/1e6:.3f} Mbit)")
    print(f"    UP5K on-chip total  : {UP5K_TOTAL/1024:.0f} Kbit "
          f"-> {'FITS' if tot_bits<=UP5K_TOTAL else 'DOES NOT FIT'} "
          f"({100*tot_bits/UP5K_TOTAL:.0f}% of on-chip mem)")
    print(f"    ECP5-25 EBR         : {ECP5_25_EBR/1024:.0f} Kbit "
          f"-> {'FITS' if tot_bits<=ECP5_25_EBR else 'DOES NOT FIT'}")

# ---- 3. Quantization: Q6.10 (16-bit) and int8 per-tensor symmetric ----
def q610(x):
    q = np.round(x * 1024.0)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.int64), q / 1024.0

def int8_sym(x):
    amax = np.abs(x).max()
    s = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.round(x / s), -127, 127).astype(np.int64)
    return q, q * s, s

def snr_db(x, xq):
    p = np.mean(x**2); e = np.mean((x - xq)**2)
    return 10*np.log10(p/e) if e > 0 else float("inf")

print("\n=== QUANTIZATION ERROR (weight SQNR) ===")
print(f"{'layer':10s} {'Q6.10 SNR(dB)':>14s} {'int8 SNR(dB)':>13s} {'int8 scale':>12s}")
quant_export = {}
for name, (ww, bb) in layers.items():
    _, wq610 = q610(ww)
    q8, wq8, s8 = int8_sym(ww)
    print(f"{name:10s} {snr_db(ww, wq610):14.1f} {snr_db(ww, wq8):13.1f} {s8:12.6f}")
    quant_export[name] = {"w": ww, "b": bb, "int8_scale": s8}

# ---- 4. Export hex (Q6.10 16-bit, row-major neuron-major j*N_IN+i) ----
def to_hex16(qint):
    # two's complement 16-bit
    return ["{:04x}".format(int(v) & 0xFFFF) for v in np.asarray(qint).ravel()]

def to_hex8(qint):
    return ["{:02x}".format(int(v) & 0xFF) for v in np.asarray(qint).ravel()]

# fc1 weights: (256,528) -> flatten neuron-major (row j then inputs i) == C-order
for name, (ww, bb) in layers.items():
    q16, _ = q610(ww)
    q8, _, _ = int8_sym(ww)
    with open(os.path.join(OUT, f"{name}_w_q610.hex"), "w") as f:
        f.write("\n".join(to_hex16(q16)) + "\n")
    with open(os.path.join(OUT, f"{name}_w_int8.hex"), "w") as f:
        f.write("\n".join(to_hex8(q8)) + "\n")
    qb16, _ = q610(bb)
    with open(os.path.join(OUT, f"{name}_b_q610.hex"), "w") as f:
        f.write("\n".join(to_hex16(qb16)) + "\n")

print("\nExported: {conv1,fc1,fc2}_w_q610.hex / _w_int8.hex / _b_q610.hex to snn_fpga/")

# ---- 5. Save budget summary json ----
summary = {
    "arch": obj.get("arch") if isinstance(obj, dict) else None,
    "betas": {"lif1": float(arr('lif1.beta')), "lif2": float(arr('lif2.beta')), "lif3": float(arr('lif3.beta'))},
    "threshold": float(arr("lif1.threshold")),
    "params": {"conv1_folded": int(n_conv1), "fc1": int(n_fc1), "fc2": int(n_fc2), "total": int(n_tot)},
    "memory_kbit": {
        "16bit_total": n_tot*16/1024, "8bit_total": n_tot*8/1024,
        "16bit_fc1": n_fc1*16/1024, "8bit_fc1": n_fc1*8/1024,
        "up5k_onchip": UP5K_TOTAL/1024, "ecp5_25_ebr": ECP5_25_EBR/1024,
    },
    "int8_scales": {k: float(v["int8_scale"]) for k, v in quant_export.items()},
}
with open(os.path.join(OUT, "fpga_weight_budget.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote fpga_weight_budget.json")
