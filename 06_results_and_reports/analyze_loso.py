"""
Pool the 13-subject LOSO folds and compute the paper-ready statistics:
  - pooled confusion matrices (SNN, CNN)
  - accuracy, macro-F1, sensitivity, specificity, per-class F1
  - per-subject mean +/- std (matches sibling-paper 'house style')
  - McNemar test (exact binomial + chi-square with continuity correction)

Reads  loso_results.json  (next to this script), writes  loso_summary.json
and prints a Markdown-ready table to stdout.
"""
import json
import os
import sys
import math

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "loso_results.json")

with open(DATA, encoding="utf-8") as f:
    blob = json.load(f)

accounts = blob["accounts"]


def metrics(cm):
    """cm rows = true [Rested(0), Fatigued(1)], cols = predicted."""
    cm = np.asarray(cm, dtype=float)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = cm.sum()
    acc = (tp + tn) / total

    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0

    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) else 0.0

    return {
        "acc": acc * 100,
        "macro_f1": (f1_0 + f1_1) / 2 * 100,
        "sensitivity": rec1 * 100,   # recall on Fatigued
        "specificity": rec0 * 100,   # recall on Rested
        "f1_rested": f1_0 * 100,
        "f1_fatigued": f1_1 * 100,
    }


# ---- pool confusion matrices + McNemar counts across accounts ----
snn_cm = np.zeros((2, 2), dtype=int)
cnn_cm = np.zeros((2, 2), dtype=int)
mc = {"both_correct": 0, "snn_only_correct": 0, "cnn_only_correct": 0, "both_wrong": 0}

for a in accounts:
    snn_cm += np.array(a["snn_confusion"], dtype=int)
    cnn_cm += np.array(a["cnn_confusion"], dtype=int)
    for k in mc:
        mc[k] += a["mcnemar_table"][k]

# ---- per-subject accuracy arrays (for mean +/- std) ----
snn_subj, cnn_subj = {}, {}
for a in accounts:
    for subj, d in a["per_subject_acc"].items():
        snn_subj[int(subj)] = d["snn"]
        cnn_subj[int(subj)] = d["cnn"]

subj_ids = sorted(snn_subj)
snn_acc_arr = np.array([snn_subj[s] for s in subj_ids])
cnn_acc_arr = np.array([cnn_subj[s] for s in subj_ids])

snn_pooled = metrics(snn_cm)
cnn_pooled = metrics(cnn_cm)

# ---- McNemar (SNN vs CNN) on pooled discordant pairs ----
b = mc["snn_only_correct"]   # SNN right, CNN wrong
c = mc["cnn_only_correct"]   # CNN right, SNN wrong
n_disc = b + c
# exact two-sided binomial
lo = min(b, c)
p_exact = min(1.0, 2 * sum(stats.binom.pmf(k, n_disc, 0.5) for k in range(lo + 1)))
# chi-square with continuity correction
chi2 = (abs(b - c) - 1) ** 2 / n_disc if n_disc else 0.0
p_chi = stats.chi2.sf(chi2, 1)

# ---- paired t-test on per-subject accuracy (secondary) ----
t_stat, t_p = stats.ttest_rel(snn_acc_arr, cnn_acc_arr)

N = int(snn_cm.sum())

summary = {
    "protocol": "leave_one_subject_out_13fold",
    "total_windows": N,
    "pooled": {
        "snn": snn_pooled,
        "cnn": cnn_pooled,
        "snn_confusion": snn_cm.tolist(),
        "cnn_confusion": cnn_cm.tolist(),
    },
    "per_subject_mean_std": {
        "snn_acc_mean": float(snn_acc_arr.mean()),
        "snn_acc_std": float(snn_acc_arr.std(ddof=1)),
        "cnn_acc_mean": float(cnn_acc_arr.mean()),
        "cnn_acc_std": float(cnn_acc_arr.std(ddof=1)),
        "snn_per_subject": snn_subj,
        "cnn_per_subject": cnn_subj,
    },
    "mcnemar": {
        "both_correct": mc["both_correct"],
        "snn_only_correct": b,
        "cnn_only_correct": c,
        "both_wrong": mc["both_wrong"],
        "discordant_pairs": n_disc,
        "chi2_cc": chi2,
        "p_exact": p_exact,
        "p_chi2_cc": p_chi,
        "direction": "SNN better" if b > c else ("CNN better" if c > b else "tie"),
    },
    "paired_ttest_per_subject_acc": {"t": float(t_stat), "p": float(t_p)},
}

with open(os.path.join(HERE, "loso_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ---------- pretty print ----------
def row(label, m):
    return (f"| {label} | {m['acc']:.1f} | {m['macro_f1']:.1f} | "
            f"{m['sensitivity']:.1f} | {m['specificity']:.1f} | "
            f"{m['f1_rested']:.1f} | {m['f1_fatigued']:.1f} |")

print(f"=== POOLED 13-SUBJECT LOSO  (N = {N} windows) ===\n")
print("| Model | Acc | MacroF1 | Sens (Fatigued) | Spec (Rested) | F1 Rested | F1 Fatigued |")
print("|---|---|---|---|---|---|---|")
print(row("SNN (ours)", snn_pooled))
print(row("CNN (iso)", cnn_pooled))
print(f"\nSNN confusion [rows=true(Rested,Fatigued), cols=pred]: {snn_cm.tolist()}")
print(f"CNN confusion: {cnn_cm.tolist()}")

print("\n=== Per-subject accuracy (mean +/- std over 13 folds) ===")
print(f"  SNN: {snn_acc_arr.mean():.1f} +/- {snn_acc_arr.std(ddof=1):.1f} %")
print(f"  CNN: {cnn_acc_arr.mean():.1f} +/- {cnn_acc_arr.std(ddof=1):.1f} %")
print(f"  paired t-test (per-subject acc): t = {t_stat:.3f}, p = {t_p:.4f}")

print("\n=== McNemar (SNN vs CNN, pooled window predictions) ===")
print(f"  both correct {mc['both_correct']} | SNN-only {b} | CNN-only {c} | both wrong {mc['both_wrong']}")
print(f"  discordant pairs n = {n_disc}")
print(f"  exact p = {p_exact:.4f}  (chi2 w/ cc = {chi2:.2f}, p = {p_chi:.4f})")
print(f"  direction: {summary['mcnemar']['direction']} on discordant pairs ({b} vs {c})")

print("\n=== Verdict ===")
d = snn_pooled["macro_f1"] - cnn_pooled["macro_f1"]
sig = "NOT statistically significant" if p_exact > 0.05 else "statistically significant"
print(f"  SNN MacroF1 {snn_pooled['macro_f1']:.1f}% vs CNN {cnn_pooled['macro_f1']:.1f}%  (delta {d:+.1f} pt)")
print(f"  Difference is {sig} (McNemar exact p = {p_exact:.4f}).")
print("\n  Wrote loso_summary.json")
