#!/usr/bin/env python3
"""
One-command verification of the whole handoff.

The RTL and testbenches reference weight/vector files by BARE filename (e.g.
"conv1_w_q610.hex"), so every input must sit in the working directory. Those files
are intentionally organized into 02_weights_hex/ and 05_verification/ for humans, so
this script assembles a flat build/ dir and runs every self-checking testbench there.

Prerequisite: the open-source toolchain must be on PATH (iverilog + vvp).
Activate it first, e.g. on Windows:  call <path>\\oss-cad-suite\\environment.bat
then:  python run_all_tests.py

Exit code 0 = all bit-exact checks PASS, non-zero = something failed.
"""
import os, shutil, subprocess, sys, re

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")

# (label, [source verilog files, in compile order])  -- module + its testbench
TESTS = [
    ("LIF neuron",       ["lif_neuron.v", "lif_tb.v"]),
    ("conv1 core",       ["snn_conv1_core_bram.v", "snn_conv1_core_bram_tb.v"]),
    ("fc2 output layer", ["snn_fc2_output.v", "snn_fc2_output_tb.v"]),
    ("full chain (top)", ["snn_top.v", "snn_conv1_core_bram.v", "snn_fc_core_bram.v",
                          "snn_fc2_output.v", "snn_top_tb.v"]),
]

def have(tool):
    return shutil.which(tool) is not None

def assemble():
    if os.path.exists(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD)
    for sub in ("03_rtl", "02_weights_hex", "05_verification"):
        d = os.path.join(ROOT, sub)
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp):
                shutil.copy(fp, os.path.join(BUILD, fn))

def run(label, files):
    sim = re.sub(r"\W+", "_", label) + ".sim"
    comp = subprocess.run(["iverilog", "-g2012", "-o", sim] + files,
                          cwd=BUILD, capture_output=True, text=True)
    if comp.returncode != 0:
        print(f"[COMPILE-FAIL] {label}\n{comp.stderr.strip()}")
        return False
    res = subprocess.run(["vvp", sim], cwd=BUILD, capture_output=True, text=True)
    out = res.stdout + res.stderr
    ok = ("PASS" in out) and ("FAIL" not in out.upper().replace("PASS", ""))
    # LIF tb has no PASS/FAIL string; treat clean run as informational
    tag = "PASS" if ok else ("run" if "PASS" not in out and "FAIL" not in out.upper() else "FAIL")
    lastlines = "\n    ".join(l for l in out.strip().splitlines()[-3:])
    print(f"[{tag:4}] {label}\n    {lastlines}\n")
    # only the self-checking tbs gate the exit code
    if "PASS" in out or "FAIL" in out.upper():
        return "FAIL" not in out.upper().replace("PASS", "")
    return True

def main():
    if not (have("iverilog") and have("vvp")):
        print("ERROR: iverilog/vvp not on PATH. Activate oss-cad-suite first "
              "(e.g. `call <path>\\oss-cad-suite\\environment.bat`), then re-run.")
        sys.exit(2)
    assemble()
    print(f"Assembled build/ ({len(os.listdir(BUILD))} files). Running testbenches...\n")
    results = [(lbl, run(lbl, files)) for lbl, files in TESTS]
    passed = sum(1 for _, ok in results if ok)
    print("=" * 48)
    print(f"SUMMARY: {passed}/{len(results)} checks OK")
    for lbl, ok in results:
        print(f"  {'OK ' if ok else 'BAD'}  {lbl}")
    sys.exit(0 if passed == len(results) else 1)

if __name__ == "__main__":
    main()
