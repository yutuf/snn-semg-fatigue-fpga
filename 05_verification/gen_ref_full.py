"""
Full-chain reference: conv1 -> fc1 -> fc2, for 2 timesteps.
Reuses the EXACT per-layer Q6.10 integer arithmetic already proven
bit-exact in isolation (gen_ref_conv1.py / gen_ref.py / gen_ref_fc2.py),
chained so the input to fc1 is conv1's real output (not a synthetic
pattern), and the input to fc2 is fc1's real output. This is what
proves the TOP-LEVEL WIRING/sequencing is correct, on top of each
layer already being individually correct.
"""
import os, numpy as np
OUT = r"C:\Users\Asuss\Downloads\snn_fpga"
SAT_MAX, SAT_MIN = 32767, -32768

def load_hex_signed(fname, shape=None):
    v = np.loadtxt(os.path.join(OUT, fname), dtype=str)
    v = np.array([int(x, 16) for x in v], dtype=np.int64)
    v = np.where(v >= 0x8000, v - 0x10000, v)
    return v.reshape(shape) if shape else v

# ---- conv1 ----
N_CIN, N_COUT, K, PAD, N_POS = 4, 16, 5, 2, 33
N_IN0, N_OUT0 = N_CIN*N_POS, N_COUT*N_POS          # 132, 528
BETA1, THRESH = 841, 307
w0 = load_hex_signed("conv1_w_q610.hex", (N_COUT, N_CIN, K))
b0 = load_hex_signed("conv1_b_q610.hex")

# ---- fc1 ----
N_IN1, N_OUT1 = 528, 256
BETA2 = 933
w1 = load_hex_signed("fc1_w_q610.hex", (N_OUT1, N_IN1))
b1 = load_hex_signed("fc1_b_q610.hex")

# ---- fc2 ----
N_IN2, N_OUT2 = 256, 2
BETA3 = 907
w2 = load_hex_signed("fc2_w_q610.hex", (N_OUT2, N_IN2))
b2 = load_hex_signed("fc2_b_q610.hex")

# deterministic conv1 input (same generator as gen_ref_conv1.py, so the
# already-verified conv1 stage is exercised identically)
spk_in0 = np.zeros((2, N_CIN, N_POS), dtype=np.int64)
for t in range(2):
    for ci in range(N_CIN):
        for pos in range(N_POS):
            spk_in0[t, ci, pos] = 1 if ((ci*7 + pos*(t+1) + 2) % 4 == 0) else 0

mem0 = np.zeros((N_COUT, N_POS), dtype=np.int64); prev0 = np.zeros((N_COUT, N_POS), dtype=np.int64)
mem1 = np.zeros(N_OUT1, dtype=np.int64); prev1 = np.zeros(N_OUT1, dtype=np.int64)
mem2 = np.zeros(N_OUT2, dtype=np.int64); prev2 = np.zeros(N_OUT2, dtype=np.int64)

conv1_spikes_all, fc1_spikes_all, fc2_spikes_all, fc2_mem_all = [], [], [], []

for t in range(2):
    # ---- conv1 ----
    spk0 = np.zeros((N_COUT, N_POS), dtype=np.int64)
    for co in range(N_COUT):
        for p in range(N_POS):
            acc = 0
            for ci in range(N_CIN):
                for k in range(K):
                    pos_in = p + k - PAD
                    if 0 <= pos_in < N_POS and spk_in0[t, ci, pos_in]:
                        acc += int(w0[co, ci, k])
            beta_mem = BETA1 * int(mem0[co, p]); leaked = beta_mem >> 10
            mem_next = leaked + acc + int(b0[co]) - (THRESH if prev0[co, p] else 0)
            mem_next = max(SAT_MIN, min(SAT_MAX, mem_next))
            s = 1 if mem_next >= THRESH else 0
            mem0[co, p] = mem_next; prev0[co, p] = s; spk0[co, p] = s
    spk0_flat = spk0.reshape(N_OUT0)   # = fc1's input (528,)
    conv1_spikes_all.append(spk0_flat.copy())

    # ---- fc1 ----
    spk1 = np.zeros(N_OUT1, dtype=np.int64)
    for j in range(N_OUT1):
        acc = int((w1[j] * spk0_flat).sum())
        beta_mem = BETA2 * int(mem1[j]); leaked = beta_mem >> 10
        mem_next = leaked + acc + int(b1[j]) - (THRESH if prev1[j] else 0)
        mem_next = max(SAT_MIN, min(SAT_MAX, mem_next))
        s = 1 if mem_next >= THRESH else 0
        mem1[j] = mem_next; prev1[j] = s; spk1[j] = s
    fc1_spikes_all.append(spk1.copy())    # = fc2's input (256,)

    # ---- fc2 ----
    spk2 = np.zeros(N_OUT2, dtype=np.int64)
    for j in range(N_OUT2):
        acc = int((w2[j] * spk1).sum())
        beta_mem = BETA3 * int(mem2[j]); leaked = beta_mem >> 10
        mem_next = leaked + acc + int(b2[j]) - (THRESH if prev2[j] else 0)
        mem_next = max(SAT_MIN, min(SAT_MAX, mem_next))
        s = 1 if mem_next >= THRESH else 0
        mem2[j] = mem_next; prev2[j] = s; spk2[j] = s
    fc2_spikes_all.append(spk2.copy())
    fc2_mem_all.append(mem2.copy())

# ---- write files for the top-level testbench ----
with open(os.path.join(OUT, "full_conv1_spike_in.b"), "w") as f:
    for t in range(2):
        for ci in range(N_CIN):
            for pos in range(N_POS):
                f.write("%d\n" % spk_in0[t, ci, pos])

with open(os.path.join(OUT, "full_conv1_spikes_expected.txt"), "w") as f:
    for t in range(2):
        for v in conv1_spikes_all[t]: f.write("%d\n" % v)

with open(os.path.join(OUT, "full_fc1_spikes_expected.txt"), "w") as f:
    for t in range(2):
        for v in fc1_spikes_all[t]: f.write("%d\n" % v)

with open(os.path.join(OUT, "full_fc2_spikes_expected.txt"), "w") as f:
    for t in range(2):
        for v in fc2_spikes_all[t]: f.write("%d\n" % v)

with open(os.path.join(OUT, "full_fc2_mem_expected.hex"), "w") as f:
    for t in range(2):
        for v in fc2_mem_all[t]: f.write("{:04x}\n".format(int(v) & 0xFFFF))

print("conv1 spikes t0/t1:", int(conv1_spikes_all[0].sum()), int(conv1_spikes_all[1].sum()), "/", N_OUT0)
print("fc1   spikes t0/t1:", int(fc1_spikes_all[0].sum()), int(fc1_spikes_all[1].sum()), "/", N_OUT1)
print("fc2   spikes t0/t1:", fc2_spikes_all[0].tolist(), fc2_spikes_all[1].tolist())
print("fc2   mem    t0/t1:", fc2_mem_all[0].tolist(), fc2_mem_all[1].tolist())
print("wrote full_conv1_spike_in.b, full_{conv1,fc1,fc2}_spikes_expected.txt, full_fc2_mem_expected.hex")
