"""
Generate a deterministic 2-timestep stimulus for the conv1+LIF1 layer
(Conv1d(4,16,k=5,pad=2) with true weight sharing across 33 positions)
and compute the bit-exact Q6.10 integer reference the RTL must reproduce.

Mirrors gen_ref.py's methodology exactly, extended to conv1's
weight-shared, zero-padded convolution structure.

Writes (to snn_fpga/):
  conv1_spike_in.b        : 2*132 lines of 0/1 (input spikes, t0 then t1)
  conv1_spikes_expected.txt : 2*528 lines of 0/1 (expected output spikes)
  conv1_mem_expected.txt  : 528 lines final membrane (signed int, Q6.10)
"""
import os, numpy as np
OUT = r"C:\Users\Asuss\Downloads\snn_fpga"
N_CIN, N_COUT, K, PAD, N_POS = 4, 16, 5, 2, 33
N_IN, N_OUT = N_CIN*N_POS, N_COUT*N_POS     # 132, 528
BETA = 841       # lif1 beta 0.82164 in Q6.10
THRESH = 307     # 0.30 in Q6.10
SAT_MAX, SAT_MIN = 32767, -32768

w = np.loadtxt(os.path.join(OUT, "conv1_w_q610.hex"), dtype=str)
w = np.array([int(x, 16) for x in w], dtype=np.int64)
w = np.where(w >= 0x8000, w - 0x10000, w)               # int16 two's comp, flat (320,)
w = w.reshape(N_COUT, N_CIN, K)                          # [c_out, c_in, k]
b = np.loadtxt(os.path.join(OUT, "conv1_b_q610.hex"), dtype=str)
b = np.array([int(x, 16) for x in b], dtype=np.int64)
b = np.where(b >= 0x8000, b - 0x10000, b)                # (16,)

# deterministic input spikes (different pattern from fc1's ref, still reproducible)
spk_in = np.zeros((2, N_CIN, N_POS), dtype=np.int64)
for t in range(2):
    for ci in range(N_CIN):
        for pos in range(N_POS):
            spk_in[t, ci, pos] = 1 if ((ci*7 + pos*(t+1) + 2) % 4 == 0) else 0

mem = np.zeros((N_COUT, N_POS), dtype=np.int64)
prev = np.zeros((N_COUT, N_POS), dtype=np.int64)
exp_spk = np.zeros((2, N_COUT, N_POS), dtype=np.int64)

for t in range(2):
    for co in range(N_COUT):
        for p in range(N_POS):
            acc = 0
            for ci in range(N_CIN):
                for k in range(K):
                    pos_in = p + k - PAD
                    if 0 <= pos_in < N_POS:
                        if spk_in[t, ci, pos_in]:
                            acc += int(w[co, ci, k])
            beta_mem = BETA * int(mem[co, p])
            leaked = beta_mem >> 10
            mem_next = leaked + acc + int(b[co]) - (THRESH if prev[co, p] else 0)
            mem_next = max(SAT_MIN, min(SAT_MAX, mem_next))
            s = 1 if mem_next >= THRESH else 0
            mem[co, p] = mem_next
            prev[co, p] = s
            exp_spk[t, co, p] = s

# flat layouts matching RTL addressing: input in_addr=c_in*33+pos, output j=c_out*33+p
spk_in_flat = spk_in.reshape(2, N_IN)
exp_spk_flat = exp_spk.reshape(2, N_OUT)
mem_flat = mem.reshape(N_OUT)

with open(os.path.join(OUT, "conv1_spike_in.b"), "w") as f:
    for t in range(2):
        for i in range(N_IN):
            f.write("%d\n" % spk_in_flat[t, i])
with open(os.path.join(OUT, "conv1_spikes_expected.txt"), "w") as f:
    for t in range(2):
        for j in range(N_OUT):
            f.write("%d\n" % exp_spk_flat[t, j])
with open(os.path.join(OUT, "conv1_mem_expected.txt"), "w") as f:
    for j in range(N_OUT):
        f.write("%d\n" % mem_flat[j])

print("t0 output spikes:", int(exp_spk_flat[0].sum()), "/", N_OUT)
print("t1 output spikes:", int(exp_spk_flat[1].sum()), "/", N_OUT)
print("wrote conv1_spike_in.b, conv1_spikes_expected.txt, conv1_mem_expected.txt")
