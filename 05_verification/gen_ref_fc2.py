"""
Deterministic 2-timestep stimulus + bit-exact Q6.10 reference for fc2
(256->2, the output layer). Verifies BOTH spikes and the exposed
membrane potential, since classification reads the membrane (mem.mean(0)
in training), not the spike count.
"""
import os, numpy as np
OUT = r"C:\Users\Asuss\Downloads\snn_fpga"
N_IN, N_OUT = 256, 2
BETA = 907      # lif3 beta 0.88531 in Q6.10
THRESH = 307    # 0.30 in Q6.10
SAT_MAX, SAT_MIN = 32767, -32768

w = np.loadtxt(os.path.join(OUT, "fc2_w_q610.hex"), dtype=str)
w = np.array([int(x, 16) for x in w], dtype=np.int64)
w = np.where(w >= 0x8000, w - 0x10000, w).reshape(N_OUT, N_IN)
b = np.loadtxt(os.path.join(OUT, "fc2_b_q610.hex"), dtype=str)
b = np.array([int(x, 16) for x in b], dtype=np.int64)
b = np.where(b >= 0x8000, b - 0x10000, b)

spk_in = np.zeros((2, N_IN), dtype=np.int64)
for t in range(2):
    for i in range(N_IN):
        spk_in[t, i] = 1 if ((i * (t + 2) + 5) % 6 == 0) else 0

mem = np.zeros(N_OUT, dtype=np.int64)
prev = np.zeros(N_OUT, dtype=np.int64)
exp_spk = np.zeros((2, N_OUT), dtype=np.int64)
exp_mem = np.zeros((2, N_OUT), dtype=np.int64)

for t in range(2):
    for j in range(N_OUT):
        acc = int((w[j] * spk_in[t]).sum())
        beta_mem = BETA * int(mem[j])
        leaked = beta_mem >> 10
        mem_next = leaked + acc + int(b[j]) - (THRESH if prev[j] else 0)
        mem_next = max(SAT_MIN, min(SAT_MAX, mem_next))
        s = 1 if mem_next >= THRESH else 0
        mem[j] = mem_next; prev[j] = s
        exp_spk[t, j] = s; exp_mem[t, j] = mem_next

with open(os.path.join(OUT, "fc2_spike_in.b"), "w") as f:
    for t in range(2):
        for i in range(N_IN): f.write("%d\n" % spk_in[t, i])
with open(os.path.join(OUT, "fc2_spikes_expected.txt"), "w") as f:
    for t in range(2):
        for j in range(N_OUT): f.write("%d\n" % exp_spk[t, j])
with open(os.path.join(OUT, "fc2_mem_expected.hex"), "w") as f:
    for t in range(2):
        for j in range(N_OUT):
            f.write("{:04x}\n".format(int(exp_mem[t, j]) & 0xFFFF))

print("t0: spikes", exp_spk[0].tolist(), "mem", exp_mem[0].tolist())
print("t1: spikes", exp_spk[1].tolist(), "mem", exp_mem[1].tolist())
print("wrote fc2_spike_in.b, fc2_spikes_expected.txt, fc2_mem_expected.txt")
