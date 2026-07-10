"""
Generate a deterministic 2-timestep stimulus for the fc1 layer (528->256)
and compute the bit-exact Q6.10 integer reference the RTL must reproduce.
Writes:
  spike_in.b        : 2*528 lines of 0/1 (input spikes, t0 then t1)  [$readmemb]
  spikes_expected.txt : 2*256 lines of 0/1 (expected output spikes)
  mem_expected.txt  : 256 lines final membrane (signed int, Q6.10)
"""
import os, numpy as np
OUT = r"C:\Users\Asuss\Downloads\snn_fpga"
N_IN, N_OUT = 528, 256
BETA = 933      # lif2 beta 0.91084 in Q6.10
THRESH = 307    # 0.30 in Q6.10
SAT_MAX, SAT_MIN = 32767, -32768

w = np.loadtxt(os.path.join(OUT, "fc1_w_q610.hex"), dtype=str)
w = np.array([int(x, 16) for x in w], dtype=np.int64)
w = np.where(w >= 0x8000, w - 0x10000, w).reshape(N_OUT, N_IN)  # int16 two's comp
b = np.loadtxt(os.path.join(OUT, "fc1_b_q610.hex"), dtype=str)
b = np.array([int(x, 16) for x in b], dtype=np.int64)
b = np.where(b >= 0x8000, b - 0x10000, b)

# deterministic input spikes
spk_in = np.zeros((2, N_IN), dtype=np.int64)
for t in range(2):
    for i in range(N_IN):
        spk_in[t, i] = 1 if ((i * (t + 1) + 3) % 5 == 0) else 0

mem = np.zeros(N_OUT, dtype=np.int64)
prev = np.zeros(N_OUT, dtype=np.int64)
exp_spk = np.zeros((2, N_OUT), dtype=np.int64)

for t in range(2):
    for j in range(N_OUT):
        acc = int((w[j] * spk_in[t]).sum())     # AC: only spiked inputs contribute
        beta_mem = BETA * int(mem[j])
        leaked = beta_mem >> 10                  # arithmetic shift (>>> on signed)
        mem_next = leaked + acc + int(b[j]) - (THRESH if prev[j] else 0)
        mem_next = max(SAT_MIN, min(SAT_MAX, mem_next))
        s = 1 if mem_next >= THRESH else 0
        mem[j] = mem_next
        prev[j] = s
        exp_spk[t, j] = s

with open(os.path.join(OUT, "spike_in.b"), "w") as f:
    for t in range(2):
        for i in range(N_IN):
            f.write("%d\n" % spk_in[t, i])
with open(os.path.join(OUT, "spikes_expected.txt"), "w") as f:
    for t in range(2):
        for j in range(N_OUT):
            f.write("%d\n" % exp_spk[t, j])
with open(os.path.join(OUT, "mem_expected.txt"), "w") as f:
    for j in range(N_OUT):
        f.write("%d\n" % mem[j])

print("t0 output spikes:", int(exp_spk[0].sum()), "/ 256")
print("t1 output spikes:", int(exp_spk[1].sum()), "/ 256")
print("wrote spike_in.b, spikes_expected.txt, mem_expected.txt")
