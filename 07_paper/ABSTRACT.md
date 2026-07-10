# Draft Abstract (paste-ready, real numbers)

> Edit for the confirmed venue's word limit. Every number here is real and sourced; the only
> placeholder is the measured-power figure, marked `[TODO]`.

---

Surface electromyography (sEMG) enables non-invasive muscle-fatigue monitoring, but wearable
deployment demands inference at extremely low energy. Spiking neural networks (SNNs) offer
event-driven, low-power computation, yet prior neuromorphic sEMG work addresses only gesture
recognition — muscle fatigue has not been tackled with SNNs. We present the first SNN-based
sEMG fatigue classifier, trained and evaluated on the recent Cerqueira et al. (2024) dataset
(13 subjects, 4-channel sEMG). Under a rigorous, leakage-free 13-fold leave-one-subject-out
protocol with pooled predictions and a paired significance test, our SNN reaches 78.3%
macro-F1, statistically indistinguishable from an iso-architecture CNN (78.0%; McNemar
p=0.354), while requiring approximately 28× less compute energy under the Horowitz 45 nm
model. Beyond simulation, we implement the full network — convolutional and fully-connected
spiking layers — as a time-multiplexed fixed-point (Q6.10) datapath in Verilog, verified
bit-exact against a software reference both per-layer and end-to-end, and synthesized with an
open-source flow (Yosys/nextpnr) onto a Lattice iCE40 device (9–33% logic per layer, ~16–17
MHz). [TODO: an absolute energy-per-inference of X µJ is measured on hardware via
vendor power analysis.] To our knowledge this is the first SNN sEMG study carried to verified
FPGA RTL rather than leaving hardware as future work, demonstrating a concrete path toward
on-body neuromorphic fatigue monitoring.

---

**Keyword candidates:** spiking neural networks; surface EMG; muscle fatigue; neuromorphic
computing; FPGA; energy-efficient inference; leave-one-subject-out; fixed-point.
