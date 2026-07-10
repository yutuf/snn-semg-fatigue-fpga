# Roadmap to IEEE HPEC

Concrete path from this handoff to a submitted paper. Priority-ordered.

---

## Where the project stands

### DONE and verified ✅
- Leakage-free, subject-independent evaluation (13-fold LOSO), pooled with a significance
  test — matches the rigor of the companion lab papers.
- Final model trained and locked (`snn_fatigue_final.pth`); float macro-F1 82.9%.
- Headline result: **SNN 78.3% vs CNN 78.0%, McNemar p=0.354; ~28× lower energy.**
- Weights quantized to Q6.10 and int8; 16-bit verified lossless (82.88% vs 82.92% float).
- Full network in Verilog (conv1→fc1→fc2), **bit-exact verified per-layer and end-to-end**.
- Open-source synthesis (Yosys/nextpnr, iCE40): real resource + Fmax; latency measured in sim.
- Memory bottleneck (fc1) quantified with a real synthesis attempt, not just estimated.

### OPEN ⏳ (the four things between here and submission)
| # | Task | Blocking for | Effort | Owner |
|---|---|---|---|---|
| 1 | **Measured power** (`report_power`) → real joules/inference | strongest HPEC claim | medium | professor (FPGA) |
| 2 | **Confirm int8 accuracy** end-to-end | the 8-bit memory story | small | student/Colab |
| 3 | **fc1 weight-sourcing decision** (SPI-flash→SPRAM / int8 on-chip / Artix BRAM) | a complete, buildable design | medium | professor (FPGA) |
| 4 | **Write & submit the paper** | submission | medium | shared; outline in `07_paper/` |

Latency optimization (parallel cores, event-driven sparsity) is **optional upside**, not
required for submission — see `04_synthesis_results/LATENCY_AND_ENERGY.md`.

---

## Priority-ordered plan

**Step 1 — Lock the venue & confirm scope (do first, zero cost).**
Confirm with the professor: is the target **HPEC** or the lab's **IDAP'26**? The two
companion papers go to IDAP'26; this one is being positioned for HPEC because the
hardware/energy angle fits HPEC better and there's nothing to lose (see cover note).
This decision sets the page limit, template, and how "finished" the FPGA piece must look.

**Step 2 — Get the measured power number (task 1 & 3 together).**
This is the single biggest differentiator: both companion papers leave FPGA as *future
work*; a real measured joules/inference makes this the only one of the three with silicon-
grounded energy. Recommended path: **port to Artix-7 in Vivado** (fc1 fits BRAM natively,
sidestepping the iCE40 SPRAM work), run synth + implement + `report_power`, multiply by the
measured cycle count ÷ achieved Fmax. See `README_HANDOFF.md` §6.

**Step 3 — Confirm int8 accuracy (task 2, parallel, cheap).**
int8 weights are already exported. Run the locked model with int8 weights through the eval
harness; if it holds near 82.9%, the 8-bit on-chip memory path becomes a clean fallback to
the external-flash approach. One Colab session.

**Step 4 — Draft the paper (task 4).**
Everything needed is in `07_paper/` — outline, all real numbers, table/figure plan. Expand
section by section. The `training_journey_report.md` is the source material; the paper is
its disciplined, results-first distillation.

**Step 5 — Figures & final polish.**
Generate the figures listed in `07_paper/FIGURES_AND_TABLES.md` from the real data
(LOSO per-fold bars, energy roadmap, resource/latency table, pipeline diagram). Internal
consistency pass (every number in text matches its table), then submit.

---

## Submission checklist
- [ ] Venue confirmed (HPEC vs IDAP'26) and correct IEEE template in use
- [ ] Author list & affiliations agreed (student, professor, any co-authors)
- [ ] Abstract states the real numbers (78.3%/78.0%, p=0.354, ~28×)
- [ ] Measured power number obtained and in the energy table (or clearly framed as the one projected value if not)
- [ ] int8 accuracy confirmed (or 16-bit-only claim made cleanly)
- [ ] All figures generated from real data (no placeholder boxes)
- [ ] Every number in prose matches its table (consistency pass)
- [ ] Related-work paragraph explicitly differentiates from the two companion lab papers
- [ ] Reproducibility statement points to dataset DOI 10.5281/zenodo.14182446 + this artifact's DOI 10.5281/zenodo.21297570
- [ ] Page/format limits met for the confirmed venue

---

## HPEC format notes (verify against the current Call for Papers)
IEEE HPEC is an annual IEEE conference (typically September, Boston area) with a strong
focus on high-performance and extreme computing, increasingly including ML acceleration and
neuromorphic hardware — a good fit for this work's energy/hardware angle. It typically
accepts both short (~2-page extended abstract) and full (~6-page IEEE double-column)
submissions across tracks.
**⚠ Do not take dates/lengths from this file as authoritative** — pull the exact page
limit, template, track, and deadline from the current year's official CfP before writing.

## Open questions for the professor
1. **Venue:** HPEC or IDAP'26 for this paper?
2. **Authorship:** author order and affiliations?
3. **FPGA target:** stay on iCE40 (matches the prior-art gesture paper's device), or move to
   Artix-7 (simpler memory, easier power measurement)? Affects how the hardware section reads.
4. **Scope of the FPGA claim:** is measured power required for this submission, or is
   verified RTL + resource/latency + analytical energy sufficient with power as stated future
   work? (Either is defensible; it changes urgency of task 1.)
