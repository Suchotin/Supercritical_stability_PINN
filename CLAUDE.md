# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A physics-informed neural network (PINN) for the **stability of supercritical water flow** in a heated channel (single-channel thermal-hydraulics stability study near the pseudo-critical point, ~25 MPa water; the Ambrosini/Churkin TEMPA-SC benchmark, Kout=20, NSPC=2.0). The network learns the transient 1D flow fields by minimizing PDE residuals (mass, momentum, energy) plus initial/boundary conditions — **no labelled supervision on the interior**. The reference transient (`Churkin-Kout20-Nspc2.0-Transient-Water-Vertical.xlsx`) is used only as time-dependent forcing (power `Q(t)`) and for validation. The scientific goal is to reproduce the **density-wave oscillations (DWO)** from physics alone.

The work lives almost entirely in Jupyter notebooks. Each `pinn_v*.ipynb` is a self-contained, sequentially-runnable iteration; later versions supersede earlier ones rather than importing from them. The only reusable Python module is `eos_iapws_spline.py`.

**Start with `HANDOFF.md`** — it holds the current state of the investigation and decisions already made. The `tech_report_*.md` / `*.md` docs (below) record diagnoses in detail.

## Layout

Everything is in this directory (which is also the git repo root; the Python 3.10 `.venv` sits one level **above**, at `../`):

Code and data:
- `eos_iapws_spline.py` — the one shared module. Builds a 2D IAPWS97 `rho(p, h)` table wrapped in a `RectBivariateSpline` (`IAPWSDensitySpline`) so density and its partials `drho/dp`, `drho/dh` are smooth and cheap in the training loop.
- `iapws_density_spline_25mpa.npz` — cached EOS table. Rebuilding calls IAPWS97 thousands of times (slow) — keep the cache.
- `Churkin-...xlsx` — reference transient (forcing + validation target).
- `fv_reference_curve.npz` — mass-flow reference curve exported from the FV solver; used as the comparison baseline in later notebooks.
- `*.pt` — model checkpoints (`v3b_*.pt`, `v4_causal*.pt`, `physics_lstm_fv_dwo.pt`); saved/loaded by the "Checkpoint helpers" cell so diagnostics can run without retraining.

Notebook lineage (newest lines first):
- **`pinn_v4_causal_good_enough.ipynb`** — the canonical PINN base ("good_enough"). Cells 0–13 (Forcing → EOS → Case → models → Physics → Sampler/Trainer) are the reference implementation; cell 16 is causal training from scratch per Wang–Sankaran–Perdikaris (CMAME 2024); the back half is diagnostics + keeper ablation. `pinn_v4_causal_new.ipynb` / `_adv` variants are documented regressions/experiments (see `tech_report_causal_v4.md`).
- **`pinn_v4_fv_solver.ipynb`** — "Variant A": a differentiable finite-volume (method-of-lines) DWO solver built on the same `Physics`/`Case`/EOS objects, no neural net. With N=600, dt=0.05 it **does reproduce the oscillations from physics** (~1 min runtime) — it is the physics ground truth and the source of `fv_reference_curve.npz`. Its plot style is the canonical style for new notebooks.
- `physics_lstm_fv_dwo.ipynb` — physics-informed recurrent (Conv+LSTM) solver on the FV discretization; physics-only curriculum training, long causal rollout.
- `wrong_basin_test.ipynb` — diagnostic: warm-starts from the true hot fields to test whether the smooth-PINN loss basin excludes the oscillatory solution.
- `pinn_v1`…`pinn_v3b_*` — historical iterations. v3b fixed the over-constrained pressure BC (see below); `_mom_normed` adds relative momentum normalization. v3 has a broken-then-`_fixed` pair — prefer `pinn_v3_energy_from_mass_fixed_notebook.ipynb`.

Documents:
- `HANDOFF.md` — investigation state + decisions; read first.
- `tech_report_causal_v4.md` — why `good_enough` beats `new`; the causal-weighting math (annealed ε, causalized keepers).
- `normalizations_tech_report.md` — every non-dimensionalization scale in the code, with concrete numbers for this case.
- `pure_forward_recipe.md` — how to train into the physically correct basin without supervised data.
- `lit_review_forward_pinn_dwo.md` — literature: no prior forward-PINN-catches-DWO-in-supercritical work exists; this is the novelty gap.
- `pinn-pressure-bc-physics.md` — the BC physics note (imposed Δp, free flow).
- PDFs: `1-s2.0-S0306454910003282-main.pdf` — **the source paper** (Ambrosini, *Annals of Nuclear Energy* 38 (2011) 615–627; defines the dimensionless Eqs. 3–9 and BCs). `Churkin-Description.pdf` — TEMPA-SC benchmark spec (`Δpch = 0.12 MPa`, `pout = 25 MPa`, power vs. time). `Recurrent_DWO.pdf` and the other two Elsevier PDFs — method references for the recurrent/FV lines.

## Running

No build/test/lint setup — a notebook research project. Develop by running cells top-to-bottom.

```bash
cd Supercritical_stability_PINN/
source ../.venv/bin/activate     # Python 3.10 venv, one level up
jupyter lab .                    # open a pinn_v*.ipynb and run cells in order
```

Dependencies are pinned in `requirements.txt` and already installed in `.venv`: `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `iapws`, `openpyxl`, `ipykernel`, `pypdf`.

**Working directory matters.** Notebooks load the `.npz` and `.xlsx` by *relative* path — run with CWD set to this directory. The `sys.path.append("/mnt/data")` in cell 0 is a harmless cloud-sandbox leftover; `eos_iapws_spline` imports because the notebook's own directory is on `sys.path`.

Training is CPU-friendly but slow; code auto-selects CUDA when present.

**Reading the PDFs.** The Read tool lacks poppler here. Use `pypdf` in the venv: `PdfReader(path).pages[i].extract_text()`. `Churkin-Description.pdf` emits many "wrong pointing object" warnings and its equation glyphs extract as garbage — prose extracts fine, math does not.

## Physical setup — boundary conditions (the crux)

Both source documents agree on how the channel is driven; getting this right is what makes oscillations possible:

- **Pressure drop Δp\* across the channel is imposed and constant** — Ambrosini: "the value of the pressure drop ... is imposed across the channel, **letting flow rate to freely oscillate**." Churkin: `Δpch = 0.12 MPa`, `pout = 25 MPa`. This is **one** constraint on the *difference* `π*(0,t) − π*(1,t) = Δπ*` plus a gauge anchor `π*(1,t) = 0`. It must NOT be two independent fixings of `π*` at each end — that over-constrains the system and kills oscillations (the v3/v4 bug fixed in `pinn_v3b_imposed_dp.ipynb`: `R_dp = (pi_in − pi_out)/dpi_star − 1`, loss key `bc_dp`).
- **Inlet enthalpy is imposed**: `h*(0,t) = −NSPC` (constant).
- **Mass flux is free.** `G*_in = ρ*_in` holds only as an *initial condition* at `t=0` (in `ic_loss`), never as a time-wise BC.
- **The instability is excited by slowly rising power** `Q(t)` from the xlsx (`NQ_prime(t)`): rising outlet temperature → falling flow → more heating → density-wave oscillations. Reference NTPC instability threshold for this case ≈ 2.9 (Churkin Table 2).

Even with correct BCs, a smooth tanh net minimizing MSE residuals averages the instability to zero — the oscillatory branch is not the minimum-residual attractor (proven: the FV solver on the *same* physics objects does oscillate). That is why the active lines use causal training, time-Fourier features, and FV/recurrent discretizations.

## Standing decisions (do not re-litigate)

- Oscillations must be **reproduced from physics** (forward model), not fitted via a supervised data-loss on the baseline flow.
- Pressure setup follows **Ambrosini/Churkin exactly**: imposed Δp, free flow.
- The **causal-PINN line must converge from random init** — no warm start or marching-bootstrap of any kind.
- **New approaches build on the user's own base**: `pinn_v4_causal_good_enough.ipynb` cells 0–13 verbatim + the FV solver + the Ambrosini paper. Do not import constructions from rejected experimental lines.
- Plots in new notebooks follow the canonical style of the FV-solver notebook.

## Architecture of a notebook (`good_enough` cells 0–13 are the canonical base)

Cells form a pipeline:

1. **`Forcing.from_excel`** — parses the transient: power `Q(t)`, inlet/outlet flow, NTPC. Establishes baseline mass flux `mdot0` from the pre-ramp window; shifts time so `tau=0` is ramp start.
2. **EOS setup** — load/build the density spline; build a 1D `IAPWSViscositySpline` (`mu(h)` at fixed `p_ref`) for friction.
3. **`Case.build`** — the central object: geometry, pseudo-critical reference state (`rho_pc`, `h_pc`, `Cp_pc`, `beta_pc`), and the **non-dimensionalization scales** (`G0_star`, `dpi_star`, `Fr`, per-equation `mass_scale`/`mom_mid_scale`/`mom_delta_scale`/`energy_scale`). See `normalizations_tech_report.md` for the numbers.
4. **Models** — interchangeable nets mapping `(z, t) -> (G_hat, h_hat, Pi_hat)`: `SmoothPINN` (tanh MLP, default), `HybridPINN` (MLP + gated Fourier branch), `SirenPINN`. Inputs rescaled to `[-1, 1]`; output "hats" are multiplied by `Case` scales in `Physics.fields`.
5. **`Physics`** — the residual engine. `residuals_raw` builds mass/momentum/energy residuals with `d(y, x)` (autograd) and chain-rule EOS derivatives (the spline is **not** autograd-differentiated — analytic partials are injected via the `value + slope*(x - x.detach())` trick). `residuals_scaled` divides by `Case` scales, with optional `relative_mom` normalization. The channel is split into three z-zones — `in`, `mid`, `out` — because the local pressure losses (Gaussian `delta_inlet`/`delta_outlet`) are near-singular at the ends.
6. **`LossWeights` + `Trainer`** — weighted sum of zone PDE losses + IC + BC + keepers, Adam, grad-norm clipping.
7. **Causal training** (cell 16 in `good_enough`) — Wang+2024 causal weighting: per-time-slice weights `w_i = exp(-ε Σ_{j<i} L_j)` with annealed ε, causalized keepers, IC anchored at `t=0`. This replaced the older stage-based `train_causal_curriculum`.
8. **Diagnostics** — `validate` (PINN flow/NTPC vs. reference), `pressure_budget` (integral momentum balance), `pseudo_critical_indicator`, heatmaps, `summarize_early_late` (→ `summary_late.csv`), per-keeper diagnostic and keeper ablation, causal-front diagnostic.

The FV solver (`pinn_v4_fv_solver.ipynb`) reuses steps 1–3 and 5 (`eos25`, `distributed_friction`, `NQ_prime`) with no neural net: semi-implicit upwind finite volumes, `ExtendedEOS25` smooth tail extension, N=600 / dt=0.05.

## Conventions and gotchas

- **Starred = dimensionless.** `_star`/`*`/`_hat` are the network's working variables. Dimensional quantities are SI internally, but IAPWS97 wants MPa and kJ/kg — note the `/1e6`, `/1e3` conversions at every IAPWS97 call.
- **EOS derivatives are analytic, gradients are manual.** Never expect autograd to flow through `RectBivariateSpline`/`CubicSpline`; preserve the `x - x.detach()` pattern when editing residuals.
- **Three-zone sampling is structural**, not cosmetic. Gaussian `delta` half-widths are tied to `alpha` (default 0.02) and the sampler boundaries (`3*alpha`).
- **The `Case` field list is append-only by intent** — duplicated fields and the "оставляем поле Lambda, чтобы старый код не ломался" comment are deliberate; be cautious removing anything.
- **Iterate by copying a notebook to a new `pinn_v{N+1}_...ipynb`** matching the established cell structure, not by editing an old version in place.
- Comments and print labels are partly in Russian; match the surrounding language when editing a cell. Communicate with the user in Russian.
