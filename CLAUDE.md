# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A physics-informed neural network (PINN) for the **stability of supercritical water flow** in a heated channel (a single-channel thermal-hydraulics stability study near the pseudo-critical point, ~25 MPa water). The network learns the transient 1D flow fields by minimizing PDE residuals (conservation of mass, momentum, energy) plus initial/boundary conditions — there is no labelled supervision on the interior. The reference transient data (`Churkin-Kout20-Nspc2.0-Transient-Water-Vertical.xlsx`) is used only as time-dependent forcing (power, inlet/outlet flow) and for validation.

The work lives almost entirely in Jupyter notebooks. Each `pinn_v*.ipynb` is a self-contained, sequentially-runnable iteration of the model; later versions supersede earlier ones rather than importing from them. The only reusable Python module is `eos_iapws_spline.py`.

## Layout

Everything is under `Supercritical_stability_PINN/`:
- `eos_iapws_spline.py` — the one shared module. Builds a 2D IAPWS97 `rho(p, h)` table and wraps it in a `RectBivariateSpline` (`IAPWSDensitySpline`) so density and its partials `drho/dp`, `drho/dh` are smooth and cheap inside the training loop.
- `iapws_density_spline_25mpa.npz` — cached EOS table. Generated once via `IAPWSDensitySpline.build()`, then loaded; rebuilding it calls IAPWS97 thousands of times (slow), so keep the cache.
- `pinn_v1_lambda=const.ipynb` → `pinn_v4_mass_reweighting.ipynb` — model iterations, newest is v4. (v3 has a broken-then-`_fixed` pair; prefer `pinn_v3_energy_from_mass_fixed_notebook.ipynb`.)
- `pinn_v3b_imposed_dp.ipynb` — branch off v3 that fixes the pressure boundary conditions to the *imposed-Δp* formulation (see "Physical setup" below). The active line of work as of June 2026.
- `Churkin-...xlsx` — reference transient (forcing + validation target).
- `summary_late.csv` — exported early/late diagnostics from a v4 run.
- `1-s2.0-S0306454910003282-main.pdf` — **the source paper**: Ambrosini, "Assessment of flow stability boundaries in a heated channel ... at supercritical pressure", *Annals of Nuclear Energy* 38 (2011) 615–627. Defines the dimensionless equations (its Eqs. 3–9) and the boundary conditions this PINN reproduces.
- `Churkin-Description.pdf` — TEMPA-SC benchmark description (OKB Gidropress). Specifies the exact boundary conditions and forcing behind the `Churkin-...xlsx` data: constant pressure/inlet-temperature, power as a function of time, `Δpch = 0.12 MPa`, `pout = 25 MPa`, `pin = pout + Δpch`.

## Running

There is no build/test/lint setup — this is a notebook research project. Develop by running notebook cells top-to-bottom.

```bash
source .venv/bin/activate            # Python 3.14 venv at repo root
# open a pinn_v*.ipynb and run cells in order, or:
jupyter lab Supercritical_stability_PINN/
```

Key dependencies (already in `.venv`): `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `iapws`, `openpyxl` (for the .xlsx), `ipykernel`.

**Working directory matters.** Notebooks load `iapws_density_spline_25mpa.npz` and the `.xlsx` by *relative* path, so run them with the CWD set to `Supercritical_stability_PINN/`. They also do `sys.path.append("/mnt/data")` and import `eos_iapws_spline` — a leftover from a cloud sandbox. On this machine the import resolves because the notebook's own directory is already on `sys.path`; the `/mnt/data` append is harmless but meaningless here.

Training is CPU-friendly but slow; the code auto-selects CUDA if present (`torch.device("cuda" if ... else "cpu")`).

**Reading the PDFs.** The Read tool needs poppler (not installed) to render PDFs here. Instead use `pypdf` inside the venv: `pip install pypdf -q` then `PdfReader(path).pages[i].extract_text()`. `Churkin-Description.pdf` has many "wrong pointing object" warnings and its equation glyphs extract as garbage — the prose (boundary conditions, methodology) extracts fine, the math does not.

## Physical setup — boundary conditions (this is the crux of the project)

The whole point is to reproduce **density-wave / flow-rate oscillations** that arise as a supercritical-pressure instability. Both source documents agree on how the channel is driven, and getting this right in the PINN is what makes oscillations possible:

- **Pressure drop Δp* across the channel is *imposed and constant*** — Ambrosini: "the value of the pressure drop ... is imposed across the channel, **letting flow rate to freely oscillate**." Churkin: `Δpch = 0.12 MPa`, `pout = 25 MPa`, `pin = pout + Δpch`. This is **one** constraint on the *difference* `π*(0,t) − π*(1,t) = Δπ*`, plus a gauge anchor `π*(1,t) = 0`. It must NOT be two independent fixings of `π*` at each end.
- **Inlet enthalpy is imposed**: `h*(0,t) = −NSPC` (constant).
- **Mass flux is free.** `G*_in = ρ*_in` holds only as an *initial condition* at `t=0` (it's in `ic_loss`), never as a time-wise boundary condition — pinning it would over-constrain the system and kill oscillations.
- **The instability is excited by slowly rising power** `Q(t)` from the xlsx (`NQ_prime(t)`). Rising outlet temperature → falling flow rate → more heating → density-wave oscillations.

**The bug fixed in `pinn_v3b_imposed_dp.ipynb`:** v3/v4 `bc_loss` fixed `π*` at *both* ends independently (`bc_pi_in` + `bc_pi_out`), which over-constrains the problem so the net learns a straight pressure line and the flow cannot oscillate. v3b replaces that with `R_dp = (pi_in − pi_out)/dpi_star − 1` (imposed Δp) + `R_pi_out = pi_out/dpi_star` (gauge), logged as `bc_dp` instead of `bc_pi_in`. Reference NTPC instability threshold for this case (Kout=20, NSPC≈1.5–2) is ≈ 2.9 (Churkin Table 2).

Open question for the next iteration: even with correct BCs, a smooth tanh net minimizing MSE residuals tends to average the instability to zero. If oscillations still don't appear, the planned next step is Fourier features **in time** + the v4 causal-curriculum (finer time stages) — `SirenPINN`/`HybridPINN`/`FourierFeatures` already exist in the notebook for exactly this, but there is no physical trigger wired up yet.

## Architecture of a notebook (v4 is the canonical reference)

Cells are organized as a pipeline. The conceptual flow:

1. **`Forcing.from_excel`** — parses the transient: power `Q(t)`, inlet/outlet flow, NTPC. Establishes the baseline mass flux `mdot0` from the pre-ramp window and shifts time so `tau=0` is ramp start.
2. **EOS setup** — load/build the density spline; build a separate `IAPWSViscositySpline` (1D, `mu(h)` at fixed `p_ref`) for the friction term.
3. **`Case.build`** — the central object. Computes all geometry, pseudo-critical reference state (`rho_pc`, `h_pc`, `Cp_pc`, `beta_pc`), and especially the **non-dimensionalization scales** (`G0_star`, `dpi_star`, `Fr`, and per-equation loss scales `mass_scale`, `mom_mid_scale`, `mom_delta_scale`, `energy_scale`). Almost every physics term is expressed in `*`-starred dimensionless variables defined here.
4. **Model** — three interchangeable nets, all mapping `(z, t) -> (G_hat, h_hat, Pi_hat)`: `SmoothPINN` (tanh MLP, the default), `HybridPINN` (smooth MLP + gated Fourier-feature branch for sharp fronts), `SirenPINN` (sine activations). Inputs are rescaled to `[-1, 1]`; outputs are dimensionless field "hats" that `Physics.fields` multiplies by the `Case` scales to recover `G`, `h`, `pi`.
5. **`Physics`** — the residual engine. `residuals_raw` builds mass/momentum/energy residuals using `d(y, x)` (autograd) for derivatives and chain-rule density derivatives from the EOS spline (the spline is **not** differentiated by autograd — its analytic partials are injected via the `value + slope*(x - x.detach())` trick so gradients flow correctly; see `eos25`, `IAPWSViscositySpline.eval_torch`). `residuals_scaled` divides each residual by its `Case` scale, with momentum optionally normalized *relative* to the sum of its own term magnitudes (`relative_mom`). The channel is split into three z-zones — `in`, `mid`, `out` — so the inlet/outlet local-loss spikes (smooth Gaussian `delta_inlet`/`delta_outlet`) get their own loss weighting.
6. **`LossWeights` + `Trainer`** — weighted sum of zone PDE losses + IC + BC, Adam, grad-norm clipping.
7. **`train_causal_curriculum`** — the actual training driver. Trains in **time stages** (`frac` of the full horizon: 0.25 → 0.5 → 0.75 → 1.0), expanding the active time window via `sampler.set_active_time` so the network learns causally from `t=0` forward. This is the intended way to train, not the plain loop in the commented-out cell.
8. **Diagnostics** (the back half of the notebook) — `validate` (PINN flow/NTPC vs. reference), `pressure_budget` (integral momentum balance: accel, gravity, friction, local losses, flux jump), `pseudo_critical_indicator` (`S_pc` — proximity of enthalpy to the pseudo-critical point), heatmaps, and `summarize_early_late` which produces `summary_late.csv`.

## Conventions and gotchas

- **Starred = dimensionless.** A trailing `_star`/`*` or `_hat` denotes the network's working variables. Dimensional quantities use SI internally (Pa, J/kg, kg/m³), but the IAPWS97 library wants MPa and kJ/kg — note the `/1e6` and `/1e3` conversions whenever IAPWS97 is called.
- **EOS derivatives are analytic, gradients are manual.** Never expect autograd to flow through `RectBivariateSpline` or `CubicSpline`. The `x - x.detach()` pattern is deliberate — preserve it when editing residuals.
- **Three-zone sampling is structural**, not cosmetic. The inlet/outlet zones exist because the local pressure losses are near-singular there; the Gaussian `delta` half-widths are tied to `alpha` (default 0.02) and the sampler boundaries (`3*alpha`).
- **The `Case` field list is append-only by intent.** Note duplicated fields (`roughness`, `friction_model`, etc. appear twice in the dataclass) and the comment "оставляем поле Lambda, чтобы старый код не ломался" — fields are kept around so older cells/notebooks don't break. Be cautious removing any.
- **Iterate by copying a notebook to a new `pinn_v{N+1}_...ipynb`**, matching the established cell structure, rather than editing an old version in place — that is the project's versioning scheme.
- Comments and print labels are partly in Russian; match the surrounding language when editing a cell.
