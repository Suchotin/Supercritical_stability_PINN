# Handoff — Supercritical-stability PINN

Context transfer for picking up this work. Read `CLAUDE.md` first for the codebase map and conventions; this file is the *current state of the investigation* and what to do next.

## The problem we're solving

A PINN for a heated channel with supercritical water (~25 MPa, the Ambrosini/Churkin benchmark) is supposed to reproduce **flow-rate (density-wave) oscillations** that the reference data shows. Two symptoms in the current model:

1. The pressure drop Δp* optimizes poorly — the network learns a nearly straight `π*(z)` line, two flat horizontals in the gauge-pressure plot.
2. The mass-flow oscillations present in the baseline (`Churkin-...xlsx`) **do not appear** — the PINN solution is smooth/damped.

The user has `SirenPINN` / `HybridPINN` / `FourierFeatures` classes in the notebook (capable of oscillations) but **no physical criterion** that decides when/where to use them — and that is not the real blocker (see diagnosis).

## Diagnosis (confirmed against both PDFs)

The root cause was a **mis-stated boundary condition**, not network capacity:

- Ambrosini 2010 (`1-s2.0-S0306454910003282-main.pdf`) and Churkin TEMPA-SC (`Churkin-Description.pdf`) both impose a **constant pressure drop** across the channel and **let the flow rate oscillate freely**. Quote (Ambrosini): *"the value of the pressure drop ... is imposed across the channel, letting flow rate to freely oscillate."* Churkin: `Δpch = 0.12 MPa`, `pout = 25 MPa`, `pin = pout + Δpch`; power assigned as a function of time; rising outlet temperature → falling flow → instability.
- The v3/v4 `bc_loss` instead fixed `π*` at **both** ends independently (`bc_pi_in` AND `bc_pi_out`) and the field code keeps inlet flux tied via the IC. Fixing both pressure ends over-constrains the system so the integral Δp can't vary and the flow has no freedom to oscillate.

## What was changed

Created **`Supercritical_stability_PINN/pinn_v3b_imposed_dp.ipynb`** (copy of `pinn_v3_energy from_mass.ipynb`) with a single, surgical change to `Physics.bc_loss` (cell 12):

```python
# OLD — two independent pressure fixings (over-constrained):
R_pi_in  = pi_in  / dpi_star - 1.0    # π*(0,t)=Δπ*
R_pi_out = pi_out / dpi_star          # π*(1,t)=0

# NEW — imposed Δp (one constraint on the difference) + gauge anchor:
R_dp     = (pi_in - pi_out) / dpi_star - 1.0   # π*(0,t)−π*(1,t)=Δπ*
R_pi_out = pi_out / dpi_star                   # π*(1,t)=0  (p_out=25 MPa reference)
```

- `bc_h_in` (`h*(0,t)=−NSPC`) unchanged — inlet enthalpy is imposed.
- `ic_loss` unchanged — `G*_in=ρ*_in` correctly stays a `t=0` initial condition only; flow is not pinned over time.
- Renamed the loss key `bc_pi_in` → `bc_dp` everywhere it's consumed: `total_loss` aggregation (cell 12), training-loop print (cell 15), `normalized_error_report` (cell 22). All cells syntax-check clean; no stale `bc_pi_in` references remain.

This was a **deliberately minimal change**: BCs only, no activation/architecture changes yet, per the user's "сначала только постановка" (setup first) decision.

## How to run / verify

From `Supercritical_stability_PINN/` (CWD matters — relative paths to the `.npz` and `.xlsx`):

```bash
source ../.venv/bin/activate
jupyter lab pinn_v3b_imposed_dp.ipynb   # run cells top-to-bottom; cell 14 builds model, cell 15 trains
```

Then inspect: does `flow-in/flow-out` from `validate` now show oscillations matching the baseline, and does `bc_dp` stay small during training?

## Decisions already made by the user (do not re-litigate)

- Oscillations should be **reproduced from physics** (forward PINN), not fitted via a supervised data-loss on the baseline flow.
- Pressure setup follows **Ambrosini/Churkin exactly**: imposed Δp, free flow.
- **Base further work on the v3 line** (`pinn_v3b_imposed_dp.ipynb`), not v4.
- This first step is **BCs only**; activations come later if needed.

## Next step if oscillations still don't appear

A smooth tanh net minimizing MSE residuals tends to average an instability to zero even with correct BCs (the oscillatory branch is not the minimum-residual attractor). Planned escalation, in order:

1. Port the **causal curriculum** from v4 (`train_causal_curriculum`, `sampler.set_active_time`) with **finer** time stages, so the net learns time causally instead of averaging the whole horizon.
2. Add **Fourier features in time `t`** specifically (currently `FourierFeatures` mixes `(z,t)` and `HybridPINN`'s gate is initialized near-off at logit −2.0), giving the net a periodic-in-time basis.
3. Re-examine momentum scaling / the `relative_mom` reweighting (present in v4, absent in v3) — dividing the momentum residual by the sum of its own term magnitudes can damp sharp transitions.

## Project memory

Durable facts are saved under the Claude memory dir for this project:
`~/.claude/projects/-Users-suchotin-pinn/memory/` — see `pinn-pressure-bc-physics.md` for the BC/forcing physics and the NTPC threshold (≈2.9 for Kout=20).
