"""Неявная (BE/CN) версия схемы TempaFV как ЛОСС: состояние = h_c (N ячеек) + Gin (скаляр); невязки:
 (E) энергия: (ρh)_{n+1}-(ρh)_n)/dt + θ·F_E(n+1) + (1-θ)·F_E(n) = 0, F_E = donor-cell (G h)_z − NQ  [N уравнений]
 (M) интегральный импульс: (IG_{n+1}-IG_n)/dt = Δπ* − θ·dpc(n+1) − (1-θ)·dpc(n), IG = ∫G dz, G(z) = Gin − ∫∂ρ/∂t
Масса — тождество (G реконструируется). Точный решатель: L-BFGS по значениям (Ньютон-эквивалент). Явный TempaFV = референс."""
import json, os, sys, time, math
import matplotlib; matplotlib.use("Agg")
import numpy as np, torch
nb = json.load(open("pinn_v11_fv_tempa_baseline.ipynb")); g = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or i > 24: continue
    src = "".join(c["source"])
    if i == 23: src = src.split("# --- проверка табуляции")[0]      # без случайной проверки
    exec(compile(src, f"c{i}", "exec"), g)
case0, physics, np_ = g["case"], g["physics"], np
TempaFV = g["TempaFV"]
N = int(os.environ.get("N", "60")); THETA = float(os.environ.get("THETA", "1.0")); DT_S = float(os.environ.get("DT_S", "0.1"))
T0_S = float(os.environ.get("T0_S", "1400")); ITERS = int(os.environ.get("ITERS", "150")); T_END_S = float(os.environ.get("T_END_S", "2100"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype = torch.float64

# --- референс: явный TempaFV (blasius, acc=0, cons=True), продлённая рампа как в ноутбуке ---
fv = TempaFV(physics, case0, N=N, friction="blasius", acc=0.0, cons=True)
case = fv.case; tab = fv.tab; dz = 1.0/N; h_in = fv.h_in; rho_in = fv.rho_in
K = case.L/case.w0; k_flow = case.rho_pc*case.w0*case.A
# продление мощности линейно (как в ячейке 26 ноутбука): используем forcing.power_torch за пределами? -> NQ_prime клампит по np.interp (держит последнее). Возьмём линейное продление 21.34 Вт/с.
forcing = g["forcing"]
slope = 21.34
def NQ_of_t(t_star):
    tau = t_star*K
    if tau <= forcing.tau[-1]: Q = float(np.interp(tau, forcing.tau, forcing.power))
    else: Q = float(forcing.power[-1] + slope*(tau - forcing.tau[-1]))
    return case.beta_pc/case.Cp_pc*Q/(case.rho_pc*case.w0*case.A)
class _PH:  # подмена NQ_prime для fv.run с продлением
    def __init__(s, ph): s.ph = ph
    def __getattr__(s, k): return getattr(s.ph, k)
    def NQ_prime(s, t): return torch.tensor([NQ_of_t(float(t.reshape(-1)[0]))])
fv.ph = _PH(physics)
T_END = (T_END_S - 2.0)/K
t_w = time.time(); ref = fv.run(T=T_END, dt_sec=0.004, rec_dt_sec=0.1, n_snap=2000)
print(f"явный TempaFV N={N}: {time.time()-t_w:.0f} s; onset-зона: ", end="")
tr, gr = ref["t_sec"], ref["Gin_kgs"]
for lo, hi in [(1600,1700),(1700,1760),(1760,1820),(1820,1900),(1900,2000)]:
    m=(tr>=lo)&(tr<hi); print(f"{lo}-{hi}: p2p={np.ptp(gr[m]):.4f}", end="  ")
print(flush=True)

# --- torch-версии замыканий (табуляции TempaFV -> кусочно-линейно, дифференцируемо) ---
hs_t = torch.tensor(tab.hs, dtype=dtype, device=dev); rho_t = torch.tensor(tab.rho_sp(tab.hs), dtype=dtype, device=dev)
def rho_of(h):
    hc = h.clamp(hs_t[0], hs_t[-1]); i = torch.searchsorted(hs_t, hc.contiguous(), right=True).clamp(1, hs_t.numel()-1)
    w = (hc-hs_t[i-1])/(hs_t[i]-hs_t[i-1]); return (1-w)*rho_t[i-1]+w*rho_t[i]
mu_h = torch.tensor(g["mu_spline"].h_grid, dtype=dtype, device=dev); mu_v = torch.tensor(g["mu_spline"].mu_grid, dtype=dtype, device=dev)
def Lambda_dyn(G, h):
    h_abs = case.h_pc + case.Cp_pc/case.beta_pc*h.clamp(tab.h_lo, tab.h_hi)
    hc = h_abs.clamp(mu_h[0], mu_h[-1]); i = torch.searchsorted(mu_h, hc.contiguous(), right=True).clamp(1, mu_h.numel()-1)
    w = (hc-mu_h[i-1])/(mu_h[i]-mu_h[i-1]); mu = (1-w)*mu_v[i-1]+w*mu_v[i]
    Re = torch.clamp(case.rho_pc*case.w0*torch.sqrt(G*G+1e-12)*case.Dh/mu, min=1.0)
    lam = (1-torch.sigmoid((Re-2300.)/200.))*64./Re + torch.sigmoid((Re-2300.)/200.)*0.3164/Re**0.25
    return lam*case.L/(2.*case.Dh)
def dpc(Gin, G, rho, h):
    Gout, rout = G[-1], rho[-1]
    return (rho.sum()*dz/case.Fr + (Lambda_dyn(G, h)*G*G.abs()/rho).sum()*dz
            + 0.5*case.kin*Gin*Gin.abs()/rho_in + 0.5*case.kout*Gout*Gout.abs()/rout)
def F_E(h, G, Gin, NQ):
    """donor-cell энергия: (G h)_z с донором слева, вход = h_in, Gin"""
    Gp = torch.clamp(G, min=0.0)
    hup = torch.cat([torch.full((1,), h_in, dtype=dtype, device=dev), h[:-1]]); Gf = torch.cat([Gin.reshape(1), Gp[:-1]])
    return (Gp*h - Gf*hup)/dz - NQ

def step_residuals(h1, Gin1, h0, G0, Gin0, rho0, IG0, t0, dt, theta):
    rho1 = rho_of(h1)
    G1 = Gin1 - dz*torch.cumsum((rho1-rho0)/dt, 0)                        # неразрывность (тождество)
    NQ0, NQ1 = NQ_of_t(t0), NQ_of_t(t0+dt)
    R_E = (rho1*h1 - rho0*h0)/dt + theta*F_E(h1, G1, Gin1, NQ1) + (1-theta)*F_E(h0, G0, Gin0, NQ0)
    IG1 = G1.sum()*dz
    R_M = (IG1 - IG0)/dt - (case.dpi_star - theta*dpc(Gin1, G1, rho1, h1) - (1-theta)*dpc(Gin0, G0, rho0, h0))
    return R_E/case.energy_scale, R_M/case.dpi_star, G1, rho1, IG1

# --- старт: состояние явного TempaFV при T0_S ---
i0 = int(np.argmin(np.abs(ref["tsnap"]*K+2.0 - T0_S)))
h0 = torch.tensor(ref["hsnap"][i0], dtype=dtype, device=dev); t0 = float(ref["tsnap"][i0])
j0 = int(np.argmin(np.abs(ref["t_sec"] - (t0*K+2.0)))); Gin0 = torch.tensor(float(ref["Gin"][j0]), dtype=dtype, device=dev)
rho0 = rho_of(h0); G0 = torch.full((N,), float(Gin0), dtype=dtype, device=dev); IG0 = G0.sum()*dz
print(f"неявный старт t={t0*K+2:.1f} s, Gin={float(Gin0)*k_flow:.4f} kg/s; θ={THETA}, dt={DT_S} s, N={N}, iters={ITERS}", flush=True)

dt = DT_S/K; t = t0
h_p = torch.nn.Parameter(h0.clone()); g_p = torch.nn.Parameter(Gin0.clone())
out = {"t": [], "Gin": [], "loss": []}; tw = time.time(); n = 0
while t*K+2 < T_END_S - 1e-6:
    opt = torch.optim.LBFGS([h_p, g_p], max_iter=ITERS, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-16)
    def closure():
        opt.zero_grad(); R_E, R_M, *_ = step_residuals(h_p, g_p, h0, G0, Gin0, rho0, IG0, t, dt, THETA)
        L = (R_E**2).mean() + R_M**2; L.backward(); return L
    opt.step(closure)
    with torch.no_grad():
        R_E, R_M, G1, rho1, IG1 = step_residuals(h_p, g_p, h0, G0, Gin0, rho0, IG0, t, dt, THETA)
        L = float((R_E**2).mean() + R_M**2)
        h0, G0, Gin0, rho0, IG0 = h_p.detach().clone(), G1.detach().clone(), g_p.detach().clone(), rho1.detach().clone(), float(IG1)
    t += dt; n += 1
    out["t"].append(t*K+2); out["Gin"].append(float(Gin0)*k_flow); out["loss"].append(L)
    if n % 500 == 0: print(f"шаг {n} t={out['t'][-1]:.1f} s Gin={out['Gin'][-1]:.4f} loss={L:.1e} ({time.time()-tw:.0f} s)", flush=True)
np.savez(f"implicit_th{THETA}_dt{DT_S}_N{N}.npz", **{k: np.array(v) for k, v in out.items()}, ref_t=tr, ref_G=gr)
tt = np.array(out["t"]); gg = np.array(out["Gin"])
print(f"\n{'окно':12s} {'неявн mean':>10s} {'p2p':>8s} | {'явн mean':>9s} {'p2p':>8s}")
for lo, hi in [(1400,1500),(1500,1600),(1600,1700),(1700,1760),(1760,1820),(1820,1900),(1900,2000),(2000,2100)]:
    m=(tt>=lo)&(tt<hi); mr=(tr>=lo)&(tr<hi)
    if m.sum()>2: print(f"{lo}-{hi}s {gg[m].mean():10.4f} {np.ptp(gg[m]):8.4f} | {gr[mr].mean():9.4f} {np.ptp(gr[mr]):8.4f}")
def onset(t_, g_):
    w=int(20/np.mean(np.diff(t_))/2); p2p=np.array([np.ptp(g_[max(0,j-w):j+w]) for j in range(len(g_))])
    base=p2p[(t_>1400)&(t_<1600)].mean(); k=np.where((p2p>10*base)&(t_>1500))[0]; return float(t_[k[0]]) if len(k) else float('nan')
on_i, on_e = onset(tt, gg), onset(tr, gr)
mt = tt > on_i+40 if np.isfinite(on_i) else tt > 1900
per = float('nan')
if mt.sum() > 40:
    s = gg[mt]-gg[mt].mean(); fr = np.fft.rfftfreq(len(s), d=float(np.mean(np.diff(tt[mt])))); sp = np.abs(np.fft.rfft(s)); per = 1/fr[int(np.argmax(sp[1:]))+1]
print(f"onset: неявный {on_i:.0f} s, явный TempaFV {on_e:.0f} s;  период неявного {per:.2f} s;  медиана лосса {np.median(out['loss']):.1e}")
