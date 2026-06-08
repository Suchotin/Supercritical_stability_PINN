"""
eos_iapws_spline.py

Smooth tabular EOS tool for PINN-Stability.

Build a 2D IAPWS97 table rho = rho(p, h), then evaluate smooth spline values
and partial derivatives:

    rho, drho_dp|h, drho_dh|p

Units:
    p   : Pa
    h   : J/kg
    rho : kg/m^3
    drho_dp : kg/(m^3 Pa)
    drho_dh : kg/(m^3 J/kg)

Hard-EOS PINN use:
    PINN outputs [G*, h*, pi*]
    p = p_out + rho_pc * w0**2 * pi*
    h = h_pc + Cp_pc / beta_pc * h*
    rho* = rho(p, h) / rho_pc

Chain-rule coefficients:
    d rho* / d pi* = w0**2 * drho_dp
    d rho* / d h*  = Cp_pc / (beta_pc * rho_pc) * drho_dh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RectBivariateSpline
from iapws import IAPWS97

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class EOSGridSpec:
    """Grid specification for the IAPWS rho(p,h) table."""

    p_min: float = 24.5e6
    p_max: float = 25.5e6
    h_min: float = 0.75e6
    h_max: float = 4.20e6
    n_p: int = 64
    n_h: int = 256
    kx: int = 3
    ky: int = 3
    smoothing: float = 0.0


class IAPWSDensitySpline:
    """Spline wrapper around an IAPWS97 rho(p,h) table."""

    def __init__(
        self,
        spec: EOSGridSpec,
        p_grid: np.ndarray,
        h_grid: np.ndarray,
        rho_grid: np.ndarray,
    ):
        self.spec = spec
        self.p_grid = np.asarray(p_grid, dtype=np.float64)
        self.h_grid = np.asarray(h_grid, dtype=np.float64)
        self.rho_grid = np.asarray(rho_grid, dtype=np.float64)

        expected = (self.p_grid.size, self.h_grid.size)
        if self.rho_grid.shape != expected:
            raise ValueError(f"rho_grid shape must be {expected}, got {self.rho_grid.shape}")

        self._spline = RectBivariateSpline(
            self.p_grid,
            self.h_grid,
            self.rho_grid,
            kx=spec.kx,
            ky=spec.ky,
            s=spec.smoothing,
        )

    @classmethod
    def build(cls, spec: Optional[EOSGridSpec] = None, verbose: bool = True) -> "IAPWSDensitySpline":
        """Build rho(p,h) grid using IAPWS97."""
        spec = spec or EOSGridSpec()
        p_grid = np.linspace(spec.p_min, spec.p_max, spec.n_p, dtype=np.float64)
        h_grid = np.linspace(spec.h_min, spec.h_max, spec.n_h, dtype=np.float64)
        rho_grid = np.empty((spec.n_p, spec.n_h), dtype=np.float64)

        for i, p in enumerate(p_grid):
            if verbose and (i == 0 or (i + 1) % max(1, spec.n_p // 8) == 0 or i == spec.n_p - 1):
                print(f"IAPWS table: pressure row {i + 1}/{spec.n_p}")
            for j, h in enumerate(h_grid):
                rho_grid[i, j] = IAPWS97(P=float(p) / 1e6, h=float(h) / 1e3).rho

        return cls(spec=spec, p_grid=p_grid, h_grid=h_grid, rho_grid=rho_grid)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IAPWSDensitySpline":
        """Load table from .npz."""
        data = np.load(Path(path), allow_pickle=False)
        spec = EOSGridSpec(
            p_min=float(data["p_grid"][0]),
            p_max=float(data["p_grid"][-1]),
            h_min=float(data["h_grid"][0]),
            h_max=float(data["h_grid"][-1]),
            n_p=int(data["p_grid"].size),
            n_h=int(data["h_grid"].size),
            kx=int(data["kx"]),
            ky=int(data["ky"]),
            smoothing=float(data["smoothing"]),
        )
        return cls(spec=spec, p_grid=data["p_grid"], h_grid=data["h_grid"], rho_grid=data["rho_grid"])

    def save(self, path: Union[str, Path]) -> None:
        """Save table to .npz."""
        np.savez_compressed(
            Path(path),
            p_grid=self.p_grid,
            h_grid=self.h_grid,
            rho_grid=self.rho_grid,
            kx=np.array(self.spec.kx),
            ky=np.array(self.spec.ky),
            smoothing=np.array(self.spec.smoothing),
        )

    def _prepare_inputs(self, p: ArrayLike, h: ArrayLike, clip: bool) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        p_arr, h_arr = np.broadcast_arrays(np.asarray(p, dtype=np.float64), np.asarray(h, dtype=np.float64))
        shape = p_arr.shape
        p_flat = p_arr.reshape(-1)
        h_flat = h_arr.reshape(-1)

        if clip:
            p_flat = np.clip(p_flat, self.p_grid[0], self.p_grid[-1])
            h_flat = np.clip(h_flat, self.h_grid[0], self.h_grid[-1])
        else:
            if np.any((p_flat < self.p_grid[0]) | (p_flat > self.p_grid[-1])):
                raise ValueError("p is outside EOS table range")
            if np.any((h_flat < self.h_grid[0]) | (h_flat > self.h_grid[-1])):
                raise ValueError("h is outside EOS table range")

        return p_flat, h_flat, shape

    def eval(self, p: ArrayLike, h: ArrayLike, clip: bool = True) -> Dict[str, np.ndarray]:
        """Return rho, drho_dp, drho_dh as numpy arrays."""
        p_flat, h_flat, shape = self._prepare_inputs(p, h, clip=clip)

        return {
            "rho": self._spline.ev(p_flat, h_flat, dx=0, dy=0).reshape(shape),
            "rho_p": self._spline.ev(p_flat, h_flat, dx=1, dy=0).reshape(shape),
            "rho_h": self._spline.ev(p_flat, h_flat, dx=0, dy=1).reshape(shape),
        }

    def eval_torch(self, p, h, clip: bool = True) -> Dict[str, "torch.Tensor"]:
        """
        Return rho, drho_dp, drho_dh as torch tensors.

        This intentionally does not rely on autograd through the spline.
        Use returned partial derivatives with chain rule in PINN residuals.
        """
        import torch

        out_np = self.eval(p.detach().cpu().numpy(), h.detach().cpu().numpy(), clip=clip)
        return {k: torch.as_tensor(v, device=p.device, dtype=p.dtype) for k, v in out_np.items()}

    def eval_star_torch(
        self,
        pi_star,
        h_star,
        *,
        p_out: float,
        rho_pc: float,
        w0: float,
        h_pc: float,
        Cp_pc: float,
        beta_pc: float,
        clip: bool = True,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Evaluate dimensionless density and chain-rule coefficients.

        Returns:
            rho_star = rho / rho_pc
            chi_pi   = d rho* / d pi* = w0**2 * drho_dp
            chi_h    = d rho* / d h* = Cp_pc/(beta_pc*rho_pc) * drho_dh
        """
        p_abs = p_out + rho_pc * w0**2 * pi_star
        h_abs = h_pc + (Cp_pc / beta_pc) * h_star
        raw = self.eval_torch(p_abs, h_abs, clip=clip)

        return {
            "rho_star": raw["rho"] / rho_pc,
            "rho_p": raw["rho_p"],
            "rho_h": raw["rho_h"],
            "chi_pi": (w0**2) * raw["rho_p"],
            "chi_h": (Cp_pc / (beta_pc * rho_pc)) * raw["rho_h"],
            "p_abs": p_abs,
            "h_abs": h_abs,
        }

    def check_derivatives(
        self,
        p: float,
        h: float,
        dp: float = 1.0e3,
        dh: float = 1.0e2,
    ) -> Dict[str, float]:
        """Compare spline derivatives against finite differences of direct IAPWS97."""
        s = self.eval(p, h, clip=False)

        def rho_iapws(pp: float, hh: float) -> float:
            return IAPWS97(P=pp / 1e6, h=hh / 1e3).rho

        rho_p_fd = (rho_iapws(p + dp, h) - rho_iapws(p - dp, h)) / (2.0 * dp)
        rho_h_fd = (rho_iapws(p, h + dh) - rho_iapws(p, h - dh)) / (2.0 * dh)

        rho_p = float(np.asarray(s["rho_p"]))
        rho_h = float(np.asarray(s["rho_h"]))

        return {
            "rho_spline": float(np.asarray(s["rho"])),
            "rho_iapws": rho_iapws(p, h),
            "rho_p_spline": rho_p,
            "rho_p_fd_iapws": rho_p_fd,
            "rho_h_spline": rho_h,
            "rho_h_fd_iapws": rho_h_fd,
            "rel_err_rho_p": abs(rho_p - rho_p_fd) / (abs(rho_p_fd) + 1e-30),
            "rel_err_rho_h": abs(rho_h - rho_h_fd) / (abs(rho_h_fd) + 1e-30),
        }
