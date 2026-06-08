---
name: pinn-pressure-bc-physics
description: "Correct pressure/flow boundary conditions for the supercritical-stability PINN, per Ambrosini 2010 and Churkin TEMPA-SC docs"
metadata: 
  node_type: memory
  type: project
  originSessionId: 36c45d2a-0e96-4b6e-b0e4-a957e9eca976
---

Постановка граничных условий для PINN устойчивости сверхкритической воды (канал Churkin/Ambrosini, 25 МПа, NSPC=2, Kout=20). Источники: `1-s2.0-S0306454910003282-main.pdf` (Ambrosini 2010, *Annals of Nuclear Energy*) и `Churkin-Description.pdf` (TEMPA-SC, бенчмарк OKB Гидропресс).

**Как авторы задают давление и расход:**
- Δp через канал ФИКСИРОВАН (imposed pressure drop): Churkin — `Δpch = 0.12 МПа`, `pout = 25 МПа`, `pin = pout + Δpch`. Постоянен во времени.
- Расход СВОБОДЕН: Ambrosini — "the value of the pressure drop ... is imposed across the channel, **letting flow rate to freely oscillate**". G*_in=ρ*_in — только стационарное НАЧАЛЬНОЕ условие (t=0), НЕ BC на всё время.
- Неустойчивость возбуждается медленным ростом мощности Q(t): "Heat flux assigned as a function of time", rate ≈ 21.3 Вт/с у порога; в начале (с 2-й по 12-ю секунду) — power surge для ускорения. Это и есть `forcing.power` из xlsx и `NQ_prime(t)`.
- Inlet BC: pressure ИЛИ flowrate + enthalpy/temperature. Outlet BC: pressure + enthalpy/temperature.
- Физика колебаний: рост T на выходе → падение расхода → ещё больший рост T → density-wave осцилляции расхода.

**Что было неправильно в pinn_v3/v4:** код фиксировал π* на ОБОИХ концах (bc_pi_in=Δπ*, bc_pi_out=0) И навязывал G*_in=ρ*_in как BC на всё время → over-constrained, колебаниям негде взяться. Правильно: imposed Δp как одно интегральное ограничение (π*_in−π*_out=Δπ*=const), расход на входе свободен.

**Исправлено в `pinn_v3b_imposed_dp.ipynb`** (копия v3): в `Physics.bc_loss` (cell 12) теперь `R_dp = (pi_in − pi_out)/dpi_star − 1` (imposed Δp) + `R_pi_out = pi_out/dpi_star` (gauge); ключ `bc_pi_in` переименован в `bc_dp` в cell 12/15/22. Это активная линия работы (июнь 2026). Решение пользователя: воспроизводить колебания из физики (не data-loss), база — линия v3, на первом шаге менять ТОЛЬКО постановку. Если колебаний не будет — следующий шаг: causal-curriculum из v4 с мелкими стадиями + Fourier-фичи по времени t. Подробный handoff — в `HANDOFF.md` в корне репо.

NTPC порог для Kout=20, NSPC=1.5 ≈ 2.93 (Churkin Table 2).
