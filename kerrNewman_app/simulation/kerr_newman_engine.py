# -*- coding: utf-8 -*-
"""
Kerr-Newman black hole geodesic engine — callable from Streamlit.

Refactored from kerrNewman_3D.py into pure functions that take parameters
and return results (no matplotlib, no global state).

Covers all sub-cases:
  Schwarzschild : a=0, Q_BH=0, q=0
  Kerr          : Q_BH=0, q=0
  Reissner-Nordström : a=0
  Kerr-Newman   : general
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── RKF45 integrator ─────────────────────────────────────────────────────

def rkf45(f, state, dt):
    """Runge-Kutta-Fehlberg 4(5) adaptive step integrator."""
    dt_min = 1e-12
    dt_max = 0.7
    tol = 1e-9

    for _ in range(12):
        k1 = f(state)
        k2 = f(state + dt / 4 * k1)
        k3 = f(state + dt * (3 / 32 * k1 + 9 / 32 * k2))
        k4 = f(state + dt * (1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3))
        k5 = f(state + dt * (439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4))
        k6 = f(state + dt * (-8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5))

        c4 = [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]
        c5 = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

        rk4 = state + dt * (c4[0] * k1 + c4[2] * k3 + c4[3] * k4 + c4[4] * k5)
        rk5 = state + dt * (c5[0] * k1 + c5[2] * k3 + c5[3] * k4 + c5[4] * k5 + c5[5] * k6)

        error = np.max(np.abs(rk5 - rk4))
        safety = 0.84 * (tol / error) ** 0.25 if error > 0 else 2.0
        dt_new = np.clip(dt * safety, dt_min, dt_max)

        if error <= tol or dt <= dt_min:
            return rk5, dt_new, True
        dt = dt_new

    return rk5, dt_min, True


# ── Metric helper functions ──────────────────────────────────────────────

def Sigma(r, theta, a):
    return r ** 2 + a ** 2 * np.cos(theta) ** 2


def Delta(r, a, Q_BH):
    return r ** 2 - 2 * r + a ** 2 + Q_BH ** 2


def EM_potential(r, theta, a, Q_BH):
    sig = Sigma(r, theta, a)
    A_t = -Q_BH * r / sig
    A_phi = Q_BH * r * a * np.sin(theta) ** 2 / sig
    return A_t, A_phi


def compute_tdot_phidot(r, theta, a, Q_BH, q, E, Lz):
    sig = Sigma(r, theta, a)
    delta = Delta(r, a, Q_BH)
    s2 = np.sin(theta) ** 2
    P = E * (r ** 2 + a ** 2) - a * Lz - q * Q_BH * r
    tdot = ((r ** 2 + a ** 2) * P / delta + a * (Lz - a * E * s2)) / sig
    phidot = (a * P / delta - (a * E - Lz / s2)) / sig
    return tdot, phidot


def R_potential(r, a, Q_BH, q, E, Lz, Q_carter):
    delta = Delta(r, a, Q_BH)
    P = E * (r ** 2 + a ** 2) - a * Lz - q * Q_BH * r
    K = (Lz - a * E) ** 2 + Q_carter
    return P ** 2 - delta * (r ** 2 + K)


def Theta_potential(theta, a, E, Lz, Q_carter):
    c2 = np.cos(theta) ** 2
    s2 = np.sin(theta) ** 2
    return Q_carter + c2 * (a ** 2 * (E ** 2 - 1) - Lz ** 2 / s2)


def dR_dr(r, a, Q_BH, q, E, Lz, Q_carter):
    delta = Delta(r, a, Q_BH)
    P = E * (r ** 2 + a ** 2) - a * Lz - q * Q_BH * r
    K = (Lz - a * E) ** 2 + Q_carter
    dP = 2 * r * E - q * Q_BH
    dDelta = 2 * r - 2
    return 2 * P * dP - dDelta * (r ** 2 + K) - delta * 2 * r


def dTheta_dtheta(theta, a, E, Lz, Q_carter):
    s = np.sin(theta)
    c = np.cos(theta)
    return -2 * s * c * a ** 2 * (E ** 2 - 1) + 2 * Lz ** 2 * c / s ** 3


# ── Parameter dataclass ─────────────────────────────────────────────────

@dataclass
class SimParams:
    """All tuneable parameters for a Kerr-Newman simulation."""
    # Black hole
    a: float = 0.0          # spin
    Q_BH: float = 0.0       # BH charge
    q: float = 0.0           # test-particle charge

    # Initial position
    r0: float = 20.0
    theta0: float = np.pi / 2
    phi0: float = 0.0

    # Initial velocity perturbation
    rdot0: float = -0.00001
    thetadot0: float = 0.0

    # Energy / angular-momentum perturbation factors
    E_factor: float = 1.001
    Lz_factor: float = 1.01

    # Integration
    dt: float = 0.1
    tf: float = 50000


# ── Result dataclass ────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Everything the GUI needs for plotting."""
    # Coordinates
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))

    # Velocities
    tdot: np.ndarray = field(default_factory=lambda: np.array([]))
    rdot: np.ndarray = field(default_factory=lambda: np.array([]))
    thetadot: np.ndarray = field(default_factory=lambda: np.array([]))
    phidot: np.ndarray = field(default_factory=lambda: np.array([]))

    # Conserved quantities along trajectory
    E_check: np.ndarray = field(default_factory=lambda: np.array([]))
    Lz_check: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_check: np.ndarray = field(default_factory=lambda: np.array([]))
    metric_norm: np.ndarray = field(default_factory=lambda: np.array([]))

    # Potential consistency
    R_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    Theta_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    p_r: np.ndarray = field(default_factory=lambda: np.array([]))
    p_th: np.ndarray = field(default_factory=lambda: np.array([]))

    # Initial conserved values
    E_init: float = 0.0
    Lz_init: float = 0.0
    Q_init: float = 0.0

    # BH geometry
    r_plus: float = 0.0
    r_minus: float = 0.0
    a: float = 0.0
    Q_BH: float = 0.0
    q: float = 0.0

    # Status
    termination: str = ""
    total_steps: int = 0
    log: str = ""


# ── Main simulation runner ──────────────────────────────────────────────

def run_simulation(p: SimParams, progress_callback=None) -> SimResult:
    """
    Run the Kerr-Newman geodesic integration and return a SimResult.

    Parameters
    ----------
    p : SimParams
    progress_callback : callable(fraction: float, msg: str) or None

    Returns
    -------
    SimResult
    """
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    a = p.a
    Q_BH = p.Q_BH
    q = p.q

    # ── Validate ──
    if a ** 2 + Q_BH ** 2 >= 1.0:
        raise ValueError(
            f"Super-extremal! a²+Q²={a ** 2 + Q_BH ** 2:.4f} ≥ 1. Reduce a or Q_BH."
        )

    r_plus = 1 + np.sqrt(1 - a ** 2 - Q_BH ** 2)
    r_minus = 1 - np.sqrt(1 - a ** 2 - Q_BH ** 2)
    log(f"r+ = {r_plus:.4f},  r- = {r_minus:.4f}")

    r0, theta0, phi0 = p.r0, p.theta0, p.phi0
    rdot0, thetadot0 = p.rdot0, p.thetadot0

    # ── Approximate circular-orbit seed ──
    denom = np.sqrt(1 - 3.0 / r0 + 2 * a / r0 ** 1.5)
    E_circ = (1 - 2.0 / r0 + a / r0 ** 1.5) / denom
    L_circ = np.sqrt(r0) * (1 - 2 * a / r0 ** 1.5 + a ** 2 / r0 ** 2) / denom

    A_t0, A_phi0 = EM_potential(r0, theta0, a, Q_BH)
    E_circ_KN = E_circ + q * A_t0
    L_circ_KN = L_circ - q * A_phi0

    E_init = E_circ_KN * p.E_factor
    Lz_init = L_circ_KN * p.Lz_factor

    tdot0, phidot0 = compute_tdot_phidot(r0, theta0, a, Q_BH, q, E_init, Lz_init)

    # ── Enforce metric norm = -1 ──
    sig0 = Sigma(r0, theta0, a)
    delta0 = Delta(r0, a, Q_BH)
    s2_0 = np.sin(theta0) ** 2

    g_tt = -(1 - (2 * r0 - Q_BH ** 2) / sig0)
    g_tphi = -a * s2_0 * (2 * r0 - Q_BH ** 2) / sig0
    g_phiphi = (r0 ** 2 + a ** 2 + a ** 2 * s2_0 * (2 * r0 - Q_BH ** 2) / sig0) * s2_0

    other = sig0 / delta0 * rdot0 ** 2 + sig0 * thetadot0 ** 2 + g_phiphi * phidot0 ** 2
    A_q = g_tt
    B_q = 2 * g_tphi * phidot0
    C_q = other + 1

    disc = B_q ** 2 - 4 * A_q * C_q
    if disc < 0:
        raise ValueError("No valid tdot0 — check initial conditions.")

    tdot0 = (-B_q + np.sqrt(disc)) / (2 * A_q)
    A_t0, A_phi0 = EM_potential(r0, theta0, a, Q_BH)
    E_init = -g_tt * tdot0 - g_tphi * phidot0 - q * A_t0
    Lz_init = g_tphi * tdot0 + g_phiphi * phidot0 + q * A_phi0

    p_th0 = sig0 * thetadot0
    Q_init = p_th0 ** 2 + np.cos(theta0) ** 2 * (-a ** 2 * (E_init ** 2 - 1) + Lz_init ** 2 / s2_0)

    p_r0 = sig0 * rdot0

    log(f"E  = {E_init:.8f}")
    log(f"Lz = {Lz_init:.8f}")
    log(f"Q  = {Q_init:.8f}")

    # ── Build EOM closure ──
    def f_eom(state):
        r = state[1]
        theta = state[2]
        p_r_loc = state[4]
        p_th_loc = state[5]
        sig = Sigma(r, theta, a)
        td, phid = compute_tdot_phidot(r, theta, a, Q_BH, q, E_init, Lz_init)
        rd = p_r_loc / sig
        thd = p_th_loc / sig
        dp_r = dR_dr(r, a, Q_BH, q, E_init, Lz_init, Q_init) / (2.0 * sig)
        dp_th = dTheta_dtheta(theta, a, E_init, Lz_init, Q_init) / (2.0 * sig)
        return np.array([td, rd, thd, phid, dp_r, dp_th])

    # ── Integrate ──
    tf = p.tf
    dt = p.dt
    nstep = max(100101, int(tf / dt + 101))

    path = np.zeros((nstep, 6))
    path[0] = [0.0, r0, theta0, phi0, p_r0, p_th0]
    tau_arr = np.zeros(nstep)
    dt_now = dt

    j = 0
    termination = "Simulation complete"
    next_report = 0.1

    while True:
        newstate, dt_new, _ = rkf45(f_eom, path[j], dt_now)
        j += 1
        tau_arr[j] = tau_arr[j - 1] + dt_now
        path[j] = newstate
        dt_now = dt_new

        frac = tau_arr[j] / tf
        if frac >= next_report:
            if progress_callback:
                progress_callback(min(frac, 1.0), f"τ = {tau_arr[j]:.0f} / {tf:.0f}")
            next_report += 0.1

        if newstate[1] < r_plus + 0.05:
            termination = f"Crossed horizon at r = {newstate[1]:.4f} (r+ = {r_plus:.4f})"
            break
        if tau_arr[j] >= tf or j >= nstep - 2:
            break

    jf = j + 1
    log(termination)

    # ── Extract arrays ──
    tau = tau_arr[:jf]
    tpath = path[:jf, 0]
    rpath = path[:jf, 1]
    thpath = path[:jf, 2]
    phpath = path[:jf, 3]
    p_r_path = path[:jf, 4]
    p_th_path = path[:jf, 5]

    sigp = Sigma(rpath, thpath, a)
    rdotp = p_r_path / sigp
    thdotp = p_th_path / sigp
    tdotp = np.empty_like(rpath)
    phdotp = np.empty_like(rpath)
    for i in range(len(rpath)):
        tdotp[i], phdotp[i] = compute_tdot_phidot(rpath[i], thpath[i], a, Q_BH, q, E_init, Lz_init)

    xp = np.sqrt(rpath ** 2 + a ** 2) * np.sin(thpath) * np.cos(phpath)
    yp = np.sqrt(rpath ** 2 + a ** 2) * np.sin(thpath) * np.sin(phpath)
    zp = rpath * np.cos(thpath)

    # ── Conserved quantities along trajectory ──
    s2p = np.sin(thpath) ** 2
    c2p = np.cos(thpath) ** 2
    deltap = Delta(rpath, a, Q_BH)
    g_tt_p = -(1 - (2 * rpath - Q_BH ** 2) / sigp)
    g_tphi_p = -a * s2p * (2 * rpath - Q_BH ** 2) / sigp
    g_rr_p = sigp / deltap
    g_phiphi_p = (rpath ** 2 + a ** 2 + a ** 2 * s2p * (2 * rpath - Q_BH ** 2) / sigp) * s2p

    A_t_p = -Q_BH * rpath / sigp
    A_phi_p = Q_BH * rpath * a * s2p / sigp

    E_chk = -(g_tt_p * tdotp + g_tphi_p * phdotp) - q * A_t_p
    Lz_chk = g_tphi_p * tdotp + g_phiphi_p * phdotp + q * A_phi_p
    Q_chk = p_th_path ** 2 + c2p * (a ** 2 * (1 - E_chk ** 2) + Lz_chk ** 2 / s2p)

    norm = (g_tt_p * tdotp ** 2 + 2 * g_tphi_p * tdotp * phdotp
            + g_rr_p * rdotp ** 2 + sigp * thdotp ** 2 + g_phiphi_p * phdotp ** 2)

    R_arr = np.array([R_potential(rpath[i], a, Q_BH, q, E_init, Lz_init, Q_init) for i in range(len(rpath))])
    Th_arr = np.array([Theta_potential(thpath[i], a, E_init, Lz_init, Q_init) for i in range(len(thpath))])

    if progress_callback:
        progress_callback(1.0, "Done")

    return SimResult(
        tau=tau, t=tpath, r=rpath, theta=thpath, phi=phpath,
        x=xp, y=yp, z=zp,
        tdot=tdotp, rdot=rdotp, thetadot=thdotp, phidot=phdotp,
        E_check=E_chk, Lz_check=Lz_chk, Q_check=Q_chk, metric_norm=norm,
        R_arr=R_arr, Theta_arr=Th_arr, p_r=p_r_path, p_th=p_th_path,
        E_init=E_init, Lz_init=Lz_init, Q_init=Q_init,
        r_plus=r_plus, r_minus=r_minus,
        a=a, Q_BH=Q_BH, q=q,
        termination=termination, total_steps=jf,
        log="\n".join(log_lines),
    )
