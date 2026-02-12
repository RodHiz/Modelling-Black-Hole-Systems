# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 21:52:30 2026

Kerr-Newman black hole geodesic integration using first-order separated equations
(Boyer-Lindquist coordinates, natural units G=M=c=1)

Supports a charged, rotating black hole with charge Q_BH and spin a.
The test particle has charge q (can be +1, -1, or 0) and mass mu=1.

State vector: [t, r, theta, phi, p_r, p_theta]
where p_r = Sigma * dr/dtau, p_theta = Sigma * dtheta/dtau

Conserved quantities (canonical):
  E  = -p_t  (generalized energy, includes EM coupling)
  Lz = p_phi (generalized angular momentum, includes EM coupling)
  Q  = Carter constant (modified for charge)

@author: rodri
"""

import numpy as np
import matplotlib.pyplot as plt

def rkf45(f, state, dt):
    """Runge-Kutta-Fehlberg 4(5) adaptive step integrator."""
    dt_min = 1e-12
    dt_max = 0.7
    tol = 1e-9

    for attempt in range(12):
        k1 = f(state)
        k2 = f(state + dt/4 * k1)
        k3 = f(state + dt * (3/32 * k1 + 9/32 * k2))
        k4 = f(state + dt * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = f(state + dt * (439/216 * k1 - 8*k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = f(state + dt * (-8/27 * k1 + 2*k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))

        coeff = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0,
                 16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]

        rk4 = state + dt * (coeff[0]*k1 + coeff[1]*k2 + coeff[2]*k3 + coeff[3]*k4 + coeff[4]*k5 + coeff[5]*k6)
        rk5 = state + dt * (coeff[6]*k1 + coeff[7]*k2 + coeff[8]*k3 + coeff[9]*k4 + coeff[10]*k5 + coeff[11]*k6)

        error = np.max(np.abs(rk5 - rk4))
        if error > 0:
            safety = 0.84 * (tol / error)**0.25
        else:
            safety = 2.0

        dt_new = np.clip(dt * safety, dt_min, dt_max)

        if (error <= tol) or (dt <= dt_min):
            return rk5, dt_new, True
        else:
            dt = dt_new

    return rk5, dt_min, True


# ==========================================================================
# Kerr-Newman metric and EM helper functions
# ==========================================================================

def Sigma_func(r, theta, a):
    return r**2 + a**2 * np.cos(theta)**2

def Delta_func(r, a, Q_BH):
    """Delta for Kerr-Newman: includes black hole charge Q_BH."""
    return r**2 - 2*r + a**2 + Q_BH**2

def EM_potential(r, theta, a, Q_BH):
    """
    Electromagnetic 4-potential A_mu for Kerr-Newman in Boyer-Lindquist coords.
    A_t = -Q_BH * r / Sigma
    A_phi = Q_BH * r * a * sin^2(theta) / Sigma
    Returns (A_t, A_phi).
    """
    sig = Sigma_func(r, theta, a)
    A_t = -Q_BH * r / sig
    A_phi = Q_BH * r * a * np.sin(theta)**2 / sig
    return A_t, A_phi

def compute_tdot_phidot(r, theta, a, Q_BH, q, E, Lz):
    """
    From conserved E and Lz (canonical), compute dt/dtau and dphi/dtau.
    
    For Kerr-Newman with charged particle:
      E_canonical = E  (includes q*A_t contribution)
      The "kinematic" part is P = E*(r^2+a^2) - a*Lz - q*Q_BH*r
      (this reduces to the uncharged Kerr expression when q=0 or Q_BH=0)
    """
    sig = Sigma_func(r, theta, a)
    delta = Delta_func(r, a, Q_BH)
    s2 = np.sin(theta)**2
    
    # P(r) for Kerr-Newman with charged particle
    P = E * (r**2 + a**2) - a * Lz - q * Q_BH * r
    
    tdot = ((r**2 + a**2) * P / delta + a * (Lz - a * E * s2)) / sig
    phidot = (a * P / delta - (a * E - Lz / s2)) / sig
    
    return tdot, phidot

def R_potential(r, a, Q_BH, q, E, Lz, Q_carter):
    """
    Radial potential R(r) for Kerr-Newman. p_r^2 = R(r).
    
    R(r) = P(r)^2 - Delta * [mu^2 * r^2 + (Lz - a*E)^2 + Q_carter]
    where P(r) = E*(r^2+a^2) - a*Lz - q*Q_BH*r
    and mu = 1 for timelike geodesics.
    """
    delta = Delta_func(r, a, Q_BH)
    P = E * (r**2 + a**2) - a * Lz - q * Q_BH * r
    K = (Lz - a*E)**2 + Q_carter
    return P**2 - delta * (r**2 + K)

def Theta_potential(theta, a, E, Lz, Q_carter):
    """
    Polar potential Theta(theta). p_theta^2 = Theta(theta).
    (Same form as Kerr — the charge does not enter the theta equation.)
    """
    c2 = np.cos(theta)**2
    s2 = np.sin(theta)**2
    return Q_carter + c2 * (a**2 * (E**2 - 1) - Lz**2 / s2)

def dR_dr(r, a, Q_BH, q, E, Lz, Q_carter):
    """Derivative of R(r) with respect to r."""
    delta = Delta_func(r, a, Q_BH)
    P = E * (r**2 + a**2) - a * Lz - q * Q_BH * r
    K = (Lz - a*E)**2 + Q_carter
    dP_dr = 2 * r * E - q * Q_BH
    dDelta_dr = 2 * r - 2
    return 2 * P * dP_dr - dDelta_dr * (r**2 + K) - delta * 2 * r

def dTheta_dtheta(theta, a, E, Lz, Q_carter):
    """Derivative of Theta(theta) with respect to theta."""
    s = np.sin(theta)
    c = np.cos(theta)
    return -2*s*c * a**2 * (E**2 - 1) + 2 * Lz**2 * c / (s**3)


# ==========================================================================
# Set up the problem
# ==========================================================================

plt.close('all')

# --- Black hole parameters ---
a    = 0.0      # Spin parameter (0 = Schwarzschild, <1 for sub-extremal)
Q_BH = 0.3       # Black hole electric charge (set 0 for uncharged Kerr)

# --- Particle charge ---
q = 0            # Particle charge: +1, -1, or 0 (neutral)

# Verify sub-extremal condition: a^2 + Q_BH^2 < M^2 = 1
if a**2 + Q_BH**2 >= 1.0:
    raise ValueError(f"Super-extremal black hole! a²+Q²={a**2+Q_BH**2:.4f} >= 1. "
                     f"Reduce a or Q_BH.")

# Event horizons for Kerr-Newman
r_plus  = 1 + np.sqrt(1 - a**2 - Q_BH**2)
r_minus = 1 - np.sqrt(1 - a**2 - Q_BH**2)
print(f"=== Kerr-Newman Black Hole ===")
print(f"Spin a = {a},  Charge Q_BH = {Q_BH},  Particle charge q = {q}")
print(f"a² + Q² = {a**2 + Q_BH**2:.4f}  (must be < 1)")
print(f"Outer horizon r+ = {r_plus:.4f}")
print(f"Inner horizon r- = {r_minus:.4f}")

# --- Initial conditions ---
r0 = 20.0
theta0 = np.pi / 2
phi0 = 0.0
rdot0 = -0.00001
thetadot0 = 0.0

# Circular orbit values for Kerr-Newman (approximate — use Kerr values as seed,
# then correct the metric norm and conserved quantities below)
# For Q_BH=0 this reduces to exact Kerr circular orbit values.
denom_circ = np.sqrt(1 - 3.0/r0 + 2*a/r0**1.5)
E_circ = (1 - 2.0/r0 + a/r0**1.5) / denom_circ
L_circ = np.sqrt(r0) * (1 - 2*a/r0**1.5 + a**2/r0**2) / denom_circ

# For charged particles, add the EM energy contribution as a correction
A_t0, A_phi0 = EM_potential(r0, theta0, a, Q_BH)
E_circ_KN = E_circ + q * A_t0   # approximate canonical energy correction
L_circ_KN = L_circ - q * A_phi0  # approximate canonical Lz correction

print(f"\nApproximate circular orbit values at r={r0}:")
print(f"  E_circ (Kerr)         = {E_circ:.6f}")
print(f"  L_circ (Kerr)         = {L_circ:.6f}")
print(f"  E_circ_KN (corrected) = {E_circ_KN:.6f}")
print(f"  L_circ_KN (corrected) = {L_circ_KN:.6f}")

# Perturb for a non-circular orbit
E_init = E_circ_KN * 1.001
Lz_init = L_circ_KN * 1.01

# Compute phidot from E, Lz
tdot0, phidot0 = compute_tdot_phidot(r0, theta0, a, Q_BH, q, E_init, Lz_init)

# Now enforce metric norm = -1 by adjusting tdot0
sig0 = Sigma_func(r0, theta0, a)
delta0 = Delta_func(r0, a, Q_BH)
s2_0 = np.sin(theta0)**2

g_tt = -(1 - 2*r0/sig0 + Q_BH**2/sig0)
g_tphi = -(2*a*r0*s2_0/sig0 - a*Q_BH**2*s2_0/sig0)
# g_tphi for KN: -(2*r - Q_BH^2)*a*sin^2(theta) / Sigma  (but let's be precise)
# Actually: g_tphi = -a*sin^2(theta)*(2*r - Q_BH^2)/Sigma
g_tphi = -a * s2_0 * (2*r0 - Q_BH**2) / sig0

# g_phiphi = (r^2+a^2+a^2*sin^2(theta)*(2r-Q_BH^2)/Sigma) * sin^2(theta)
g_phiphi = (r0**2 + a**2 + a**2 * s2_0 * (2*r0 - Q_BH**2) / sig0) * s2_0

other = (sig0/delta0 * rdot0**2
         + sig0 * thetadot0**2
         + g_phiphi * phidot0**2)

A_quad = g_tt
B_quad = 2 * g_tphi * phidot0
C_quad = other + 1  # +1 because we need g_mu_nu u^mu u^nu = -1

disc = B_quad**2 - 4*A_quad*C_quad
if disc >= 0:
    tdot0 = (-B_quad + np.sqrt(disc)) / (2 * A_quad)
    # Recompute conserved quantities with corrected tdot
    # For Kerr-Newman, the conserved canonical energy and angular momentum are:
    #   E = -g_tt*tdot - g_tphi*phidot - q*A_t
    #   Lz = g_tphi*tdot + g_phiphi*phidot + q*A_phi
    # (note signs: E = -p_t, Lz = p_phi, with p_mu = g_mu_nu u^nu + q A_mu)
    A_t0, A_phi0 = EM_potential(r0, theta0, a, Q_BH)
    E_init  = -g_tt * tdot0 - g_tphi * phidot0 - q * A_t0
    Lz_init = g_tphi * tdot0 + g_phiphi * phidot0 + q * A_phi0

    # Carter constant (for Kerr-Newman):
    # Q = p_theta^2 + cos^2(theta)*[a^2(mu^2 - E_kin^2) + Lz^2/sin^2(theta)]
    # where E_kin is the "kinematic" energy = E + q*A_t (but actually Q is built
    # from the canonical quantities). The standard form is:
    # Q = p_theta^2 + cos^2(theta)*[-a^2*(E^2-1) + Lz^2/sin^2(theta)]
    # This is the same form as Kerr because the theta equation is unchanged.
    p_th0 = sig0 * thetadot0
    Q_init = p_th0**2 + np.cos(theta0)**2 * (-a**2*(E_init**2 - 1) + Lz_init**2/s2_0)
else:
    raise ValueError("No valid tdot0 found — check initial conditions")

# Verify metric norm
g_norm_check = (g_tt*tdot0**2 + 2*g_tphi*phidot0*tdot0 + sig0/delta0*rdot0**2
                + sig0*thetadot0**2 + g_phiphi*phidot0**2)
print(f"\nInitial conditions:")
print(f"  E  = {E_init:.8f}")
print(f"  Lz = {Lz_init:.8f}")
print(f"  Q  = {Q_init:.8f}")
print(f"  q  = {q}  (particle charge)")
print(f"  tdot0 = {tdot0:.8f}")
print(f"  Metric norm = {g_norm_check:.12f}  (should be -1)")

# Verify potentials match initial velocities
p_r0 = sig0 * rdot0
p_th0 = sig0 * thetadot0
R_check = R_potential(r0, a, Q_BH, q, E_init, Lz_init, Q_init)
Theta_check = Theta_potential(theta0, a, E_init, Lz_init, Q_init)
print(f"\n  p_r0^2     = {p_r0**2:.8f},  R(r0)     = {R_check:.8f}")
print(f"  p_th0^2    = {p_th0**2:.8f},  Theta(th0)= {Theta_check:.8f}")


# ==========================================================================
# Equations of motion
# ==========================================================================

def f_kerr_newman(state):
    """
    Equations of motion for Kerr-Newman geodesics with charged particle.
    
    State: [t, r, theta, phi, p_r, p_theta]
    where p_r = Sigma * dr/dtau, p_theta = Sigma * dtheta/dtau.
    """
    r     = state[1]
    theta = state[2]
    p_r   = state[4]
    p_th  = state[5]
    
    sig = Sigma_func(r, theta, a)
    
    # Coordinate velocities from conserved quantities
    tdot, phidot = compute_tdot_phidot(r, theta, a, Q_BH, q, E_init, Lz_init)
    rdot   = p_r / sig
    thdot  = p_th / sig
    
    # Momentum equations
    dp_r_dtau  = dR_dr(r, a, Q_BH, q, E_init, Lz_init, Q_init) / (2.0 * sig)
    dp_th_dtau = dTheta_dtheta(theta, a, E_init, Lz_init, Q_init) / (2.0 * sig)
    
    return np.array([tdot, rdot, thdot, phidot, dp_r_dtau, dp_th_dtau])


# ==========================================================================
# Integration
# ==========================================================================

state0 = np.array([0.0, r0, theta0, phi0, p_r0, p_th0])

dt = 0.1
tf = 50000
tf_split = 0.1

if tf < 10000:
    nstep = 100101
else:
    nstep = int(tf / dt + 101)

path = np.zeros((nstep, 6))
path[0, :] = state0
tau_arr = np.zeros(nstep)
tau_arr[0] = 0

j = 0
dt_now = dt
run_sim = True

print(f"\nArray of size {nstep} allocated")

while run_sim:
    oldstate = path[j, :]
    
    newstate, dt_new, accepted = rkf45(f_kerr_newman, oldstate, dt_now)
    
    tau_arr[j + 1] = tau_arr[j] + dt_now
    path[j + 1, :] = newstate
    dt_now = dt_new
    j += 1
    
    if newstate[1] < r_plus + 0.05:
        jfinal = j + 1
        print("End of simulation")
        print(f"Object crossed event horizon at r = {newstate[1]:.4f} (r+ = {r_plus:.4f})")
        run_sim = False
    elif tau_arr[j] >= tf * tf_split:
        print(f"  proper time: ({(tau_arr[j] / tf) * 100:.0f}%), step {j}, dt={dt_now:.6f}")
        tf_split += 0.1
    if tau_arr[j] >= tf or j >= nstep - 2:
        jfinal = j + 1
        run_sim = False
        print("Simulation complete")


# ==========================================================================
# Extract results
# ==========================================================================

tpath     = path[:jfinal, 0]
rpath     = path[:jfinal, 1]
thetapath = path[:jfinal, 2]
phipath   = path[:jfinal, 3]
p_r_path  = path[:jfinal, 4]
p_th_path = path[:jfinal, 5]
tau       = tau_arr[:jfinal]

# Reconstruct velocities
sigpath   = Sigma_func(rpath, thetapath, a)
deltapath = Delta_func(rpath, a, Q_BH)
s2path    = np.sin(thetapath)**2
c2path    = np.cos(thetapath)**2

rdotpath     = p_r_path / sigpath
thetadotpath = p_th_path / sigpath
tdotpath     = np.zeros_like(rpath)
phidotpath   = np.zeros_like(rpath)
for i in range(len(rpath)):
    tdotpath[i], phidotpath[i] = compute_tdot_phidot(rpath[i], thetapath[i], a, Q_BH, q, E_init, Lz_init)

# Convert to Cartesian (Boyer-Lindquist -> oblate spheroidal -> Cartesian)
xpath = np.sqrt(rpath**2 + a**2) * np.sin(thetapath) * np.cos(phipath)
ypath = np.sqrt(rpath**2 + a**2) * np.sin(thetapath) * np.sin(phipath)
zpath = rpath * np.cos(thetapath)


# ==========================================================================
# Compute conserved quantities for verification
# ==========================================================================

# Kerr-Newman metric components along trajectory
g_tt_path    = -(1 - (2*rpath - Q_BH**2)/sigpath)
g_tphi_path  = -a * s2path * (2*rpath - Q_BH**2) / sigpath
g_rr_path    = sigpath / deltapath
g_thth_path  = sigpath
g_phiphi_path = (rpath**2 + a**2 + a**2 * s2path * (2*rpath - Q_BH**2) / sigpath) * s2path

# EM potential along trajectory
A_t_path   = -Q_BH * rpath / sigpath
A_phi_path = Q_BH * rpath * a * s2path / sigpath

# Energy: E = -(g_tt*tdot + g_tphi*phidot) - q*A_t
E_check = -(g_tt_path * tdotpath + g_tphi_path * phidotpath) - q * A_t_path

# Angular momentum: Lz = g_tphi*tdot + g_phiphi*phidot + q*A_phi
Lz_check = g_tphi_path * tdotpath + g_phiphi_path * phidotpath + q * A_phi_path

# Carter constant Q = p_theta^2 + cos^2(theta)*[a^2(1-E^2) + Lz^2/sin^2(theta)]
# Use E_check and Lz_check for self-consistent diagnostic
Q_check = p_th_path**2 + c2path * (a**2 * (1 - E_check**2) + Lz_check**2 / s2path)

# Metric norm (should be -1)
metric_norm = (g_tt_path * tdotpath**2
               + 2 * g_tphi_path * tdotpath * phidotpath
               + g_rr_path * rdotpath**2
               + g_thth_path * thetadotpath**2
               + g_phiphi_path * phidotpath**2)

# Also check via potentials
R_check_arr = np.array([R_potential(rpath[i], a, Q_BH, q, E_init, Lz_init, Q_init) for i in range(len(rpath))])
Th_check_arr = np.array([Theta_potential(thetapath[i], a, E_init, Lz_init, Q_init) for i in range(len(thetapath))])

# ==========================================================================
# Plotting
# ==========================================================================

# --- r, theta, phi vs proper time ---
plt.figure('time / radius')
plt.plot(tau, rpath)
plt.axhline(r_plus, color='r', linestyle='--', label=f'r+ = {r_plus:.2f}')
plt.xlabel('Proper time τ')
plt.ylabel('r')
plt.title('Radial coordinate vs proper time')
plt.grid(True)
plt.legend()

plt.figure('time / theta')
plt.plot(tau, thetapath)
plt.xlabel('Proper time τ')
plt.ylabel('θ')
plt.title('Polar angle vs proper time')
plt.grid(True)

plt.figure('time / phi')
plt.plot(tau, phipath)
plt.xlabel('Proper time τ')
plt.ylabel('φ')
plt.title('Azimuthal angle vs proper time')
plt.grid(True)

# --- Conserved quantities: Raw values (tight ylims) ---
fig_raw, axes_raw = plt.subplots(4, 1, figsize=(10, 12))
fig_raw.suptitle('Conserved Quantities — Raw Values', fontweight='bold', fontsize=14)

pad = 0.05  # 5% padding on ylims

def set_tight_ylim(ax, data, pad_frac=0.1):
    """Set ylim tightly around data: [min - buff, max + buff] where buff = pad_frac * range."""
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    drange = dmax - dmin
    if drange == 0:
        # Data is constant — use a small absolute buffer around the value
        buff = max(abs(dmin) * 1e-6, 1e-15)
    else:
        buff = pad_frac * drange
    ax.set_ylim(dmin - buff, dmax + buff)

axes_raw[0].plot(tau, E_check, linewidth=0.8)
axes_raw[0].axhline(E_init, color='r', linestyle='--', alpha=0.5, label=f'E₀ = {E_init:.6f}')
axes_raw[0].set_xlabel('Proper time τ')
axes_raw[0].set_ylabel('E')
axes_raw[0].set_title('Energy (canonical)')
set_tight_ylim(axes_raw[0], E_check)
axes_raw[0].grid(True)
axes_raw[0].legend()

axes_raw[1].plot(tau, Lz_check, linewidth=0.8)
axes_raw[1].axhline(Lz_init, color='r', linestyle='--', alpha=0.5, label=f'Lz₀ = {Lz_init:.6f}')
axes_raw[1].set_xlabel('Proper time τ')
axes_raw[1].set_ylabel('Lz')
axes_raw[1].set_title('Angular momentum (z) (canonical)')
set_tight_ylim(axes_raw[1], Lz_check)
axes_raw[1].grid(True)
axes_raw[1].legend()

axes_raw[2].plot(tau, Q_check, linewidth=0.8)
axes_raw[2].axhline(Q_init, color='r', linestyle='--', alpha=0.5, label=f'Q₀ = {Q_init:.6f}')
axes_raw[2].set_xlabel('Proper time τ')
axes_raw[2].set_ylabel('Q')
axes_raw[2].set_title('Carter constant')
set_tight_ylim(axes_raw[2], Q_check)
axes_raw[2].grid(True)
axes_raw[2].legend()

axes_raw[3].plot(tau, metric_norm, linewidth=0.8)
axes_raw[3].axhline(-1.0, color='r', linestyle='--', alpha=0.5, label='Expected: -1')
axes_raw[3].set_xlabel('Proper time τ')
axes_raw[3].set_ylabel('g_μν ẋᵘẋᵛ')
axes_raw[3].set_title('Metric norm (should be -1)')
axes_raw[3].grid(True)
axes_raw[3].legend()

fig_raw.tight_layout()

# --- Conserved quantities: Difference from initial values ---
fig_diff, axes_diff = plt.subplots(4, 1, figsize=(10, 12))
fig_diff.suptitle('Conserved Quantities — Deviation from Initial Value', fontweight='bold', fontsize=14)

dE = E_check - E_init
axes_diff[0].plot(tau, dE, linewidth=0.8)
axes_diff[0].axhline(0, color='r', linestyle='--', alpha=0.5)
axes_diff[0].set_xlabel('Proper time τ')
axes_diff[0].set_ylabel('E - E₀')
axes_diff[0].set_title('Energy drift')
set_tight_ylim(axes_diff[0], dE)
axes_diff[0].grid(True)

dLz = Lz_check - Lz_init
axes_diff[1].plot(tau, dLz, linewidth=0.8)
axes_diff[1].axhline(0, color='r', linestyle='--', alpha=0.5)
axes_diff[1].set_xlabel('Proper time τ')
axes_diff[1].set_ylabel('Lz - Lz₀')
axes_diff[1].set_title('Angular momentum drift')
set_tight_ylim(axes_diff[1], dLz)
axes_diff[1].grid(True)

dQ = Q_check - Q_init
axes_diff[2].plot(tau, dQ, linewidth=0.8)
axes_diff[2].axhline(0, color='r', linestyle='--', alpha=0.5)
axes_diff[2].set_xlabel('Proper time τ')
axes_diff[2].set_ylabel('Q - Q₀')
axes_diff[2].set_title('Carter constant drift')
set_tight_ylim(axes_diff[2], dQ)
axes_diff[2].grid(True)

dNorm = metric_norm - (-1.0)
axes_diff[3].plot(tau, dNorm, linewidth=0.8)
axes_diff[3].axhline(0, color='r', linestyle='--', alpha=0.5)
axes_diff[3].set_xlabel('Proper time τ')
axes_diff[3].set_ylabel('norm - (-1)')
axes_diff[3].set_title('Metric norm deviation')
set_tight_ylim(axes_diff[3], dNorm)
axes_diff[3].grid(True)

fig_diff.tight_layout()

# --- Potential consistency: 2 subplots ---
fig_pot, (ax_r, ax_th) = plt.subplots(2, 1, figsize=(10, 8))
fig_pot.suptitle('Potential Consistency Check', fontweight='bold', fontsize=14)

ax_r.plot(tau, p_r_path**2, label='p_r²', linewidth=0.8)
ax_r.plot(tau, R_check_arr, '--', label='R(r)', linewidth=0.8)
ax_r.set_xlabel('Proper time τ')
ax_r.set_ylabel('Value')
ax_r.set_title('Radial: p_r² vs R(r)  [should overlap]')
ax_r.legend()
ax_r.grid(True)

ax_th.plot(tau, p_th_path**2, label='p_θ²', linewidth=0.8)
ax_th.plot(tau, Th_check_arr, '--', label='Θ(θ)', linewidth=0.8)
ax_th.set_xlabel('Proper time τ')
ax_th.set_ylabel('Value')
ax_th.set_title('Polar: p_θ² vs Θ(θ)  [should overlap]')
ax_th.legend()
ax_th.grid(True)

fig_pot.tight_layout()

# --- Coordinate time vs proper time ---
plt.figure('Coordinate Time vs Proper Time')
plt.plot(tau, tpath, 'b-', linewidth=1)
plt.xlabel('Proper Time (τ)')
plt.ylabel('Coordinate Time (t)')
plt.title('Coordinate Time vs Proper Time')
plt.grid(True)

# --- 2D projections ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
angle = np.linspace(0, 2 * np.pi, 500)

ax1.set_title('x vs y (top view)')
ax1.plot(xpath, ypath, 'b-', linewidth=0.5, label='Trajectory')
ax1.plot(xpath[-1], ypath[-1], 'or', label='Final position')
rho_plus = np.sqrt(r_plus**2 + a**2)
ax1.plot(rho_plus * np.cos(angle), rho_plus * np.sin(angle), 'k', label=f'Outer horizon (r+={r_plus:.2f})')
ax1.plot(0, 0, 'ok', label='Singularity')
ax1.grid(True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.axis('equal')

ax2.set_title('x vs z (side view)')
ax2.plot(xpath, zpath, 'b-', linewidth=0.5)
ax2.plot(xpath[-1], zpath[-1], 'or')
ax2.plot(rho_plus * np.cos(angle), r_plus * np.sin(angle), 'k')
ax2.plot(0, 0, 'ok')
ax2.grid(True)
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.axis('equal')

ax3.set_title('y vs z')
ax3.plot(ypath, zpath, 'b-', linewidth=0.5)
ax3.plot(ypath[-1], zpath[-1], 'or')
ax3.plot(rho_plus * np.sin(angle), r_plus * np.cos(angle), 'k')
ax3.plot(0, 0, 'ok')
ax3.grid(True)
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.axis('equal')

ax4.axis('off')
info_text = (f"Kerr-Newman BH\n"
             f"a = {a}, Q_BH = {Q_BH}\n"
             f"Particle charge q = {q}\n"
             f"r+ = {r_plus:.4f}")
ax4.text(0.5, 0.7, info_text, transform=ax4.transAxes,
         fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
handles, labels = ax1.get_legend_handles_labels()
ax4.legend(handles, labels, loc='center', fontsize=12)

plt.tight_layout()

# ============ 3D PLOT ======================================================
fig = plt.figure('3D Kerr-Newman Geodesic', figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(xpath, ypath, zpath, 'b-', linewidth=0.5, label='Geodesic')
ax.scatter(xpath[-1], ypath[-1], zpath[-1], color='red', s=50, label='Final Position')

ring_angle = np.linspace(0, 2 * np.pi, 200)
ring_x = a * np.cos(ring_angle)
ring_y = a * np.sin(ring_angle)
ring_z = np.zeros_like(ring_angle)
ax.plot(ring_x, ring_y, ring_z, 'k-', linewidth=3, label=f'Ring singularity (a={a})')

u_h = np.linspace(0, 2 * np.pi, 80)
v_h = np.linspace(0, np.pi, 40)
U_h, V_h = np.meshgrid(u_h, v_h)
rho_h = np.sqrt(r_plus**2 + a**2)
x_h = rho_h * np.sin(V_h) * np.cos(U_h)
y_h = rho_h * np.sin(V_h) * np.sin(U_h)
z_h = r_plus * np.cos(V_h)
ax.plot_wireframe(x_h, y_h, z_h, color='black', alpha=0.3,
                  linewidth=0.8, rstride=3, cstride=3, label=f'Outer horizon (r+={r_plus:.2f})')

theta_ergo = np.linspace(0, np.pi, 40)
phi_ergo = np.linspace(0, 2 * np.pi, 80)
PHI_e, THETA_e = np.meshgrid(phi_ergo, theta_ergo)
r_ergo = 1 + np.sqrt(1 - a**2 * np.cos(THETA_e)**2 - Q_BH**2)
# Protect against negative sqrt argument at poles if Q_BH is large
r_ergo = np.where(1 - a**2 * np.cos(THETA_e)**2 - Q_BH**2 >= 0,
                  1 + np.sqrt(np.maximum(0, 1 - a**2 * np.cos(THETA_e)**2 - Q_BH**2)),
                  np.nan)
rho_ergo = np.sqrt(r_ergo**2 + a**2)
x_ergo = rho_ergo * np.sin(THETA_e) * np.cos(PHI_e)
y_ergo = rho_ergo * np.sin(THETA_e) * np.sin(PHI_e)
z_ergo = r_ergo * np.cos(THETA_e)
ax.plot_wireframe(x_ergo, y_ergo, z_ergo, color='blue', alpha=0.15,
                  linewidth=0.5, rstride=3, cstride=3, label='Ergosphere')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D Geodesic: Kerr-Newman (a={a}, Q_BH={Q_BH}, q={q})', fontweight='bold')
ax.legend(loc='upper left', fontsize=8)

max_range = np.max([xpath.max() - xpath.min(),
                    ypath.max() - ypath.min(),
                    zpath.max() - zpath.min()]) / 2
mid_x = (xpath.max() + xpath.min()) / 2
mid_y = (ypath.max() + ypath.min()) / 2
mid_z = (zpath.max() + zpath.min()) / 2
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

print(f"\n--- Final Summary ---")
print(f"Black hole: a = {a}, Q_BH = {Q_BH}")
print(f"Particle charge: q = {q}")
print(f"Initial radius: r0 = {r0}")
print(f"E  = {E_init:.8f}  (drift: {abs(E_check[-1]-E_init):.2e})")
print(f"Lz = {Lz_init:.8f}  (drift: {abs(Lz_check[-1]-Lz_init):.2e})")
print(f"Q  = {Q_init:.8f}  (drift: {abs(Q_check[-1]-Q_init):.2e})")
print(f"Metric norm final = {metric_norm[-1]:.12f}  (should be -1)")
print(f"Metric norm max deviation = {np.max(np.abs(metric_norm + 1)):.2e}")
print(f"Total steps: {jfinal}")