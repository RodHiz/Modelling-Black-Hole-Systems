# -*- coding: utf-8 -*-
"""
Black Hole Geodesic Explorer — Streamlit GUI
=============================================
Interactive simulation of test-particle orbits around:
  • Schwarzschild  (a = 0, Q = 0)
  • Kerr           (Q = 0)
  • Reissner-Nordström (a = 0)
  • Kerr-Newman    (general)

All powered by the same Kerr-Newman engine with parameters toggled to zero.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from simulation.kerr_newman_engine import SimParams, run_simulation

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Hole Geodesic Explorer",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.4rem 0;
        text-align: center;
    }
    .metric-card h3 {
        color: #e94560;
        margin: 0 0 0.3rem 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: #eee;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
        font-family: 'Courier New', monospace;
    }
    .sidebar-title {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e94560;
        margin-bottom: 1rem;
    }
    .bh-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .bh-schwarzschild { background: #1b998b; color: #fff; }
    .bh-kerr { background: #2d6a4f; color: #fff; }
    .bh-rn { background: #e76f51; color: #fff; }
    .bh-kn { background: #9b2226; color: #fff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }
    .example-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        border: 1px solid #415a77;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .example-card h4 { color: #00d4ff; margin: 0 0 0.3rem 0; }
    .example-card p { color: #ccc; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# BH type definitions
# ─────────────────────────────────────────────────────────────────────────
BH_TYPES = {
    "Schwarzschild": {"a": 0.0, "Q_BH": 0.0, "q": 0.0,
                      "desc": "Non-rotating, uncharged — the simplest GR black hole.",
                      "badge": "bh-schwarzschild"},
    "Kerr": {"a": 0.6, "Q_BH": 0.0, "q": 0.0,
             "desc": "Rotating, uncharged — models most astrophysical BHs.",
             "badge": "bh-kerr"},
    "Reissner-Nordström": {"a": 0.0, "Q_BH": 0.3, "q": 0.0,
                           "desc": "Non-rotating, charged — theoretical interest.",
                           "badge": "bh-rn"},
    "Kerr-Newman": {"a": 0.5, "Q_BH": 0.3, "q": 0.0,
                     "desc": "Rotating *and* charged — the most general stationary BH.",
                     "badge": "bh-kn"},
}

# ─────────────────────────────────────────────────────────────────────────
# Common example presets
# ─────────────────────────────────────────────────────────────────────────
EXAMPLES = {
    "Stable Circular Orbit": {
        "desc": "A nearly circular orbit with tiny perturbation — the particle traces a clean ring around the black hole.",
        "bh": "Schwarzschild", "a": 0.0, "Q_BH": 0.0, "q": 0.0,
        "r0": 12.0, "theta0_deg": 90.0, "rdot0": 0.0, "thetadot0": 0.0,
        "E_factor": 1.0, "Lz_factor": 1.0, "tf": 10000,
    },
    "Precessing Ellipse": {
        "desc": "An elliptical orbit that precesses — the classic GR perihelion shift, visible as a rosette pattern.",
        "bh": "Schwarzschild", "a": 0.0, "Q_BH": 0.0, "q": 0.0,
        "r0": 20.0, "theta0_deg": 90.0, "rdot0": -0.00001, "thetadot0": 0.0,
        "E_factor": 1.001, "Lz_factor": 1.01, "tf": 25000,
    },
    "Plunging Orbit": {
        "desc": "A particle that spirals inward and crosses the event horizon — watch r(τ) plummet to r₊.",
        "bh": "Schwarzschild", "a": 0.0, "Q_BH": 0.0, "q": 0.0,
        "r0": 8.0, "theta0_deg": 90.0, "rdot0": -0.01, "thetadot0": 0.0,
        "E_factor": 0.97, "Lz_factor": 0.92, "tf": 5000,
    },
    "Frame-Dragged Spiral": {
        "desc": "A prograde orbit around a fast-spinning Kerr BH — frame-dragging twists the trajectory.",
        "bh": "Kerr", "a": 0.9, "Q_BH": 0.0, "q": 0.0,
        "r0": 10.0, "theta0_deg": 90.0, "rdot0": -0.00001, "thetadot0": 0.0,
        "E_factor": 1.002, "Lz_factor": 1.02, "tf": 10000,
    },
    "Off-Plane Torus": {
        "desc": "A tilted orbit that oscillates above and below the equatorial plane — creates a torus-shaped trajectory.",
        "bh": "Kerr", "a": 0.5, "Q_BH": 0.0, "q": 0.0,
        "r0": 15.0, "theta0_deg": 70.0, "rdot0": -0.00001, "thetadot0": 0.001,
        "E_factor": 1.001, "Lz_factor": 1.01, "tf": 25000,
    },
    "Charged Repulsion": {
        "desc": "A charged particle repelled by a like-charged Reissner-Nordström BH — the orbit is wider than neutral.",
        "bh": "Reissner-Nordström", "a": 0.0, "Q_BH": 0.3, "q": 1.0,
        "r0": 20.0, "theta0_deg": 90.0, "rdot0": -0.00001, "thetadot0": 0.0,
        "E_factor": 1.001, "Lz_factor": 1.01, "tf": 10000,
    },
    "Chaotic Kerr-Newman": {
        "desc": "A charged particle around a spinning, charged BH — complex interplay of frame-dragging and EM forces.",
        "bh": "Kerr-Newman", "a": 0.5, "Q_BH": 0.3, "q": -1.0,
        "r0": 15.0, "theta0_deg": 75.0, "rdot0": -0.0001, "thetadot0": 0.0005,
        "E_factor": 1.005, "Lz_factor": 1.03, "tf": 25000,
    },
    "Zoom-Whirl Orbit": {
        "desc": "The particle zooms in close to the BH, whirls several times, then zooms back out — a signature of strong-field GR.",
        "bh": "Schwarzschild", "a": 0.0, "Q_BH": 0.0, "q": 0.0,
        "r0": 20.0, "theta0_deg": 90.0, "rdot0": -0.0001, "thetadot0": 0.0,
        "E_factor": 0.975, "Lz_factor": 0.95, "tf": 10000,
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚫ Black Hole Explorer</div>', unsafe_allow_html=True)

    # ── Example presets ──
    st.subheader("Quick Examples")
    example_names = ["— Custom —"] + list(EXAMPLES.keys())
    chosen_example = st.selectbox(
        "Load a preset",
        example_names,
        index=0,
        help="Pick a famous orbit type to auto-fill all parameters below. Choose '— Custom —' to set your own."
    )

    if chosen_example != "— Custom —":
        ex = EXAMPLES[chosen_example]
        st.caption(f"*{ex['desc']}*")
    else:
        ex = None

    st.divider()

    # BH type selector
    default_bh = ex["bh"] if ex else "Schwarzschild"
    bh_idx = list(BH_TYPES.keys()).index(default_bh) if default_bh in BH_TYPES else 0
    bh_type = st.selectbox(
        "Black hole type", list(BH_TYPES.keys()), index=bh_idx,
        help="Selects the spacetime metric. Schwarzschild = simplest (no spin, no charge). "
             "Kerr = rotating. Reissner-Nordström = charged. Kerr-Newman = rotating + charged."
    )
    bh_info = BH_TYPES[bh_type]
    st.markdown(f'<span class="bh-badge {bh_info["badge"]}">{bh_type}</span>', unsafe_allow_html=True)
    st.caption(bh_info["desc"])

    st.divider()

    # ── Black hole parameters ──
    st.subheader("Black Hole Parameters")

    if bh_type in ("Kerr", "Kerr-Newman"):
        default_a = ex["a"] if ex else bh_info["a"]
        a_val = st.slider(
            "Spin  *a*", 0.0, 0.999, default_a, 0.01,
            help="Dimensionless spin parameter J/M². Ranges from 0 (non-rotating) to ~1 (extremal). "
                 "Higher spin → stronger frame-dragging, pulling nearby objects into co-rotation."
        )
    else:
        a_val = 0.0
        st.text("Spin a = 0  (fixed for this type)")

    if bh_type in ("Reissner-Nordström", "Kerr-Newman"):
        max_Q = float(np.sqrt(max(1 - a_val ** 2 - 0.001, 0.001)))
        default_Q = ex["Q_BH"] if ex else bh_info["Q_BH"]
        Q_val = st.slider(
            "BH charge  *Q*", 0.0, min(max_Q, 0.999), min(default_Q, max_Q), 0.01,
            help="Black hole electric charge in natural units. Must satisfy a²+Q²<1 (sub-extremal). "
                 "Charge modifies the horizon structure and creates an electromagnetic field."
        )
    else:
        Q_val = 0.0
        st.text("Charge Q = 0  (fixed for this type)")

    # Sub-extremal check
    extremal = a_val ** 2 + Q_val ** 2
    if extremal >= 1.0:
        st.error(f"a² + Q² = {extremal:.3f} ≥ 1  — super-extremal!")
    else:
        rp = 1 + np.sqrt(1 - extremal)
        rm = 1 - np.sqrt(1 - extremal)
        st.success(f"a²+Q² = {extremal:.3f}  ·  r₊ = {rp:.3f}  ·  r₋ = {rm:.3f}")

    st.divider()

    # ── Particle parameters ──
    st.subheader("Test Particle")
    if bh_type in ("Reissner-Nordström", "Kerr-Newman"):
        default_q = ex["q"] if ex else 0.0
        q_val = st.select_slider(
            "Particle charge  *q*", options=[-1.0, 0.0, 1.0], value=default_q,
            help="Electric charge of the test particle. +1 or −1 couples it to the BH's electromagnetic field; "
                 "0 = neutral (pure geodesic). Same-sign charges repel; opposite attract."
        )
    else:
        q_val = 0.0
        st.text("q = 0 (neutral — no EM coupling)")

    st.divider()

    # ── Initial conditions ──
    st.subheader("Initial Conditions")
    default_r0 = ex["r0"] if ex else 20.0
    r0 = st.slider(
        "Initial radius  r₀", 4.0, 50.0, default_r0, 0.5,
        help="Starting radial coordinate in units of M (half the Schwarzschild radius). "
             "Larger r₀ → weaker gravity. The ISCO is at r ≈ 6M for Schwarzschild."
    )

    default_th = ex["theta0_deg"] if ex else 90.0
    theta0_deg = st.slider(
        "Initial polar angle  θ₀ (degrees)", 0.1, 179.9, default_th, 0.1,
        help="Polar angle from the spin axis. 90° = equatorial plane. "
             "Values ≠ 90° give orbits that bob above and below the equator."
    )
    theta0 = np.deg2rad(theta0_deg)

    default_rdot = ex["rdot0"] if ex else -0.00001
    rdot0 = st.number_input(
        "Initial radial velocity  ṙ₀", value=default_rdot, format="%.6f", step=0.00001,
        help="dr/dτ at the start. Negative = falling inward, positive = moving outward. "
             "Zero = no initial radial motion (tangential orbit)."
    )

    default_thdot = ex["thetadot0"] if ex else 0.0
    thetadot0 = st.number_input(
        "Initial polar angular velocity  θ̇₀", value=default_thdot, format="%.6f", step=0.0001,
        help="dθ/dτ at the start. Non-zero tilts the orbit out of the equatorial plane "
             "and produces 3D motion (θ oscillation → torus shapes)."
    )

    st.divider()

    # ── Orbit perturbation ──
    st.subheader("Orbit Perturbation")
    default_Ef = ex["E_factor"] if ex else 1.001
    E_factor = st.slider(
        "Energy multiplier", 0.95, 1.10, default_Ef, 0.001,
        help="Scales the circular-orbit energy. >1 = more energy (wider/unbound). "
             "<1 = less energy (tighter, may plunge). 1.0 = exact circular orbit."
    )

    default_Lz = ex["Lz_factor"] if ex else 1.01
    Lz_factor = st.slider(
        "Angular momentum multiplier", 0.90, 1.15, default_Lz, 0.001,
        help="Scales the circular-orbit angular momentum Lz. >1 = more Lz (wider). "
             "<1 = less Lz (may plunge). 1.0 = exact circular orbit."
    )

    st.divider()

    # ── Integration ──
    st.subheader("Integration")
    default_tf = ex["tf"] if ex else 10000
    tf = st.select_slider(
        "Final time  τ_f",
        options=[1000, 5000, 10000, 25000, 50000, 100000],
        value=default_tf,
        help="Total proper time to integrate. Proper time τ is the clock carried by "
             "the orbiting particle. Longer = more orbits but slower to compute."
    )
    dt = st.number_input(
        "Initial time step  dt", value=0.1, format="%.3f", step=0.01,
        help="Starting step size for the adaptive RKF45 integrator. "
             "0.1 is recommended for most orbits. "
             "It auto-adjusts — smaller = more accurate start but slower."
    )
    st.caption("*Recommended: dt = 0.1*")

    st.divider()
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────
st.title("🕳️ Black Hole Geodesic Explorer")
st.markdown(f"**Metric:** {bh_type} &nbsp;|&nbsp; *a* = {a_val}, *Q* = {Q_val}, *q* = {q_val}")

# ─────────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────────

if run_btn or "result" not in st.session_state:
    params = SimParams(
        a=a_val, Q_BH=Q_val, q=q_val,
        r0=r0, theta0=theta0, phi0=0.0,
        rdot0=rdot0, thetadot0=thetadot0,
        E_factor=E_factor, Lz_factor=Lz_factor,
        dt=dt, tf=float(tf),
    )
    progress_bar = st.progress(0, text="Setting up…")

    def update_progress(frac, msg):
        progress_bar.progress(min(frac, 1.0), text=msg)

    try:
        result = run_simulation(params, progress_callback=update_progress)
        st.session_state["result"] = result
        st.session_state["params"] = params
        progress_bar.empty()
    except Exception as e:
        progress_bar.empty()
        st.error(f"Simulation failed: {e}")
        st.stop()

res = st.session_state.get("result")
if res is None:
    st.info("Click **Run Simulation** in the sidebar to start.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# Summary metrics row
# ─────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><h3>Energy E</h3><p>{res.E_init:.6f}</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h3>Ang. Mom. Lz</h3><p>{res.Lz_init:.6f}</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h3>Carter Q</h3><p>{res.Q_init:.6f}</p></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h3>r₊</h3><p>{res.r_plus:.4f}</p></div>',
                unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><h3>Steps</h3><p>{res.total_steps:,}</p></div>',
                unsafe_allow_html=True)

st.caption(f"**Termination:** {res.termination}")

# ─────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────
(tab_3d, tab_2d, tab_coords, tab_energy, tab_angmom,
 tab_carter, tab_norm, tab_potentials, tab_examples, tab_log) = st.tabs([
    "🌐 3D Trajectory",
    "📐 2D Projections",
    "📈 Coordinates vs τ",
    "⚡ Energy",
    "🔄 Angular Momentum",
    "🎯 Carter Constant",
    "📏 Metric Norm",
    "🔬 Potentials",
    "💡 Examples",
    "📋 Log",
])

# ── Plotly theme ──
PLOTLY_TEMPLATE = "plotly_dark"
TRAJ_COLOR = "#00d4ff"
HORIZON_COLOR = "#ff4060"
ERGO_COLOR = "#6c63ff"


# ─────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────

def _sphere_mesh3d(r_h, a_val, color, name, opacity=0.25, n_u=24, n_v=12):
    """
    Lightweight Mesh3d sphere (much fewer WebGL draw-calls than Surface).
    Uses oblate-spheroidal conversion matching Boyer-Lindquist coords.
    """
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v + 1)  # include poles
    U, V = np.meshgrid(u, v)
    rho = np.sqrt(r_h ** 2 + a_val ** 2)
    x = (rho * np.sin(V) * np.cos(U)).ravel()
    y = (rho * np.sin(V) * np.sin(U)).ravel()
    z = (r_h * np.cos(V)).ravel()
    # Build triangle indices for the mesh
    ii, jj, kk = [], [], []
    for iv in range(n_v):
        for iu in range(n_u):
            p00 = iv * n_u + iu
            p01 = iv * n_u + (iu + 1) % n_u
            p10 = (iv + 1) * n_u + iu
            p11 = (iv + 1) * n_u + (iu + 1) % n_u
            ii += [p00, p00]
            jj += [p01, p10]
            kk += [p10, p11]
    return go.Mesh3d(x=x, y=y, z=z, i=ii, j=jj, k=kk,
                     color=color, opacity=opacity, name=name,
                     hoverinfo="name", flatshading=True)


def horizon_mesh(r_h, a_val, color, name, opacity=0.25):
    return _sphere_mesh3d(r_h, a_val, color, name, opacity, n_u=24, n_v=12)


def ergosphere_mesh(a_val, Q_val):
    """Lightweight wireframe-style ergosphere using Scatter3d lines."""
    traces = []
    n_ph = 24
    n_th = 20
    ph_arr = np.linspace(0, 2 * np.pi, n_ph)
    th_arr = np.linspace(0.05, np.pi - 0.05, n_th)
    # Draw latitude rings
    for th_val in th_arr[::4]:
        arg = 1 - a_val ** 2 * np.cos(th_val) ** 2 - Q_val ** 2
        if arg < 0:
            continue
        r_e = 1 + np.sqrt(arg)
        rho_e = np.sqrt(r_e ** 2 + a_val ** 2)
        xe = rho_e * np.sin(th_val) * np.cos(ph_arr)
        ye = rho_e * np.sin(th_val) * np.sin(ph_arr)
        ze = np.full_like(ph_arr, r_e * np.cos(th_val))
        traces.append(go.Scatter3d(
            x=xe, y=ye, z=ze, mode="lines",
            line=dict(color=ERGO_COLOR, width=1.5),
            name="Ergosphere", showlegend=bool(th_val == th_arr[0]),
            hoverinfo="name"))
    return traces


def horizon_circle_2d(r_h, a_val, n=200):
    ang = np.linspace(0, 2 * np.pi, n)
    rho = np.sqrt(r_h ** 2 + a_val ** 2)
    return rho * np.cos(ang), rho * np.sin(ang)


def make_2d_fig(px, py, lx, ly, res, proj_name):
    """Build a single 2D projection figure with horizon and singularity."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=px, y=py, mode="lines",
                                line=dict(color=TRAJ_COLOR, width=1), name="Trajectory"))
    fig.add_trace(go.Scatter(x=[px[-1]], y=[py[-1]], mode="markers",
                              marker=dict(color="red", size=8), name="Final"))
    # Horizon
    hx, hy = horizon_circle_2d(res.r_plus, res.a)
    if proj_name == "xz":
        hy_p = res.r_plus * np.sin(np.linspace(0, 2 * np.pi, 200))
        fig.add_trace(go.Scatter(x=hx, y=hy_p, mode="lines",
                                  line=dict(color=HORIZON_COLOR, width=2, dash="dash"),
                                  name=f"Horizon r₊={res.r_plus:.2f}"))
    elif proj_name == "yz":
        hy2 = np.sqrt(res.r_plus ** 2 + res.a ** 2) * np.sin(np.linspace(0, 2 * np.pi, 200))
        hz2 = res.r_plus * np.cos(np.linspace(0, 2 * np.pi, 200))
        fig.add_trace(go.Scatter(x=hy2, y=hz2, mode="lines",
                                  line=dict(color=HORIZON_COLOR, width=2, dash="dash"),
                                  name=f"Horizon r₊={res.r_plus:.2f}"))
    else:
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines",
                                  line=dict(color=HORIZON_COLOR, width=2, dash="dash"),
                                  name=f"Horizon r₊={res.r_plus:.2f}"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                              marker=dict(color="white", size=6, symbol="x"),
                              name="Singularity"))
    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title=lx, yaxis_title=ly,
                      title=f"{lx} vs {ly}", height=550,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


# ─────────────────────────────────────────────────────────────────────────
# TAB: 3D Trajectory
# ─────────────────────────────────────────────────────────────────────────
with tab_3d:
    fig3d = go.Figure()

    # Downsample trajectory to at most 15 000 points for WebGL stability
    MAX_3D_PTS = 15_000
    n_pts = len(res.x)
    if n_pts > MAX_3D_PTS:
        idx = np.linspace(0, n_pts - 1, MAX_3D_PTS, dtype=int)
        x3, y3, z3 = res.x[idx], res.y[idx], res.z[idx]
    else:
        x3, y3, z3 = res.x, res.y, res.z

    fig3d.add_trace(go.Scatter3d(
        x=x3, y=y3, z=z3,
        mode="lines", line=dict(color=TRAJ_COLOR, width=2), name="Geodesic"))
    fig3d.add_trace(go.Scatter3d(
        x=[res.x[-1]], y=[res.y[-1]], z=[res.z[-1]],
        mode="markers", marker=dict(color="red", size=5), name="Final position"))

    fig3d.add_trace(horizon_mesh(res.r_plus, res.a, HORIZON_COLOR, f"Outer horizon r₊={res.r_plus:.2f}"))
    if res.r_minus > 0.01:
        fig3d.add_trace(horizon_mesh(res.r_minus, res.a, "#ff8800",
                                     f"Inner horizon r₋={res.r_minus:.2f}", opacity=0.15))
    if abs(res.a) > 1e-6:
        ergo_traces = ergosphere_mesh(res.a, res.Q_BH)
        for tr in ergo_traces:
            fig3d.add_trace(tr)
        ang = np.linspace(0, 2 * np.pi, 100)
        fig3d.add_trace(go.Scatter3d(
            x=res.a * np.cos(ang), y=res.a * np.sin(ang), z=np.zeros(100),
            mode="lines", line=dict(color="white", width=4), name="Ring singularity"))

    rng = max(res.x.max() - res.x.min(), res.y.max() - res.y.min(),
              res.z.max() - res.z.min()) / 2 * 1.05
    mx = (res.x.max() + res.x.min()) / 2
    my = (res.y.max() + res.y.min()) / 2
    mz = (res.z.max() + res.z.min()) / 2
    fig3d.update_layout(
        template=PLOTLY_TEMPLATE,
        scene=dict(
            xaxis=dict(range=[mx - rng, mx + rng], title="X"),
            yaxis=dict(range=[my - rng, my + rng], title="Y"),
            zaxis=dict(range=[mz - rng, mz + rng], title="Z"),
            aspectmode="cube",
        ),
        title=f"3D Geodesic — {bh_type} (a={res.a}, Q={res.Q_BH}, q={res.q})",
        height=700, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True, key="plot_3d")

# ─────────────────────────────────────────────────────────────────────────
# TAB: 2D Projections — all 3 planes stacked vertically
# ─────────────────────────────────────────────────────────────────────────
with tab_2d:
    st.markdown("### X – Y  (top view)")
    st.plotly_chart(make_2d_fig(res.x, res.y, "X", "Y", res, "xy"),
                    use_container_width=True, key="plot_xy")

    st.markdown("---")
    st.markdown("### X – Z  (side view)")
    st.plotly_chart(make_2d_fig(res.x, res.z, "X", "Z", res, "xz"),
                    use_container_width=True, key="plot_xz")

    st.markdown("---")
    st.markdown("### Y – Z")
    st.plotly_chart(make_2d_fig(res.y, res.z, "Y", "Z", res, "yz"),
                    use_container_width=True, key="plot_yz")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Coordinates vs τ
# ─────────────────────────────────────────────────────────────────────────
with tab_coords:
    # ── r(τ) ──
    st.markdown("### Radius  r(τ)")
    fig_r = go.Figure()
    fig_r.add_trace(go.Scattergl(x=res.tau, y=res.r, mode="lines",
                                  line=dict(color=TRAJ_COLOR, width=1.5), name="r"))
    fig_r.add_hline(y=res.r_plus, line_dash="dash", line_color=HORIZON_COLOR,
                    annotation_text=f"r₊ = {res.r_plus:.2f}")
    fig_r.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Proper time  τ",
                        yaxis_title="r  (radial coordinate)", title="r(τ)", height=450)
    st.plotly_chart(fig_r, use_container_width=True, key="plot_r_tau")

    st.markdown("---")

    # ── θ(τ) ──
    st.markdown("### Polar Angle  θ(τ)")
    fig_th = go.Figure()
    fig_th.add_trace(go.Scattergl(x=res.tau, y=res.theta, mode="lines",
                                   line=dict(color="#ffd166", width=1.5), name="θ"))
    fig_th.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Proper time  τ",
                         yaxis_title="θ  (polar angle, rad)", title="θ(τ)", height=450)
    st.plotly_chart(fig_th, use_container_width=True, key="plot_th_tau")

    st.markdown("---")

    # ── φ(τ) ──
    st.markdown("### Azimuthal Angle  φ(τ)")
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scattergl(x=res.tau, y=res.phi, mode="lines",
                                   line=dict(color="#06d6a0", width=1.5), name="φ"))
    fig_ph.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Proper time  τ",
                         yaxis_title="φ  (azimuthal angle, rad)", title="φ(τ)", height=450)
    st.plotly_chart(fig_ph, use_container_width=True, key="plot_ph_tau")

    st.markdown("---")

    # ── t(τ) ──
    st.markdown("### Coordinate Time  t(τ)")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scattergl(x=res.tau, y=res.t, mode="lines",
                                  line=dict(color="#ef476f", width=1.5), name="t"))
    fig_t.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Proper time  τ",
                        yaxis_title="Coordinate time  t", title="t(τ)", height=450)
    st.plotly_chart(fig_t, use_container_width=True, key="plot_t_tau")

    st.markdown("---")

    # ── Carter constant Q(τ) ──
    st.markdown("### Carter Constant  Q(τ)")
    st.caption("The Carter constant governs polar oscillation — a flat line confirms conservation.")
    fig_qc = go.Figure()
    fig_qc.add_trace(go.Scattergl(x=res.tau, y=res.Q_check, mode="lines",
                                   line=dict(color="#ffd166", width=1.5), name="Q(τ)"))
    fig_qc.add_hline(y=res.Q_init, line_dash="dash", line_color="#ff4060",
                     annotation_text=f"Q₀ = {res.Q_init:.6f}")
    fig_qc.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Proper time  τ",
                         yaxis_title="Q", title="Carter Constant Q(τ)", height=450)
    st.plotly_chart(fig_qc, use_container_width=True, key="plot_q_tau")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Energy
# ─────────────────────────────────────────────────────────────────────────
with tab_energy:
    st.markdown("#### Energy  *E*")
    st.caption("The canonical energy is conserved along the geodesic. "
               "For Kerr-Newman it includes the electromagnetic coupling −qA_t. "
               "A flat line confirms the integrator is preserving this symmetry.")
    fig_e = go.Figure()
    fig_e.add_trace(go.Scattergl(x=res.tau, y=res.E_check, mode="lines",
                                  line=dict(color="#00d4ff", width=1), name="E(τ)"))
    fig_e.add_hline(y=res.E_init, line_dash="dash", line_color="#ff4060",
                    annotation_text=f"E₀ = {res.E_init:.6f}")
    fig_e.update_layout(template=PLOTLY_TEMPLATE, title="Energy vs Proper Time",
                        xaxis_title="Proper time  τ", yaxis_title="E", height=500)
    st.plotly_chart(fig_e, use_container_width=True, key="plot_energy")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Angular Momentum
# ─────────────────────────────────────────────────────────────────────────
with tab_angmom:
    st.markdown("#### Angular Momentum  *Lz*")
    st.caption("The z-component of canonical angular momentum. Includes +qA_φ for charged particles. "
               "Its conservation reflects the axial symmetry of the Kerr-Newman spacetime.")
    fig_l = go.Figure()
    fig_l.add_trace(go.Scattergl(x=res.tau, y=res.Lz_check, mode="lines",
                                  line=dict(color="#06d6a0", width=1), name="Lz(τ)"))
    fig_l.add_hline(y=res.Lz_init, line_dash="dash", line_color="#ff4060",
                    annotation_text=f"Lz₀ = {res.Lz_init:.6f}")
    fig_l.update_layout(template=PLOTLY_TEMPLATE, title="Angular Momentum vs Proper Time",
                        xaxis_title="Proper time  τ", yaxis_title="Lz", height=500)
    st.plotly_chart(fig_l, use_container_width=True, key="plot_angmom")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Carter Constant
# ─────────────────────────────────────────────────────────────────────────
with tab_carter:
    st.markdown("#### Carter Constant  *Q*")
    st.caption("A hidden constant of motion unique to Kerr-type spacetimes, arising from a Killing tensor. "
               "It governs the polar (θ) oscillation. Q = 0 confines motion to the equatorial plane.")
    fig_q = go.Figure()
    fig_q.add_trace(go.Scattergl(x=res.tau, y=res.Q_check, mode="lines",
                                  line=dict(color="#ffd166", width=1), name="Q(τ)"))
    fig_q.add_hline(y=res.Q_init, line_dash="dash", line_color="#ff4060",
                    annotation_text=f"Q₀ = {res.Q_init:.6f}")
    fig_q.update_layout(template=PLOTLY_TEMPLATE, title="Carter Constant vs Proper Time",
                        xaxis_title="Proper time  τ", yaxis_title="Q", height=500)
    st.plotly_chart(fig_q, use_container_width=True, key="plot_carter")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Metric Norm
# ─────────────────────────────────────────────────────────────────────────
with tab_norm:
    st.markdown("#### Metric Norm  *g*_μν ẋᵘẋᵛ")
    st.caption("For a massive particle on a timelike geodesic this must equal −1 everywhere. "
               "Deviations indicate numerical error in the integrator.")
    fig_n = go.Figure()
    fig_n.add_trace(go.Scattergl(x=res.tau, y=res.metric_norm, mode="lines",
                                  line=dict(color="#ef476f", width=1), name="norm(τ)"))
    fig_n.add_hline(y=-1.0, line_dash="dash", line_color="#ff4060",
                    annotation_text="Expected: −1")
    fig_n.update_layout(template=PLOTLY_TEMPLATE, title="Metric Norm vs Proper Time",
                        xaxis_title="Proper time  τ", yaxis_title="g_μν ẋᵘẋᵛ", height=500)
    st.plotly_chart(fig_n, use_container_width=True, key="plot_norm")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Potentials
# ─────────────────────────────────────────────────────────────────────────
with tab_potentials:
    st.markdown("#### Potential Consistency")
    st.caption("The separated first-order equations give p_r² = R(r) and p_θ² = Θ(θ). "
               "If the curves overlap perfectly the integration is self-consistent.")
    cols = st.columns(2)
    with cols[0]:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scattergl(x=res.tau, y=res.p_r ** 2, mode="lines",
                                       line=dict(color="#00d4ff", width=1), name="p_r²"))
        fig_pr.add_trace(go.Scattergl(x=res.tau, y=res.R_arr, mode="lines",
                                       line=dict(color="#ff4060", width=1, dash="dash"), name="R(r)"))
        fig_pr.update_layout(template=PLOTLY_TEMPLATE, title="Radial: p_r² vs R(r)",
                             xaxis_title="Proper time  τ", height=400)
        st.plotly_chart(fig_pr, use_container_width=True, key="plot_pr")
    with cols[1]:
        fig_pth = go.Figure()
        fig_pth.add_trace(go.Scattergl(x=res.tau, y=res.p_th ** 2, mode="lines",
                                        line=dict(color="#06d6a0", width=1), name="p_θ²"))
        fig_pth.add_trace(go.Scattergl(x=res.tau, y=res.Theta_arr, mode="lines",
                                        line=dict(color="#ffd166", width=1, dash="dash"), name="Θ(θ)"))
        fig_pth.update_layout(template=PLOTLY_TEMPLATE, title="Polar: p_θ² vs Θ(θ)",
                              xaxis_title="Proper time  τ", height=400)
        st.plotly_chart(fig_pth, use_container_width=True, key="plot_pth")

# ─────────────────────────────────────────────────────────────────────────
# TAB: Examples
# ─────────────────────────────────────────────────────────────────────────
with tab_examples:
    st.markdown("### Common Starting Conditions")
    st.caption("To use an example: select it from the **Load a preset** dropdown in the sidebar, "
               "then press **▶ Run Simulation**.")

    for name, ex_info in EXAMPLES.items():
        st.markdown(f"""<div class="example-card">
            <h4>{name}</h4>
            <p>{ex_info['desc']}</p>
            <p style="margin-top:0.4rem; color:#888; font-size:0.8rem;">
                <b>BH:</b> {ex_info['bh']} &nbsp;|&nbsp;
                <b>a</b>={ex_info['a']}, <b>Q</b>={ex_info['Q_BH']}, <b>q</b>={ex_info['q']} &nbsp;|&nbsp;
                <b>r₀</b>={ex_info['r0']}, <b>θ₀</b>={ex_info['theta0_deg']}° &nbsp;|&nbsp;
                <b>ṙ₀</b>={ex_info['rdot0']}, <b>θ̇₀</b>={ex_info['thetadot0']} &nbsp;|&nbsp;
                <b>E×</b>{ex_info['E_factor']}, <b>Lz×</b>{ex_info['Lz_factor']} &nbsp;|&nbsp;
                <b>τ_f</b>={ex_info['tf']}
            </p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB: Log
# ─────────────────────────────────────────────────────────────────────────
with tab_log:
    st.markdown("#### Simulation Log")
    st.code(res.log, language="text")
    st.markdown("#### Final Summary")
    st.markdown(f"""
| Quantity | Value |
|----------|-------|
| **E** | {res.E_init:.8f} |
| **Lz** | {res.Lz_init:.8f} |
| **Q** | {res.Q_init:.8f} |
| **Metric norm (final)** | {res.metric_norm[-1]:.12f} |
| **Steps** | {res.total_steps:,} |
""")
