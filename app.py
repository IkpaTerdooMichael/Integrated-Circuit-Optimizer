"""
IC Optimizer - ML-driven analog circuit sizing demo.
Bayesian optimization of a 2-stage CMOS op-amp for Power/Performance/Area.

Deploy: Streamlit Community Cloud (free) -> point at your GitHub repo -> app.py
Run locally: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="IC Optimizer", layout="wide")
st.title("IC Optimizer - ML-Driven Analog Circuit Sizing")
st.caption(
    "Bayesian optimization of a 2-stage CMOS op-amp. "
    "Demonstrates ML applied to IC design-space exploration (PPA trade-offs)."
)

# ---------------------------------------------------------------------------
# Surrogate circuit model
# (Closed-form analytical model so the app runs anywhere - no ngspice needed.
#  In a real flow this would call SPICE.)
# ---------------------------------------------------------------------------
def simulate_opamp(W1, W3, W5, W6, Ibias):
    """
    Simplified analytical model of a 2-stage CMOS op-amp.
    Returns dict of performance metrics.
    """
    L = 0.18          # um
    Vdd = 1.8         # V
    Cox = 8.6e-3      # F/m^2 (approx, 0.18um tech)

    # Intrinsic gains (rough)
    gm1 = 2 * Ibias * 1e-6 / 0.2          # A/V
    ro1 = 1.0 / (0.05 * Ibias * 1e-6)
    A1 = gm1 * ro1

    gm5 = 2 * (Ibias * 1e-6) / 0.2
    ro5 = 1.0 / (0.05 * Ibias * 1e-6)
    A2 = gm5 * ro5

    gain_db = 20 * np.log10(max(A1 * A2, 1))

    # Dominant pole / GBW estimate
    Cc = 1e-12                             # 1 pF compensation
    gbw = gm1 / (2 * np.pi * Cc)           # Hz

    # Power & area
    power_mw = Vdd * (2 * Ibias) * 1e-3    # mW
    area_um2 = (W1 + W3 + W5 + W6) * L

    return {
        "gain_db": float(gain_db),
        "gbw_hz": float(gbw),
        "power_mw": float(power_mw),
        "area_um2": float(area_um2),
    }


def cost_fn(params, gain_target, gbw_target, w_power, w_area, w_pen):
    W1, W3, W5, W6, Ib = params
    m = simulate_opamp(W1, W3, W5, W6, Ib)

    # Spec penalties (soft constraints)
    pen = 0.0
    if m["gain_db"] < gain_target:
        pen += (gain_target - m["gain_db"]) * w_pen
    if m["gbw_hz"] < gbw_target:
        pen += (gbw_target - m["gbw_hz"]) / gbw_target * w_pen

    return w_power * m["power_mw"] + w_area * (m["area_um2"] / 100.0) + pen


# ---------------------------------------------------------------------------
# Sidebar - targets & weights
# ---------------------------------------------------------------------------
st.sidebar.header("Design Targets")
gain_target = st.sidebar.slider("Min DC gain (dB)", 30.0, 80.0, 60.0, 1.0)
gbw_target = st.sidebar.slider(
    "Min GBW (MHz)", 1.0, 200.0, 10.0, 1.0
) * 1e6

st.sidebar.header("Objective Weights")
w_power = st.sidebar.slider("Power weight", 0.0, 5.0, 1.0, 0.1)
w_area = st.sidebar.slider("Area weight", 0.0, 5.0, 1.0, 0.1)
w_pen = st.sidebar.slider("Spec penalty weight", 0.0, 50.0, 10.0, 1.0)

st.sidebar.header("Optimizer")
n_calls = st.sidebar.slider("BO iterations", 20, 150, 60, 5)

run = st.sidebar.button("Run Optimization", type="primary")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = None

if run:
    space = [
        Real(1.0, 50.0, name="W1"),
        Real(1.0, 50.0, name="W3"),
        Real(1.0, 100.0, name="W5"),
        Real(1.0, 100.0, name="W6"),
        Real(1.0, 50.0, name="Ibias"),
    ]

    history = []
    progress = st.progress(0.0, text="Running Bayesian optimization...")

    def wrapped(p):
        c = cost_fn(p, gain_target, gbw_target, w_power, w_area, w_pen)
        m = simulate_opamp(*p)
        history.append({
            "W1": p[0], "W3": p[1], "W5": p[2], "W6": p[3], "Ibias": p[4],
            "cost": c, **m
        })
        progress.progress(len(history) / n_calls,
                          text=f"Iteration {len(history)}/{n_calls}")
        return c

    res = gp_minimize(wrapped, space, n_calls=n_calls, random_state=0)
    progress.empty()
    st.session_state.history = pd.DataFrame(history)
    st.session_state.best = res

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.history is not None:
    df = st.session_state.history
    best_row = df.loc[df["cost"].idxmin()]

    st.subheader("Optimized Design")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DC Gain", f"{best_row['gain_db']:.1f} dB",
              delta=f"{best_row['gain_db'] - gain_target:+.1f} vs target")
    c2.metric("GBW", f"{best_row['gbw_hz']/1e6:.1f} MHz",
              delta=f"{(best_row['gbw_hz'] - gbw_target)/1e6:+.1f} vs target")
    c3.metric("Power", f"{best_row['power_mw']:.2f} mW")
    c4.metric("Area", f"{best_row['area_um2']:.1f} um2")

    st.subheader("Optimal Sizing")
    sizing = {
        "W1 (um)": f"{best_row['W1']:.2f}",
        "W3 (um)": f"{best_row['W3']:.2f}",
        "W5 (um)": f"{best_row['W5']:.2f}",
        "W6 (um)": f"{best_row['W6']:.2f}",
        "Ibias (uA)": f"{best_row['Ibias']:.2f}",
    }
    st.table(pd.DataFrame(sizing.items(), columns=["Parameter", "Value"]))

    # Convergence plot
    st.subheader("Convergence")
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df["cost"], marker="o", ms=3, lw=1)
    ax1.plot(df["cost"].cummin(), lw=2, label="Best so far")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost (lower = better)")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    # PPA scatter
    st.subheader("PPA Trade-off Space")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sc = ax2.scatter(df["power_mw"], df["area_um2"],
                     c=df["gain_db"], cmap="viridis", s=30)
    ax2.scatter(best_row["power_mw"], best_row["area_um2"],
                color="red", marker="*", s=250, label="Best")
    ax2.set_xlabel("Power (mW)")
    ax2.set_ylabel("Area (um2)")
    ax2.legend()
    plt.colorbar(sc, label="DC Gain (dB)")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    # Full history
    st.subheader("Full Optimization History")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False),
                       "ic_optimizer_results.csv", "text/csv")

else:
    st.info("Set your targets in the sidebar and click Run Optimization.")
    st.markdown(
        """
        ### How it works
        1. **Design variables** - transistor widths `W1, W3, W5, W6` and bias current `Ibias`.
        2. **Surrogate model** - analytical estimate of gain, GBW, power, area.
        3. **Objective** - weighted sum of power + area + spec-miss penalties.
        4. **Optimizer** - Gaussian-process Bayesian optimization (`gp_minimize`).

        Swap `simulate_opamp()` for a real SPICE call (PySpice / ngspice) to make it production-grade.
        """
    )

st.divider()
st.caption("Built for SSI2 portfolio - Streamlit + scikit-optimize - MIT License")
