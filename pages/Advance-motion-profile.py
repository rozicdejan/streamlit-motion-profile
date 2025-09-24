import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================
# Motion & Physics Helpers
# =========================
def calc_profile(positions, acc_time, dec_time, max_velocity=None):
    if len(positions) < 2:
        return None

    segments = []
    total_time = 0.0
    phase_markers = []
    seg_end_times = []

    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i+1]
        dist = abs(p1 - p0)
        dirn = 1 if p1 >= p0 else -1

        if dist == 0:
            dt_idle = 0.1
            segments.append(dict(p0=p0, p1=p1, dist=0.0, dir=dirn,
                                 t_acc=0.0, t_const=dt_idle, t_dec=0.0,
                                 vmax=0.0, t0=total_time, t1=total_time+dt_idle))
            phase_markers.append({
                "t_acc_end": total_time,
                "t_const_end": total_time + dt_idle,
                "t_dec_end": total_time + dt_idle
            })
            total_time += dt_idle
            seg_end_times.append(total_time)
            continue

        if (max_velocity is None) and (acc_time + dec_time) > 0:
            vmax = 2 * dist / (acc_time + dec_time)
            t_const = 0.0
        else:
            vmax = max_velocity if max_velocity is not None else dist
            acc_dec_dist = 0.5 * vmax * (acc_time + dec_time)
            if acc_dec_dist >= dist:  # triangular
                vmax = np.sqrt(2 * dist / (acc_time + dec_time)) if (acc_time + dec_time) > 0 else dist
                t_const = 0.0
            else:
                t_const = (dist - acc_dec_dist) / vmax

        t_seg = acc_time + t_const + dec_time

        segments.append(dict(
            p0=p0, p1=p1, dist=dist, dir=dirn,
            t_acc=acc_time, t_const=t_const, t_dec=dec_time,
            vmax=vmax * dirn, t0=total_time, t1=total_time + t_seg
        ))

        phase_markers.append({
            "t_acc_end": total_time + acc_time,
            "t_const_end": total_time + acc_time + t_const,
            "t_dec_end": total_time + t_seg
        })

        total_time += t_seg
        seg_end_times.append(total_time)

    dt = 0.01
    t = np.arange(0, total_time + dt, dt)
    pos = np.zeros_like(t)
    vel = np.zeros_like(t)
    acc = np.zeros_like(t)
    cum_dist = np.zeros_like(t)

    for i, tt in enumerate(t):
        seg = None
        for s in segments:
            if s['t0'] <= tt <= s['t1']:
                seg = s
                break
        if seg is None:
            pos[i] = positions[-1]
            vel[i] = 0.0
            acc[i] = 0.0
            if i > 0:
                cum_dist[i] = cum_dist[i-1]
            continue

        tau = tt - seg['t0']
        t_acc, t_const, t_dec = seg['t_acc'], seg['t_const'], seg['t_dec']
        vmax, p0 = seg['vmax'], seg['p0']

        if seg['dist'] == 0:
            a = 0.0; v = 0.0; p = p0
        elif tau <= t_acc:
            a = vmax / t_acc if t_acc > 0 else 0.0
            v = a * tau
            p = p0 + 0.5 * a * tau**2
        elif tau <= t_acc + t_const:
            a = 0.0
            v = vmax
            p = p0 + 0.5 * vmax * t_acc + vmax * (tau - t_acc)
        else:
            td = tau - t_acc - t_const
            a = -vmax / t_dec if t_dec > 0 else 0.0
            v = vmax + a * td
            d_acc_const = 0.5 * vmax * t_acc + vmax * t_const
            p = p0 + d_acc_const + vmax * td + 0.5 * a * td**2

        pos[i] = p
        vel[i] = v
        acc[i] = a
        if i > 0:
            cum_dist[i] = cum_dist[i-1] + abs(v) * dt

    cum_total = np.sum([abs(positions[i+1] - positions[i]) for i in range(len(positions)-1)])
    cum_dist[-1] = cum_total

    return t, pos, vel, acc, cum_dist, phase_markers, seg_end_times


def estimate_power(acc_units, vel_units, unit_to_m=0.001, mass_kg=5.0,
                   coulomb_fric_N=0.0, visc_fric_N_per_unit_s=0.0):
    acc_m = acc_units * unit_to_m
    vel_m = vel_units * unit_to_m
    sign_v = np.sign(vel_m)
    F = mass_kg * acc_m + coulomb_fric_N * sign_v + visc_fric_N_per_unit_s * vel_units
    P = F * vel_m
    dt = 0.01
    energy = np.cumsum(P) * dt
    energy_pos = np.cumsum(np.where(P > 0, P, 0.0)) * dt
    return P, energy, energy_pos


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Motion Profile Lab", layout="wide")
    st.title("🚀 Motion Profile Lab")
    st.caption("Overlay profiles • Shaded phases • Cumulative distance • Power/Energy • Playback")

    # Sidebar — Inputs
    st.sidebar.header("Waypoints & Parameters")
    unit = st.sidebar.selectbox("Distance unit", ["mm", "m"], index=0)
    unit_to_m = 0.001 if unit == "mm" else 1.0

    if 'positions' not in st.session_state:
        st.session_state.positions = [0, 500, 0, 560, 0, 620, 0]

    npts = st.sidebar.number_input("Number of waypoints", 2, 50, len(st.session_state.positions))
    if len(st.session_state.positions) < npts:
        st.session_state.positions.extend([0] * (npts - len(st.session_state.positions)))
    elif len(st.session_state.positions) > npts:
        st.session_state.positions = st.session_state.positions[:npts]

    for i in range(npts):
        st.session_state.positions[i] = st.sidebar.number_input(
            f"Position {i+1} ({unit})",
            value=float(st.session_state.positions[i]),
            step=1.0 if unit == "mm" else 0.001,
            format="%.3f" if unit == "m" else "%.0f"
        )
    positions = st.session_state.positions

    st.sidebar.subheader("Motion Params")
    acc_time = st.sidebar.number_input("Acceleration time (s)", 0.01, 60.0, 0.5, 0.01)
    dec_time = st.sidebar.number_input("Deceleration time (s)", 0.01, 60.0, 0.5, 0.01)
    use_vmax = st.sidebar.checkbox("Limit Vmax", value=True)
    vmax = st.sidebar.number_input(f"Vmax ({unit}/s)", 0.1, 1e9, 100.0, 0.1) if use_vmax else None

    st.sidebar.subheader("Physics (for Power/Energy)")
    mass_kg = st.sidebar.number_input("Load mass (kg)", 0.0, 10000.0, 5.0, 0.1)
    coulomb_fric_N = st.sidebar.number_input("Coulomb friction (N)", 0.0, 10000.0, 0.0, 0.1)
    visc_fric = st.sidebar.number_input(f"Viscous friction (N per {unit}/s)", 0.0, 1000.0, 0.0, 0.01)

    st.sidebar.subheader("Overlay Controls")
    if st.sidebar.button("➕ Add Profile"):
        if "profiles" not in st.session_state:
            st.session_state.profiles = []
        result = calc_profile(positions, acc_time, dec_time, vmax)
        if result is not None:
            t, p, v, a, cd, markers, seg_ends = result
            st.session_state.profiles.append({
                "positions": positions.copy(),
                "acc": acc_time, "dec": dec_time, "vmax": vmax,
                "t": t, "pos": p, "vel": v, "acc": a,
                "cumdist": cd, "markers": markers, "seg_ends": seg_ends,
                "unit": unit, "unit_to_m": unit_to_m,
                "mass_kg": mass_kg, "coulomb": coulomb_fric_N, "visc": visc_fric
            })

    if st.sidebar.button("🗑️ Clear Profiles"):
        st.session_state.pop("profiles", None)

    if "profiles" not in st.session_state or len(st.session_state.profiles) == 0:
        st.info("👈 Define waypoints & params, then click **Add Profile** to plot.")
        return

    # =========================
    # Metrics (TOP)
    # =========================
    last = st.session_state.profiles[-1]
    t, v, a, cd = last["t"], last["vel"], last["acc"], last["cumdist"]

    # Compute power/energy for metrics
    P_last, E_last, Epos_last = estimate_power(
        a, v, unit_to_m=last["unit_to_m"], mass_kg=last["mass_kg"],
        coulomb_fric_N=last["coulomb"], visc_fric_N_per_unit_s=last["visc"]
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total time", f"{t[-1]:.2f} s")
    col2.metric("Max |v|", f"{np.max(np.abs(v)):.2f} {unit}/s")
    col3.metric("Max |a|", f"{np.max(np.abs(a)):.2f} {unit}/s²")
    col4.metric("Total distance", f"{cd[-1]:.2f} {unit}")
    col5.metric("Energy (+)", f"{Epos_last[-1]:.2f} J")

    st.markdown("---")

    # =========================
    # Plots
    # =========================
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(f"Position s(t) [{unit}]",
                        f"Velocity v(t) [{unit}/s]",
                        f"Acceleration a(t) [{unit}/s²]",
                        f"Cumulative Distance [{unit}]",
                        "Power (W)")
    )
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf"]

    for i, prof in enumerate(st.session_state.profiles):
        t, p, v, a, cd = prof["t"], prof["pos"], prof["vel"], prof["acc"], prof["cumdist"]
        color = colors[i % len(colors)]
        vmax_label = f"{prof['vmax']:.3f}" if isinstance(prof['vmax'], (int, float)) else "auto"
        label = f"Acc={prof['acc']}s, Dec={prof['dec']}s, Vmax={vmax_label} {unit}/s"

        fig.add_trace(go.Scatter(x=t, y=p, name=f"Pos {label}", line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=v, name=f"Vel {label}", line=dict(color=color)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=a, name=f"Acc {label}", line=dict(color=color)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=cd, name=f"Cum {label}", line=dict(color=color)), row=4, col=1)

        for m in prof["markers"]:
            fig.add_vrect(x0=m["t_acc_end"]-prof["acc"], x1=m["t_acc_end"],
                          fillcolor="lightgreen", opacity=0.18, line_width=0, row=2, col=1)
            fig.add_vrect(x0=m["t_acc_end"], x1=m["t_const_end"],
                          fillcolor="lightblue", opacity=0.18, line_width=0, row=2, col=1)
            fig.add_vrect(x0=m["t_const_end"], x1=m["t_dec_end"],
                          fillcolor="salmon", opacity=0.18, line_width=0, row=2, col=1)

        y_vals = np.interp(prof["seg_ends"], t, cd)
        fig.add_trace(go.Scatter(x=prof["seg_ends"], y=y_vals, mode="markers",
                                 marker=dict(symbol="x", size=8), showlegend=False), row=4, col=1)

        P, E, Epos = estimate_power(a, v, unit_to_m=prof["unit_to_m"],
                                    mass_kg=prof["mass_kg"], coulomb_fric_N=prof["coulomb"],
                                    visc_fric_N_per_unit_s=prof["visc"])
        fig.add_trace(go.Scatter(x=t, y=P, name=f"Power {i+1}", line=dict(color=color)), row=5, col=1)

    fig.update_layout(height=1100, title="Motion Profiles", title_x=0.5, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Time (s)", row=5, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Playback (interactive)
    # =========================
    st.subheader("▶️ Playback")
    play_idx = st.selectbox("Profile for playback",
                            options=list(range(1, len(st.session_state.profiles)+1)),
                            index=len(st.session_state.profiles)-1) - 1
    pprof = st.session_state.profiles[play_idx]
    tb, pb = pprof["t"], pprof["pos"]

    if "play_time" not in st.session_state:
        st.session_state.play_time = 0.0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "play_speed" not in st.session_state:
        st.session_state.play_speed = 1.0

    play_cols = st.columns([1,1,2,2,2])
    with play_cols[0]:
        if st.button("▶️ Play" if not st.session_state.is_playing else "⏸ Pause"):
            st.session_state.is_playing = not st.session_state.is_playing
    with play_cols[1]:
        if st.button("⏮ Reset"):
            st.session_state.play_time = 0.0
    with play_cols[2]:
        st.session_state.play_speed = st.slider("Speed ×", 0.25, 4.0, st.session_state.play_speed, 0.25)
    with play_cols[3]:
        st.session_state.play_time = st.slider("Playback time (s)", 0.0, float(tb[-1]),
                                               float(st.session_state.play_time), 0.01)
    with play_cols[4]:
        st.caption("Tip: Use >1× to fast-forward")

    y_curr = np.interp(st.session_state.play_time, tb, pb)
    pb_fig = go.Figure()
    pb_fig.add_trace(go.Scatter(x=tb, y=pb, mode="lines", name="Position"))
    pb_fig.add_trace(go.Scatter(x=[st.session_state.play_time], y=[y_curr],
                                mode="markers", marker=dict(size=12, symbol="circle-open"),
                                name="Marker"))
    pb_fig.update_layout(height=300, title=f"Playback — Profile {play_idx+1}", showlegend=False)
    pb_fig.update_xaxes(title="Time (s)")
    pb_fig.update_yaxes(title=f"Position ({unit})")
    st.plotly_chart(pb_fig, use_container_width=True)

    if st.session_state.is_playing:
        next_time = st.session_state.play_time + 0.03 * st.session_state.play_speed
        if next_time > tb[-1]:
            next_time = tb[-1]
            st.session_state.is_playing = False
        st.session_state.play_time = float(next_time)
        time.sleep(0.03)
        st.rerun()

    # =========================
    # Data & Download
    # =========================
    st.subheader("📋 Data (last profile)")
    df = pd.DataFrame({
        "Time (s)": t,
        f"Position ({unit})": last["pos"],
        f"Velocity ({unit}/s)": last["vel"],
        f"Acceleration ({unit}/s²)": last["acc"],
        f"Cumulative Distance ({unit})": last["cumdist"],
        "Power (W)": P_last,
        "Energy (+) J": Epos_last,
        "Energy (signed) J": E_last
    })
    st.dataframe(df.iloc[::10], use_container_width=True, height=260)

    st.subheader("📥 Download")
    st.download_button("Download CSV (last profile)", df.to_csv(index=False),
                       file_name="motion_profile.csv", mime="text/csv")

    # =========================
    # Sidebar Footer Branding
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.image("https://www.dafra.si/design/images/logo.png", use_column_width=True)
    st.sidebar.markdown(
        """
        <div style="font-size:13px; color:gray; text-align:center; margin-top:10px;">
            Made by <b>Dejan Rožič</b><br>for <b>Dafra d.o.o.</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Stick to bottom
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] img {
            margin-top: 50px;
        }
        [data-testid="stSidebar"] div.block-container {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
