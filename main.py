import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Helper math ----
def trapezoid_motion(distance, vmax, t_acc, t_dec, dt=0.005):
    a = vmax / t_acc
    b = vmax / t_dec

    d_acc = 0.5 * vmax * t_acc
    d_dec = 0.5 * vmax * t_dec
    d_min = d_acc + d_dec

    if distance < d_min:
        # triangular profile
        v_peak = np.sqrt((2 * distance * a * b) / (a + b))
        t_acc_eff = v_peak / a
        t_dec_eff = v_peak / b
        t_cruise = 0
        profile_type = "triangle"
    else:
        v_peak = vmax
        t_acc_eff = t_acc
        t_dec_eff = t_dec
        d_cruise = distance - d_min
        t_cruise = d_cruise / vmax
        profile_type = "trapezoid"

    t_total = t_acc_eff + t_cruise + t_dec_eff

    times = np.arange(0, t_total + dt, dt)
    velocities = []
    positions = []

    for t in times:
        if t <= t_acc_eff:
            v = a * t
            x = 0.5 * a * t ** 2
        elif t <= t_acc_eff + t_cruise:
            tau = t - t_acc_eff
            v = v_peak
            x = 0.5 * a * t_acc_eff ** 2 + v_peak * tau
        else:
            tau = t - (t_acc_eff + t_cruise)
            v = max(0, v_peak - b * tau)
            x = (0.5 * a * t_acc_eff ** 2 + v_peak * t_cruise +
                 v_peak * tau - 0.5 * b * tau ** 2)
        velocities.append(v)
        positions.append(x)

    # normalize final position to exactly distance
    scale = distance / positions[-1] if positions[-1] != 0 else 1
    positions = [p * scale for p in positions]

    data = pd.DataFrame({"Time [s]": times, "Velocity": velocities, "Position": positions})

    summary = {
        "Profile type": profile_type,
        "Distance": distance,
        "Requested Vmax": vmax,
        "Peak velocity": v_peak,
        "Accel (a)": a,
        "Decel (b)": b,
        "t_acc": t_acc_eff,
        "t_cruise": t_cruise,
        "t_dec": t_dec_eff,
        "Total time": t_total,
    }

    return data, summary

# ---- Streamlit UI ----
st.set_page_config(page_title="Trapezoidal Motion Profile", layout="wide")

st.title("Trapezoidal Motion Profile Designer")
st.write("Enter move parameters to compute trapezoidal or triangular motion profiles.")

col1, col2, col3 = st.columns(3)

with col1:
    distance = st.number_input("Move distance (m)", value=0.5, step=0.1, format="%.3f")
    vmax = st.number_input("Max speed (m/s)", value=0.8, step=0.1, format="%.3f")

with col2:
    t_acc = st.number_input("Accel time (s)", value=0.3, step=0.05, format="%.3f")
    t_dec = st.number_input("Decel time (s)", value=0.25, step=0.05, format="%.3f")

with col3:
    dt = st.number_input("Sampling dt (s)", value=0.005, step=0.001, min_value=0.001, max_value=0.05, format="%.3f")

# compute profile
data, summary = trapezoid_motion(distance, vmax, t_acc, t_dec, dt)

# show results
st.subheader("Results")
st.json(summary)

# plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Velocity vs Time")
    fig, ax = plt.subplots()
    ax.plot(data["Time [s]"], data["Velocity"], label="Velocity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Position vs Time")
    fig, ax = plt.subplots()
    ax.plot(data["Time [s]"], data["Position"], label="Position", color="orange")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position (m)")
    ax.grid(True)
    st.pyplot(fig)

# export CSV
st.subheader("Export Data")
st.download_button("Download CSV", data.to_csv(index=False).encode("utf-8"), "motion_profile.csv", "text/csv")