import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Multi-Segment Motion Profile")

st.write("Add waypoints with distance, velocity, accel time, and decel time. The system calculates segment motion automatically.")

# --- Session state for points ---
if "points" not in st.session_state:
    st.session_state.points = [
        {"distance": 0.0, "velocity": 0.0, "t_acc": 0.3, "t_dec": 0.3},
        {"distance": 1.0, "velocity": 0.8, "t_acc": 0.3, "t_dec": 0.3},
    ]

# --- Add/Remove points ---
col1, col2 = st.columns([1,1])
with col1:
    if st.button("➕ Add Point"):
        st.session_state.points.append({"distance": 0.0, "velocity": 0.0, "t_acc": 0.3, "t_dec": 0.3})
with col2:
    if st.button("➖ Remove Last Point") and len(st.session_state.points) > 2:
        st.session_state.points.pop()

# --- Editable table ---
df = pd.DataFrame(st.session_state.points)
edited = st.data_editor(df, num_rows="dynamic")
st.session_state.points = edited.to_dict("records")

# --- Compute profile ---
def compute_segments(points, dt=0.01):
    times, velocities, positions = [], [], []
    t = 0
    s = 0

    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        v1, v2 = p1["velocity"], p2["velocity"]
        d = p2["distance"] - p1["distance"]
        t_acc = max(1e-6, p2.get("t_acc", 0.3))
        t_dec = max(1e-6, p2.get("t_dec", 0.3))

        if d <= 0:
            continue

        # Calculate accel/decel rates
        a = (v2 - v1) / t_acc if v2 > v1 else 0
        b = (v1 - v2) / t_dec if v1 > v2 else 0

        # Distance covered during accel/decel
        if v2 > v1:  # accelerating
            d_acc = (v1 + v2) / 2 * t_acc
        else:        # decelerating
            d_acc = (v1 + v2) / 2 * t_dec

        d_cruise = max(0, d - d_acc)
        t_cruise = d_cruise / v2 if v2 > 0 else 0

        # Generate samples
        ts = np.arange(0, t_acc + t_cruise + t_dec, dt)
        for tau in ts:
            if tau <= t_acc and v2 > v1:  # accel phase
                v = v1 + a * tau
            elif tau >= t_acc + t_cruise and v1 > v2:  # decel phase
                tau_dec = tau - (t_acc + t_cruise)
                v = max(v2, v1 - b * tau_dec)
            else:  # cruise or steady
                v = v2
            s += v * dt
            times.append(t + tau)
            velocities.append(v)
            positions.append(s)
        t += t_acc + t_cruise + t_dec

    return pd.DataFrame({"Time [s]": times, "Velocity": velocities, "Position": positions})

if len(st.session_state.points) >= 2:
    profile = compute_segments(st.session_state.points)
    
    st.subheader("Results")
    st.write(profile.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Velocity vs Time")
        fig, ax = plt.subplots()
        ax.plot(profile["Time [s]"], profile["Velocity"])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity")
        st.pyplot(fig)

    with col2:
        st.subheader("Position vs Time")
        fig, ax = plt.subplots()
        ax.plot(profile["Time [s]"], profile["Position"], color="orange")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position")
        st.pyplot(fig)

    st.download_button(
        "Download CSV", profile.to_csv(index=False).encode("utf-8"),
        "multi_profile.csv", "text/csv"
    )
