import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Custom Multi-Point Motion Profile with Diagnostics")

# Example points
points = [
    {"distance": 0, "velocity": 0, "t_acc": 0.3, "t_dec": 0},
    {"distance": 500, "velocity": 1200, "t_acc": 0.3, "t_dec": 0.3},
    {"distance": 0, "velocity": 1200, "t_acc": 0.3, "t_dec": 0.3},
    {"distance": 560, "velocity": 1200, "t_acc": 0.3, "t_dec": 0.3},
]

# --- Compute motion profile ---
def compute_segments(points, dt=0.001):
    times, velocities, positions = [], [], []
    t = 0
    s = points[0]["distance"]

    switch_points = []  # store phase markers

    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        v1, v2 = p1["velocity"], p2["velocity"]
        d1, d2 = p1["distance"], p2["distance"]
        d = d2 - d1
        t_acc = max(1e-6, p2.get("t_acc", 0.3))
        t_dec = max(1e-6, p2.get("t_dec", 0.3))

        if d == 0:
            continue

        # Compute accel/decel rates
        if v2 > v1:
            a = (v2 - v1) / t_acc
        elif v1 > v2:
            a = -(v1 - v2) / t_dec
        else:
            a = 0

        # Estimate total segment time
        avg_v = (v1 + v2) / 2 if (v1 != v2) else v2
        t_seg = abs(d) / max(avg_v, 1e-6)

        # Save switch points for markers
        switch_points.append((t, "start"))
        if v2 > v1:
            switch_points.append((t + t_acc, "end_acc"))
        if v1 > v2:
            switch_points.append((t + t_seg - t_dec, "start_dec"))
        switch_points.append((t + t_seg, "end"))

        # Sample motion
        ts = np.arange(0, t_seg, dt)
        for tau in ts:
            if tau <= t_acc and v2 > v1:  # accelerating
                v = v1 + (v2 - v1) * (tau / t_acc)
            elif tau >= t_seg - t_dec and v1 > v2:  # decelerating
                tau_dec = tau - (t_seg - t_dec)
                v = v1 - (v1 - v2) * (tau_dec / t_dec)
            else:
                v = v2
            s += np.sign(d) * v * dt
            times.append(t + tau)
            velocities.append(v)
            positions.append(s)

        t += t_seg

    df = pd.DataFrame({"Time [s]": times, "Velocity": velocities, "Position": positions})
    return df, switch_points

# Compute
profile, switches = compute_segments(points)

# Results
st.subheader("Motion Profile Data")
st.write(profile.head())

# --- Velocity vs Time with markers ---
st.subheader("Velocity vs Time with Phase Markers")
fig, ax = plt.subplots()
ax.plot(profile["Time [s]"], profile["Velocity"], label="Velocity")

for t_mark, label in switches:
    ax.axvline(t_mark, color="red", linestyle="--", alpha=0.6)
    ax.text(t_mark, max(profile["Velocity"])*0.9, label, rotation=90, fontsize=8)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Velocity (mm/s)")
ax.grid(True)
st.pyplot(fig)

# --- Position vs Time ---
st.subheader("Position vs Time")
fig, ax = plt.subplots()
ax.plot(profile["Time [s]"], profile["Position"], color="orange", label="Position")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position (mm)")
ax.grid(True)
st.pyplot(fig)

# --- Cumulative distance travelled ---
distance_travelled = np.sum(np.abs(np.diff(profile["Position"])))
st.subheader("Cumulative Distance")
st.write(f"**Total distance travelled (including back & forth): {distance_travelled:.2f} mm**")

# --- Velocity distribution (heatmap/histogram) ---
st.subheader("Velocity Distribution (Time Spent at Each Speed)")
fig, ax = plt.subplots()
ax.hist(profile["Velocity"], bins=30, color="purple", alpha=0.7)
ax.set_xlabel("Velocity (mm/s)")
ax.set_ylabel("Time Samples (≈ duration / dt)")
ax.grid(True)
st.pyplot(fig)

# Export CSV
st.download_button("Download CSV", profile.to_csv(index=False).encode("utf-8"),
                   "motion_profile.csv", "text/csv")
