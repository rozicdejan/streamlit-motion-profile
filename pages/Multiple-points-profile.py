import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_motion_profile(positions, acc_time, dec_time, max_velocity=None):
    """
    Calculate motion profile for given positions with trapezoidal velocity profile
    """
    if len(positions) < 2:
        return None, None, None, None
    
    # Calculate segments
    segments = []
    total_time = 0
    
    for i in range(len(positions) - 1):
        start_pos = positions[i]
        end_pos = positions[i + 1]
        distance = abs(end_pos - start_pos)
        direction = 1 if end_pos > start_pos else -1
        
        if distance == 0:
            # No movement, just add a small time step
            segments.append({
                'start_pos': start_pos,
                'end_pos': end_pos,
                'distance': 0,
                'direction': direction,
                'acc_time': 0,
                'dec_time': 0,
                'const_time': 0.1,  # Small constant time for zero movement
                'max_vel': 0,
                'start_time': total_time,
                'end_time': total_time + 0.1
            })
            total_time += 0.1
            continue
        
        # Calculate maximum velocity if not provided
        # For trapezoidal profile: distance = 0.5 * max_vel * (acc_time + dec_time) + max_vel * const_time
        # Minimum time needed for acc + dec
        min_time = acc_time + dec_time
        min_distance = 0.5 * (acc_time + dec_time) * distance / min_time if min_time > 0 else distance
        
        if max_velocity is None or min_time == 0:
            # Calculate required max velocity
            if acc_time + dec_time > 0:
                max_vel = 2 * distance / (acc_time + dec_time)
                const_time = 0
            else:
                max_vel = distance  # Instantaneous movement
                const_time = 1
        else:
            max_vel = max_velocity
            # Calculate constant velocity time
            acc_dec_distance = 0.5 * max_vel * (acc_time + dec_time)
            if acc_dec_distance >= distance:
                # Triangular profile (no constant velocity phase)
                max_vel = np.sqrt(2 * distance / (acc_time + dec_time)) if (acc_time + dec_time) > 0 else distance
                const_time = 0
            else:
                const_time = (distance - acc_dec_distance) / max_vel
        
        segment_time = acc_time + const_time + dec_time
        
        segments.append({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'distance': distance,
            'direction': direction,
            'acc_time': acc_time,
            'dec_time': dec_time,
            'const_time': const_time,
            'max_vel': max_vel * direction,
            'start_time': total_time,
            'end_time': total_time + segment_time
        })
        
        total_time += segment_time
    
    # Generate time arrays and profiles
    dt = 0.01  # 10ms resolution
    time_array = np.arange(0, total_time + dt, dt)
    position_array = np.zeros_like(time_array)
    velocity_array = np.zeros_like(time_array)
    acceleration_array = np.zeros_like(time_array)
    
    for i, t in enumerate(time_array):
        # Find which segment we're in
        current_segment = None
        for seg in segments:
            if seg['start_time'] <= t <= seg['end_time']:
                current_segment = seg
                break
        
        if current_segment is None:
            # Use last position if beyond all segments
            position_array[i] = positions[-1]
            velocity_array[i] = 0
            acceleration_array[i] = 0
            continue
        
        # Time within current segment
        seg_time = t - current_segment['start_time']
        
        if current_segment['distance'] == 0:
            # No movement segment
            position_array[i] = current_segment['start_pos']
            velocity_array[i] = 0
            acceleration_array[i] = 0
        else:
            # Calculate position, velocity, acceleration for this segment
            acc_time = current_segment['acc_time']
            const_time = current_segment['const_time']
            dec_time = current_segment['dec_time']
            max_vel = current_segment['max_vel']
            start_pos = current_segment['start_pos']
            direction = current_segment['direction']
            
            if seg_time <= acc_time and acc_time > 0:
                # Acceleration phase
                acceleration = max_vel / acc_time
                velocity = acceleration * seg_time
                position = start_pos + 0.5 * acceleration * seg_time**2
            elif seg_time <= acc_time + const_time:
                # Constant velocity phase
                acceleration = 0
                velocity = max_vel
                position = start_pos + 0.5 * max_vel * acc_time + max_vel * (seg_time - acc_time)
            else:
                # Deceleration phase
                dec_seg_time = seg_time - acc_time - const_time
                acceleration = -max_vel / dec_time if dec_time > 0 else 0
                velocity = max_vel + acceleration * dec_seg_time
                const_distance = 0.5 * max_vel * acc_time + max_vel * const_time
                position = start_pos + const_distance + max_vel * dec_seg_time + 0.5 * acceleration * dec_seg_time**2
            
            position_array[i] = position
            velocity_array[i] = velocity
            acceleration_array[i] = acceleration
    
    return time_array, position_array, velocity_array, acceleration_array

def main():
    st.set_page_config(page_title="Motion Profile Calculator", layout="wide")
    
    st.title("🚀 Motion Profile Calculator")
    st.markdown("Calculate and visualize motion profiles with trapezoidal velocity curves")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")
    
    # Position input methods
    input_method = st.sidebar.radio("Input Method", ["Manual Entry", "Upload Table"])
    
    positions = []
    
    if input_method == "Manual Entry":
        st.sidebar.subheader("Position Waypoints")
        
        # Initialize session state for positions
        if 'positions' not in st.session_state:
            st.session_state.positions = [0, 500, 0, 560, 0, 620, 0]
        
        # Number of waypoints
        num_points = st.sidebar.number_input("Number of waypoints", min_value=2, max_value=20, value=len(st.session_state.positions))
        
        # Adjust list size
        if len(st.session_state.positions) != num_points:
            if len(st.session_state.positions) < num_points:
                st.session_state.positions.extend([0] * (num_points - len(st.session_state.positions)))
            else:
                st.session_state.positions = st.session_state.positions[:num_points]
        
        # Input fields for each position
        for i in range(num_points):
            st.session_state.positions[i] = st.sidebar.number_input(
                f"Position {i+1}", 
                value=float(st.session_state.positions[i]), 
                format="%.2f"
            )
        
        positions = st.session_state.positions
    
    else:
        st.sidebar.subheader("Upload Position Data")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'position' in df.columns:
                    positions = df['position'].tolist()
                elif len(df.columns) >= 1:
                    positions = df.iloc[:, 0].tolist()
                else:
                    st.sidebar.error("No valid position column found")
                    positions = [0, 500, 0]
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                positions = [0, 500, 0]
        else:
            positions = [0, 500, 0, 560, 0, 620, 0]  # Default
    
    # Motion parameters
    st.sidebar.subheader("Motion Parameters")
    acc_time = st.sidebar.number_input("Acceleration Time (s)", min_value=0.01, max_value=10.0, value=0.5, step=0.01)
    dec_time = st.sidebar.number_input("Deceleration Time (s)", min_value=0.01, max_value=10.0, value=0.5, step=0.01)
    
    use_max_vel = st.sidebar.checkbox("Set Maximum Velocity")
    max_velocity = None
    if use_max_vel:
        max_velocity = st.sidebar.number_input("Maximum Velocity (units/s)", min_value=1.0, value=100.0)
    
    # Calculate button
    if st.sidebar.button("🔄 Calculate Motion Profile", type="primary"):
        if len(positions) < 2:
            st.error("Please provide at least 2 position waypoints")
        else:
            # Calculate motion profile
            with st.spinner("Calculating motion profile..."):
                time_array, pos_array, vel_array, acc_array = calculate_motion_profile(
                    positions, acc_time, dec_time, max_velocity
                )
            
            if time_array is not None:
                # Store results in session state
                st.session_state.results = {
                    'time': time_array,
                    'position': pos_array,
                    'velocity': vel_array,
                    'acceleration': acc_array,
                    'positions': positions
                }
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Summary information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Time", f"{results['time'][-1]:.2f} s")
        with col2:
            st.metric("Max Velocity", f"{np.max(np.abs(results['velocity'])):.2f} units/s")
        with col3:
            st.metric("Max Acceleration", f"{np.max(np.abs(results['acceleration'])):.2f} units/s²")
        with col4:
            st.metric("Total Distance", f"{np.sum(np.abs(np.diff(results['positions']))):.2f} units")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Position s(t)', 'Velocity v(t)', 'Acceleration a(t)'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Position plot
        fig.add_trace(
            go.Scatter(
                x=results['time'], 
                y=results['position'],
                mode='lines',
                name='Position',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add waypoint markers
        waypoint_times = []
        segment_time = 0
        for i in range(len(results['positions'])):
            if i == 0:
                waypoint_times.append(0)
            else:
                # Find the time when we reach this position
                pos_target = results['positions'][i]
                idx = np.argmin(np.abs(results['position'] - pos_target))
                waypoint_times.append(results['time'][idx])
        
        fig.add_trace(
            go.Scatter(
                x=waypoint_times,
                y=results['positions'],
                mode='markers',
                name='Waypoints',
                marker=dict(color='red', size=8, symbol='diamond')
            ),
            row=1, col=1
        )
        
        # Velocity plot
        fig.add_trace(
            go.Scatter(
                x=results['time'], 
                y=results['velocity'],
                mode='lines',
                name='Velocity',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Acceleration plot
        fig.add_trace(
            go.Scatter(
                x=results['time'], 
                y=results['acceleration'],
                mode='lines',
                name='Acceleration',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Motion Profile Analysis",
            showlegend=True,
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Position (units)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (units/s)", row=2, col=1)
        fig.update_yaxes(title_text="Acceleration (units/s²)", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Data Table"):
            df_results = pd.DataFrame({
                'Time (s)': results['time'][::10],  # Sample every 10th point for display
                'Position (units)': results['position'][::10],
                'Velocity (units/s)': results['velocity'][::10],
                'Acceleration (units/s²)': results['acceleration'][::10]
            })
            st.dataframe(df_results, use_container_width=True)
        
        # Download button
        csv_data = pd.DataFrame({
            'Time': results['time'],
            'Position': results['position'],
            'Velocity': results['velocity'],
            'Acceleration': results['acceleration']
        })
        
        csv_string = csv_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_string,
            file_name="motion_profile.csv",
            mime="text/csv"
        )
    
    else:
        # Instructions
        st.info("👈 Enter your position waypoints and motion parameters in the sidebar, then click 'Calculate Motion Profile' to generate the analysis.")
        
        # Example
        st.subheader("📖 Example")
        st.markdown("""
        **Input positions:** 0, 500, 0, 560, 0, 620, 0
        
        This creates a motion profile that:
        1. Moves from 0 to 500 units
        2. Returns to 0
        3. Moves to 560 units
        4. Returns to 0
        5. Moves to 620 units
        6. Returns to 0
        
        Each movement follows a trapezoidal velocity profile with your specified acceleration and deceleration times.
        """)

if __name__ == "__main__":
    main()
