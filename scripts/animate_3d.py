import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("Loading trajectory.csv...")
df = pd.read_csv("../data/trajectory.csv")
print(f"Data loaded. Shape: {df.shape}")

# Ensure columns exist
if not all(c in df.columns for c in ['frame', 'time', 'id', 'x', 'y', 'z', 'diameter']):
    raise ValueError("Missing required columns in trajectory.csv.")

# Filter only active particles (optional, but requested in README)
if 'active' in df.columns:
    df = df[df['active'] == 1]

frames = sorted(df['frame'].unique())
print(f"Found {len(frames)} frames to animate.")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scale diameter to point size (can adjust multiplier as needed)
df['marker_size'] = np.clip(df['diameter'] * 1e5, 1, 50) 

# Calculate absolute max limits for symmetrical plotting
max_dist = max(df['x'].abs().max(), df['z'].abs().max())
max_height = df['y'].max() * 1.05

def init():
    ax.clear()
    return []

# Function to update each frame
def update(frame_idx):
    ax.clear()

    # Get data for current frame
    frame_df = df[df['frame'] == frame_idx]

    # Plot (y in the simulation is height, so map it to z in matplotlib)
    sc = ax.scatter(
        frame_df['x'], 
        frame_df['z'], 
        frame_df['y'], 
        s=frame_df['marker_size'],
        c=np.log10(frame_df['diameter']),
        cmap='viridis',
        alpha=0.6,
        marker='o'
    )

    # Draw a ground plane (floor) grid
    xx, zz = np.meshgrid(np.linspace(-max_dist, max_dist, 10), np.linspace(-max_dist, max_dist, 10))
    yy = np.zeros_like(xx)
    ax.plot_wireframe(xx, zz, yy, color='gray', alpha=0.3)

    # Need constant limits across all frames for a stable video
    ax.set_xlim([-max_dist, max_dist])
    ax.set_ylim([-max_dist, max_dist])
    ax.set_zlim([0, max_height])
    
    # Axes Labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Height Y (m)')
    
    # Set the frame time text
    if not frame_df.empty:
        t = frame_df['time'].iloc[0]
        ax.set_title(f"Lunar Regolith Simulation\\nTime: {t:.2f} s")
    
    return [sc]

# Set up the animation
print("Generating animation (this may take a while)...")
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=100)

writer = animation.PillowWriter(fps=10, metadata=dict(artist='Simulation Engine'))
ani.save("../figures/trajectory_3d.gif", writer=writer)

print("Saved animation to ../figures/trajectory_3d.gif.")
