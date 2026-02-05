import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

# === CONFIGURATION ===
csv_file = "obs_action_data.csv"  # Replace with your CSV file name
plot_velocity_vectors = False  # Set True to show Vx,Vy,Vz arrows

# === READ CSV DATA ===
cols = [
    "X","Y","Z",
    "Roll","Pitch","Yaw",
    "Vx","Vy","Vz",
    "Wx","Wy","Wz",
    "Avg_m1","Avg_m2","Avg_m3","Avg_m4",
    "Delta_m1","Delta_m2","Delta_m3","Delta_m4",
    "Act_m1","Act_m2","Act_m3","Act_m4"
]
df = pd.read_csv(csv_file, names=cols)

# Ensure all relevant columns are numeric
numeric_cols = ["X", "Y", "Z", "Vx", "Vy", "Vz"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Drop rows where position data is missing
df = df.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)

# === EXTRACT POSITION ===
x, y, z = -df["X"].values, -df["Y"].values, -df["Z"].values+1
t = np.arange(len(df))  # time index (can replace with actual time if available)

# === CREATE SEGMENTS FOR COLOR GRADIENT ===
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# === COLOR MAP SETUP ===
norm = Normalize(t.min(), t.max())
lc = Line3DCollection(segments, cmap='viridis', norm=norm)
lc.set_array(t)
lc.set_linewidth(2)

# === PLOT ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.add_collection(lc)

# Set limits to match data range
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())

# Optional velocity vectors
if plot_velocity_vectors:
    step = max(1, len(df)//20)
    ax.quiver(
        x[::step], y[::step], z[::step],
        df["Vx"][::step], df["Vy"][::step], df["Vz"][::step],
        length=0.001, normalize=True, color='red', label='Velocity Vectors'
    )

# === LABELS, COLORBAR, STYLE ===
ax.set_title("Real Trajectory (mm)", fontsize=15)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
plt.colorbar(lc, ax=ax, label="Time Step")
ax.grid(True)
plt.tight_layout()
plt.savefig(csv_file.replace(".csv", ".png"), dpi=300)
