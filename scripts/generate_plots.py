import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Loading results.csv...")
df = pd.read_csv("../data/results.csv")
print(f"Data loaded. Shape: {df.shape}")

print("Generating Figure 1: Particle Size Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["diameter"], bins=np.logspace(-6, -2, 80), edgecolor="black", alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("Particle Diameter (m)")
ax.set_ylabel("Count")
ax.set_title("Particle Size Distribution (Log-Normal, Clipped)")
ax.axvline(1e-6, color="red", linestyle="--", label="Min clip (1 um)")
ax.axvline(1e-2, color="red", linestyle="--", label="Max clip (10 mm)")
ax.legend()
plt.tight_layout()
plt.savefig("../figures/fig1_size_distribution.png", dpi=200)
plt.close()

print("Generating Figure 2: Landing Map...")
fig, ax = plt.subplots(figsize=(14, 6))
jitter_y = np.random.uniform(-500, 500, len(df))
scatter = ax.scatter(
    df["final_x"],
    jitter_y,
    c=np.log10(df["mass"]),
    cmap="gray_r",
    s=0.3,
    alpha=0.5,
    rasterized=True
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("log10(mass) [kg]")
ax.set_xlabel("Final Radial Distance from Lander (m)")
ax.set_ylabel("Lateral Jitter (visual only)")
ax.set_title("Landing Map — Particle Final Positions")
plt.tight_layout()
plt.savefig("../figures/fig2_landing_map.png", dpi=200)
plt.close()

print("Generating Figure 3: Local Distribution Scale...")
fig, ax = plt.subplots(figsize=(10, 6))
local = df[df["final_x"] <= 5000]
ax.hist(local["final_x"], bins=100, edgecolor="black", alpha=0.7)
ax.set_xlabel("Final Radial Distance (m)")
ax.set_ylabel("Particle Count")
ax.set_title("Landing Distribution — Local Scale (0 - 5 km)")
plt.tight_layout()
plt.savefig("../figures/fig3_local_distribution.png", dpi=200)
plt.close()

print("Generating Figure 4: Long-Tail Distribution Scale...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["final_x"], bins=np.logspace(0, np.log10(3e6), 100),
        edgecolor="black", alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("Final Radial Distance (m)")
ax.set_ylabel("Particle Count")
ax.set_title("Landing Distribution — Long Tail (Log Scale, up to 3000 km)")
plt.tight_layout()
plt.savefig("../figures/fig4_long_tail_distribution.png", dpi=200)
plt.close()

print("Generating Figure 5: Temporal Accumulation Curve...")
fig, ax = plt.subplots(figsize=(10, 6))
sorted_times = np.sort(df["t_start"].values)
cumulative = np.arange(1, len(sorted_times) + 1)
ax.plot(sorted_times, cumulative, linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cumulative Particles Ejected")
ax.set_title("Temporal Accumulation of Ejected Particles")
ax.axvline(5.0, color="red", linestyle="--", alpha=0.5, label="End of ejection window")
ax.legend()
plt.tight_layout()
plt.savefig("../figures/fig5_temporal_accumulation.png", dpi=200)
plt.close()

print("Generating Figure 6: Vertical Envelope Analysis...")
fig, ax = plt.subplots(figsize=(12, 6))
df["x_bin"] = pd.cut(df["final_x"], bins=80)
grouped = df.groupby("x_bin", observed=True)["peak_height"]
stats = grouped.agg(["min", "mean", "max"]).dropna()
bin_centers = [interval.mid for interval in stats.index]
ax.fill_between(bin_centers, stats["min"], stats["max"],
                alpha=0.3, label="Min-Max Envelope")
ax.plot(bin_centers, stats["mean"], color="black", linewidth=1.5, label="Mean")
ax.plot(bin_centers, stats["max"], color="red", linewidth=0.8,
        linestyle="--", label="Max")
ax.plot(bin_centers, stats["min"], color="blue", linewidth=0.8,
        linestyle="--", label="Min")
ax.set_xlabel("Final Radial Distance (m)")
ax.set_ylabel("Peak Height (m)")
ax.set_title("Vertical Flight Envelope — Peak Height vs. Landing Distance")
ax.legend()
plt.tight_layout()
plt.savefig("../figures/fig6_envelope_analysis.png", dpi=200)
plt.close()

print("All 6 figures generated successfully.")
