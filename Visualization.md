# Post-Processing and Visualization Guide

This document describes how to analyze the simulation output (`results.csv`) and generate the six required figures for the COSC 4397 Houston to the Moon project deliverables.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Loading the Data](#loading-the-data)
3. [Figure Specifications](#figure-specifications)
   - [Figure 1: Particle Size Distribution](#figure-1-particle-size-distribution)
   - [Figure 2: Landing Map](#figure-2-landing-map)
   - [Figure 3: Local Distribution Scale](#figure-3-local-distribution-scale)
   - [Figure 4: Long-Tail Distribution Scale](#figure-4-long-tail-distribution-scale)
   - [Figure 5: Temporal Accumulation Curve](#figure-5-temporal-accumulation-curve)
   - [Figure 6: Vertical Envelope Analysis](#figure-6-vertical-envelope-analysis)
4. [Data Dictionary](#data-dictionary)
5. [Interpreting the Results](#interpreting-the-results)
6. [Common Issues](#common-issues)

---

## Environment Setup

The visualization pipeline requires Python 3.8+ with the following packages:

```bash
pip install pandas matplotlib numpy
```

All figures can be generated from a single Python script or Jupyter notebook. The examples below use `matplotlib` for plotting and `pandas` for data manipulation.

---

## Loading the Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results.csv")
print(df.shape)       # (N, 6)
print(df.describe())  # Sanity-check ranges
```

Expected columns:

| Column | Type | Unit | Description |
|---|---|---|---|
| `id` | int | - | Unique particle identifier |
| `diameter` | float | m | Particle diameter |
| `mass` | float | kg | Particle mass (spherical, 2700 kg/m^3) |
| `final_x` | float | m | Final radial position from lander origin |
| `peak_height` | float | m | Maximum altitude achieved during flight |
| `t_start` | float | s | Staggered ejection time |

At 1 million particles, the CSV is approximately 80-100 MB. Pandas handles this without issue on any modern machine.

---

## Figure Specifications

### Figure 1: Particle Size Distribution

**Purpose:** Validate that the stochastic diameter generator produces the expected log-normal distribution with hard clipping at the physical boundaries.

**Construction:**

```python
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
plt.savefig("fig1_size_distribution.png", dpi=200)
```

**What to verify:**

- The histogram should be right-skewed on a logarithmic x-axis.
- No bars should appear below 1e-6 m or above 1e-2 m.
- The mode should be near `exp(mu) = exp(-10) ~ 4.5e-5 m` (45 um).

---

### Figure 2: Landing Map

**Purpose:** Visualize the spatial distribution of final resting positions, showing how particle mass correlates with travel distance.

**Construction:**

```python
fig, ax = plt.subplots(figsize=(14, 6))

# Add random vertical jitter for visual density (the simulation is 2D radial)
jitter_y = np.random.uniform(-500, 500, len(df))

scatter = ax.scatter(
    df["final_x"],
    jitter_y,
    c=np.log10(df["mass"]),
    cmap="gray_r",
    s=0.3,
    alpha=0.5,
    rasterized=True    # keeps file size manageable at 1M points
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("log10(mass) [kg]")
ax.set_xlabel("Final Radial Distance from Lander (m)")
ax.set_ylabel("Lateral Jitter (visual only)")
ax.set_title("Landing Map — Particle Final Positions")
plt.tight_layout()
plt.savefig("fig2_landing_map.png", dpi=200)
```

**What to verify:**

- Lighter particles (darker points on the `gray_r` colormap) should cluster at larger distances.
- Heavier particles (lighter points) should be concentrated closer to the lander.
- The spread should span from ~1 km to 60+ km depending on simulation parameters.

**Tip for large N:** Use `rasterized=True` on the scatter plot. Without it, saving 1M vector points as PDF/SVG produces files > 500 MB.

---

### Figure 3: Local Distribution Scale

**Purpose:** Analyze the particle landing density in the immediate vicinity of the lander, where surface erosion is most severe.

**Construction:**

```python
fig, ax = plt.subplots(figsize=(10, 6))

local = df[df["final_x"] <= 5000]  # Within 5 km of the lander
ax.hist(local["final_x"], bins=100, edgecolor="black", alpha=0.7)
ax.set_xlabel("Final Radial Distance (m)")
ax.set_ylabel("Particle Count")
ax.set_title("Landing Distribution — Local Scale (0 - 5 km)")
plt.tight_layout()
plt.savefig("fig3_local_distribution.png", dpi=200)
```

**What to verify:**

- A concentration peak near the ejection radius (~6 m outward) for heavy particles.
- A decline in density with distance as the plume weakens.
- This figure characterizes the primary erosion transport zone.

---

### Figure 4: Long-Tail Distribution Scale

**Purpose:** Demonstrate the extended travel range of microscopic particles in a vacuum environment and validate integrator stability over large distances.

**Construction:**

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["final_x"], bins=np.logspace(0, np.log10(3e6), 100),
        edgecolor="black", alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("Final Radial Distance (m)")
ax.set_ylabel("Particle Count")
ax.set_title("Landing Distribution — Long Tail (Log Scale, up to 3000 km)")
plt.tight_layout()
plt.savefig("fig4_long_tail_distribution.png", dpi=200)
```

**What to verify:**

- The distribution should span several orders of magnitude (meters to hundreds of kilometers).
- Micron-scale dust particles can travel thousands of kilometers in vacuum under the V_GAS = 2000 m/s plume.
- No unphysical spikes or gaps, which would indicate integrator instability.
- The tail validates that RK4 with adaptive substepping remains stable over long integration times.

---

### Figure 5: Temporal Accumulation Curve

**Purpose:** Show the ejection timeline — how particles are injected into the simulation over time — and confirm the staggered activation model.

**Construction:**

```python
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
plt.savefig("fig5_temporal_accumulation.png", dpi=200)
```

**What to verify:**

- A steady, approximately linear rise from t=0 to t=5 s (the uniform activation window).
- A flat plateau after t=5 s — no new particles are injected.
- The total count at the plateau should equal N.

---

### Figure 6: Vertical Envelope Analysis

**Purpose:** Map the vertical hazard zone by showing the maximum, minimum, and mean peak heights as a function of radial distance.

**Construction:**

```python
fig, ax = plt.subplots(figsize=(12, 6))

# Bin particles by final radial distance
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
plt.savefig("fig6_envelope_analysis.png", dpi=200)
```

**What to verify:**

- The shaded envelope should widen at intermediate distances where particles follow diverse ballistic arcs.
- Very heavy nearby particles (short final_x) should have low peak heights.
- Very light distant particles should have high peak heights (several km or more).
- The envelope defines the vertical "danger zone" for any hardware orbiting or hovering near the landing site.

---

## Data Dictionary

For reference, every field in `results.csv` is derived from the simulation as follows:

| Field | Source | Notes |
|---|---|---|
| `id` | Sequential assignment at initialization | Range: [0, N-1] |
| `diameter` | `log-normal(-10, 2)`, clipped to [1e-6, 1e-2] | Meters. Mode ~ 45 um. |
| `mass` | `2700 * (4/3) * pi * (d/2)^3` | Kilograms. Ranges from ~10^-12 to ~10^-5. |
| `final_x` | Last `px[i]` value when simulation ends | Meters from lander origin. Positive = outward. |
| `peak_height` | Running maximum of `py[i]` during integration | Meters above ground. |
| `t_start` | `uniform(0, 5)` | Seconds. Time at which particle enters the simulation. |

---

## Interpreting the Results

### Expected Ranges (N=5000, default parameters)

| Metric | Typical Value |
|---|---|
| Average final_x | ~50 km |
| Max final_x | ~60 km |
| Min final_x | ~4 km |
| Average peak_height | ~10 km |
| Max peak_height | ~21 km |
| Diameter mode | ~45 um |
| Particle mass range | 10^-12 to 10^-5 kg |

### Physical Interpretation

- **Small particles (< 10 um):** Dominated by drag. Accelerated rapidly to near-gas velocity, travel furthest (50-60 km), reach moderate heights. Represent the fine dust hazard.
- **Medium particles (10-100 um):** Balanced regime. Moderate travel (10-40 km) with the highest peak altitudes due to favorable lift-to-weight ratio.
- **Large particles (> 100 um):** Gravity-dominated. Travel shorter distances (< 10 km) on low arcs. Represent the ballistic ejecta hazard for nearby structures.

---

## Common Issues

### "My scatter plot file is huge"

At 1M particles, vector formats (SVG, PDF) will embed one path element per point. Use `rasterized=True` on scatter plots and save as PNG, or use `plt.savefig(..., dpi=150)` to limit file size.

### "All final_x values seem very large"

This is expected. V_GAS = 2000 m/s in vacuum means particles are accelerated to km/s velocities. In a real mission, the gas density decays much faster and the plume duration is shorter. The current parameters model a sustained worst-case scenario.

### "peak_height is zero for some particles"

These are particles that were never activated (t_start > simulation end time) or that stopped immediately after their first bounce. Check the `t_start` column — if it's close to the max simulation time and the run was short, the particle may not have been injected.

### "How do I filter by particle size class?"

```python
fine_dust = df[df["diameter"] < 10e-6]         # < 10 um
sand      = df[(df["diameter"] >= 10e-6) & (df["diameter"] < 1e-3)]  # 10 um - 1 mm
gravel    = df[df["diameter"] >= 1e-3]          # > 1 mm
```

### "Can I animate the particle trajectories?"

The current simulation only outputs the final state. To capture trajectories, you would need to modify `lunar_sim.cpp` to periodically copy particle positions from device to host and append them to a time-series output file. This would add significant I/O overhead and is not recommended at 1M particles without binary output.
