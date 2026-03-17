# Lunar Regolith Ejection Dynamics Simulator

A GPU-accelerated computational physics engine that models the trajectories of lunar regolith particles entrained by a supersonic rocket exhaust plume in a vacuum environment. Developed for the COSC 4397 Houston to the Moon project.

The simulator solves coupled equations of motion for up to **one million stochastically generated particles** using a fourth-order Runge-Kutta integration scheme, with discrete-element collision detection handled via spatial hashing on the GPU.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Building and Running](#building-and-running)
4. [Physical Model](#physical-model)
5. [Numerical Methods](#numerical-methods)
6. [GPU Architecture](#gpu-architecture)
7. [Simulation Parameters](#simulation-parameters)
8. [Output Format](#output-format)
9. [3D Trajectory Reconstruction](#3d-trajectory-reconstruction)
10. [Performance and Scaling](#performance-and-scaling)
11. [Frequently Asked Questions](#frequently-asked-questions)
12. [Project Structure](#project-structure)

---

## Quick Start

```bash
# Default run (5 000 particles)
./run_simulation.sh

# Custom particle count
./run_simulation.sh 100000

# Full-scale run (1 million particles)
./run_simulation.sh 1000000
```

Results are written to `results.csv` and `trajectory.csv` in the working directory.

---

## Prerequisites

| Requirement | Version Tested | Notes |
|---|---|---|
| Linux | Ubuntu 22.04+ | Any distro with ROCm support |
| AMD GPU | RDNA2 (gfx1030/1032) | RX 6600 XT or equivalent |
| ROCm | 6.3.1 | Provides `amdclang++` with OpenMP target offload |
| C++ Standard | C++17 | Required by the build script |

The build script expects ROCm installed at `/opt/rocm-6.3.1`. If your installation path differs, edit the `ROCM_PATH` variable in `run_simulation.sh`.

**CPU fallback:** If GPU compilation fails, you can compile for CPU-only execution with:

```bash
g++ -O3 -std=c++17 -fopenmp lunar_sim.cpp -o simulator_cpu
./simulator_cpu 5000
```

This runs the same code using OpenMP threads on the CPU instead of offloading to the GPU. It is functionally identical but significantly slower at large particle counts.

---

## Building and Running

The `run_simulation.sh` script handles compilation and execution in one step:

```bash
./run_simulation.sh [N]
```

- `N` is the number of particles (default: 5000).
- The script compiles `lunar_sim.cpp` with `-O3` optimization and OpenMP offload targeting `gfx1030`.
- A runtime ISA override (`HSA_OVERRIDE_GFX_VERSION=10.3.0`) maps gfx1030 to gfx1032 for RX 6600 XT hardware.
- On success, the compiled binary `simulator_gpu` is executed immediately with the specified `N`.

### What Happens During a Run

1. **Initialization** (host): Particles are generated with stochastic diameters and staggered start times. Initial positions are distributed along an arc at the ejection radius.
2. **Device mapping**: All particle arrays are transferred to GPU memory once.
3. **Main loop** (device): Each timestep performs integration, grid construction, collision detection, impulse resolution, and delta application — all as GPU kernels.
4. **Termination**: The loop exits when simulated time reaches 30 seconds, all particles have stopped, or no particles remain active past 6 seconds.
5. **Output** (host): Results are copied back from device memory and written to CSV.

Progress is printed every simulated second:

```
t=5.04s  active=5000  stopped=0  escaped=0  collisions=0  (16%)
```

---

## Physical Model

The simulation couples five physical subsystems to model regolith transport in a lunar landing scenario.

### 1. Particle Generation

Particle diameters are drawn from a **log-normal distribution** with parameters `mu = -10.0` and `sigma = 2.0`, then hard-clamped to the range [1 um, 10 mm]. This produces a right-skewed size distribution representative of lunar regolith.

Each particle's mass is derived from its diameter assuming a spherical shape and a bulk density of **2700 kg/m^3** (consistent with lunar basalt):

```
mass = rho * (4/3) * pi * (d/2)^3
```

### 2. Exhaust Plume Gas Dynamics

The rocket plume is modeled as a radially expanding supersonic gas field:

- **Gas velocity**: 2000 m/s directed radially outward from the lander origin. This represents the exhaust velocity of a bipropellant engine (e.g., Aerojet Rocketdyne R-4D class).
- **Gas density decay**: Follows an inverse-square law from a reference density of 1.0 kg/m^3 at a reference radius of 1.0 m:

```
rho_gas(r) = rho_0 * (R_ref / r)^2
```

This ensures the plume's entrainment force weakens with distance, matching the physical behavior of an underexpanded jet expanding into vacuum.

### 3. Aerodynamic Forces

**Drag** is computed using an empirical Reynolds-number-dependent drag coefficient suitable for spheres across the subsonic-to-supersonic regime:

```
C_d = 24/Re + 6/(1 + sqrt(Re)) + 0.4
```

where `Re = (rho_gas * |v_rel| * d) / mu_gas`, and `mu_gas = 1.5e-5 Pa*s`. The drag force is:

```
F_drag = 0.5 * rho_gas * |v_rel|^2 * C_d * A
```

where `A = pi * (d/2)^2` is the particle cross-sectional area.

**Lift** is modeled as a fixed fraction (20%) of the drag magnitude, applied vertically upward. This is a simplification that captures the net upward entrainment effect of the expanding plume without requiring full 3D flow field resolution.

### 4. Gravity and Ground Interaction

- **Lunar gravity**: 1.63 m/s^2 directed downward, applied continuously.
- **Ground bounce**: When a particle's y-coordinate drops to zero, its vertical velocity is reversed and scaled by a coefficient of restitution (COR = 0.4), and its horizontal velocity is scaled by a friction factor (0.8).
- **Stop condition**: If a particle's total speed drops below 0.05 m/s after a bounce, it is deactivated and its final position is recorded.

### 5. Particle-Particle Collisions

Collisions are resolved using an impulse-based model:

- Two particles collide when their center-to-center distance is less than the sum of their radii **and** they are approaching (relative velocity has a negative component along the collision normal).
- The impulse magnitude preserves momentum and applies the coefficient of restitution:

```
j = -(1 + COR) * v_normal / (1/m_a + 1/m_b)
```

- Overlapping particles are separated by displacing each by half the overlap distance along the collision normal.

---

## Numerical Methods

### Fourth-Order Runge-Kutta Integration (RK4)

The equations of motion form a coupled ODE system:

```
dx/dt = vx          dvx/dt = ax(x, y, vx, vy)
dy/dt = vy          dvy/dt = ay(x, y, vx, vy)
```

The classical RK4 scheme evaluates the acceleration function at four sample points per timestep to achieve fourth-order accuracy:

```
k1 = f(t_n, y_n)
k2 = f(t_n + dt/2, y_n + dt/2 * k1)
k3 = f(t_n + dt/2, y_n + dt/2 * k2)
k4 = f(t_n + dt,   y_n + dt * k3)

y_{n+1} = y_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
```

### Adaptive Substepping

Within each global timestep (DT = 0.01 s), the integrator subdivides into smaller substeps when the acceleration magnitude would cause a velocity change exceeding 100 m/s in a single substep. This prevents numerical instability for small, lightweight particles experiencing very high drag accelerations close to the plume origin.

### Collision Detection: Spatial Hashing with Counting Sort

The domain is partitioned into a uniform grid of 0.05 m cells spanning [-20 m, +20 m] horizontally and [0, 20 m] vertically (800 x 400 = 320,000 cells). Each timestep:

1. **Count**: Atomically count particles per cell.
2. **Prefix sum**: Compute an exclusive prefix sum over the counts to produce a CSR-style offset array.
3. **Scatter**: Write each particle's index into a flat array at the appropriate offset.

Collision detection then iterates over all pairs within each cell. This reduces the O(N^2) brute-force check to approximately O(N * k) where k is the average number of particles per cell.

### Race-Free Collision Resolution

Impulses are accumulated into per-particle buffers (`dvx`, `dvy`, `dposx`, `dposy`) during the resolution pass, then applied in a separate kernel. This separation ensures:

- All impulse computations read from the same consistent post-integration snapshot of positions and velocities.
- No two kernels simultaneously read and write the same particle's state.
- Atomic operations on the accumulation buffers handle the case where multiple pairs involve the same particle.

---

## GPU Architecture

### Data Layout: Structure of Arrays (SoA)

Particle data is stored as separate flat arrays per field rather than as an array of structs. This ensures that when a GPU wavefront (64 threads on RDNA2) accesses, e.g., the x-position of consecutive particles, the memory reads are contiguous and can be served in a single cache line fetch.

### OpenMP Target Offload

All compute kernels use `#pragma omp target teams distribute parallel for`, which maps to GPU thread blocks (workgroups) and threads (work-items). Device-callable functions (`get_acceleration_vals`, `update_state`, `get_grid_index`) are declared with `#pragma omp declare target`.

### Memory Mapping Strategy

| Map Type | Arrays | Rationale |
|---|---|---|
| `tofrom` | `px, py, pvx, pvy, pactive, pimp, ppeak` | Modified on device, needed on host for CSV output |
| `tofrom` | `pescaped, pescape_speed, pescape_time` | Escape tracking — written on device, read on host |
| `to` | `pdiam, pmass, pt_start` | Read-only on device |
| `alloc` | Grid arrays, collision arrays, accumulators | Temporary per-timestep; no host transfer needed |

All device memory is allocated once at the start of the simulation via `#pragma omp target data` and released when the block exits. Per-timestep host-device transfers are limited to a single 4-byte collision count read, plus periodic bulk transfers of position/velocity arrays for trajectory snapshots (every `SNAPSHOT_INTERVAL` seconds).

---

## Simulation Parameters

### Constants (compile-time, defined in `lunar_sim.cpp`)

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| Particle density | `RHO_PARTICLE` | 2700 | kg/m^3 | Bulk density of lunar basalt |
| Ejection radius | `EJECTION_RADIUS` | 6.0 | m | Radial distance of particle spawn arc |
| Snapshot interval | `SNAPSHOT_INTERVAL` | 0.1 | s | Time between trajectory frames |
| Max tracers | `MAX_TRACERS` | 10000 | - | Max particles written to trajectory |
| Lunar gravity | `G_LUNAR` | 1.63 | m/s^2 | Surface gravitational acceleration |
| Global timestep | `DT` | 0.01 | s | Outer integration step |
| Reference gas density | `RHO_G0` | 1.0 | kg/m^3 | Gas density at r = R_REF |
| Reference radius | `R_REF` | 1.0 | m | Radius for density normalization |
| Gas velocity | `V_GAS` | 2000 | m/s | Radial exhaust velocity |
| Coefficient of restitution | `COR` | 0.4 | - | Energy loss on bounce / collision |
| Surface friction | `FRICTION` | 0.8 | - | Tangential velocity retention on bounce |
| Stop velocity | `V_STOP` | 0.05 | m/s | Speed below which particle deactivates |
| Gas dynamic viscosity | `MU_GAS` | 1.5e-5 | Pa*s | For Reynolds number computation |
| Lunar escape velocity | `V_ESCAPE` | 2376 | m/s | Particles exceeding this speed are flagged as escaped |
| Collision domain half-width | `COLLISION_DOMAIN` | 20.0 | m | Spatial hash grid extent |
| Grid cell size | `GRID_SIZE` | 0.05 | m | Side length of each grid cell |
| Grid width | `GRID_W` | 800 | cells | Horizontal grid resolution |
| Grid height | `GRID_H` | 400 | cells | Vertical grid resolution |
| Total grid cells | `NUM_CELLS` | 320,000 | cells | GRID_W * GRID_H |

### Runtime Parameters

| Parameter | Default | Set Via | Description |
|---|---|---|---|
| Particle count (N) | 5000 | CLI argument | `./simulator_gpu 100000` |
| Max simulation time | 30 s | Hard-coded | Loop terminates at t = 30 s |
| Max collisions per step | N * 8 | Derived | Overflow is capped, not UB |

### Stochastic Initialization

| Distribution | Parameters | Range | Purpose |
|---|---|---|---|
| Diameter | Log-normal(mu=-10, sigma=2) | [1 um, 10 mm] | Realistic regolith size spread |
| Start time | Uniform(0, 5) | [0 s, 5 s] | Staggered activation |
| Spawn angle | Uniform(-81 deg, +81 deg) | 162 deg arc | Avoids degenerate same-point collisions |
| Position jitter | Uniform(-0.05, +0.05) | +/- 5 cm | Additional radial/vertical spread |

---

## Output Format

### `results.csv`

One row per particle, written after the simulation completes:

| Column | Type | Unit | Description |
|---|---|---|---|
| `id` | int | - | Unique particle identifier (0-indexed) |
| `diameter` | float (sci) | m | Particle diameter |
| `mass` | float (sci) | kg | Particle mass |
| `final_x` | float | m | Final radial position from origin |
| `peak_height` | float | m | Maximum altitude reached during flight |
| `t_start` | float | s | Time at which the particle was activated |
| `escaped` | int | - | 1 if particle exceeded lunar escape velocity, 0 otherwise |
| `escape_speed` | float (sci) | m/s | Speed at the moment of escape (0 if not escaped) |
| `escape_time` | float | s | Simulation time when escape was detected (0 if not escaped) |
| `theta` | float | rad | Azimuthal angle assigned for 3D reconstruction (see below) |

Example rows:

```csv
id,diameter,mass,final_x,peak_height,t_start,escaped,escape_speed,escape_time,theta
0,1.510524e-05,4.872424e-12,58770.172286,5478.336700,0.4999,0,0.000000e+00,0.0000,3.721054
1,1.272784e-04,2.914918e-09,51826.447607,12562.420994,3.2544,0,0.000000e+00,0.0000,1.048273
```

### `trajectory.csv`

Time-series 3D positions and velocities for a random subsample of tracer particles, written during the simulation at regular intervals. Used for animated 3D visualisation.

| Column | Type | Unit | Description |
|---|---|---|---|
| `frame` | int | - | Sequential frame number (0, 1, 2, ...) |
| `time` | float | s | Simulation time |
| `id` | int | - | Particle ID |
| `x` | float | m | 3D x position |
| `y` | float | m | 3D y position (height above ground) |
| `z` | float | m | 3D z position |
| `vx` | float | m/s | 3D x velocity |
| `vy` | float | m/s | 3D y velocity |
| `vz` | float | m/s | 3D z velocity |
| `diameter` | float (sci) | m | Particle diameter |
| `active` | int | - | 1 if particle is still being integrated |

By default, up to 10,000 tracer particles are sampled and frames are written every 0.1 s. For a 30 s simulation this produces ~300 frames. These values are controlled by the `MAX_TRACERS` and `SNAPSHOT_INTERVAL` constants.

### Console Output

Progress is printed every simulated second:

```
t=5.04s  active=5000  stopped=0  escaped=0  collisions=0  (16%)
```

- **active**: Number of particles currently being integrated.
- **stopped**: Cumulative count of particles that have come to rest.
- **escaped**: Cumulative count of particles that exceeded escape velocity.
- **collisions**: Number of collision pairs detected this timestep.
- **%**: Percentage of the 30 s max simulation time elapsed.

After the simulation completes, a summary of escaped particles is printed with count, and min/mean/max for escape speed, diameter, and mass.

---

## 3D Trajectory Reconstruction

### Animation Preview
![3D Trajectory](figures/trajectory_3d.gif)

The physics engine is 2D: each particle has a radial position `px` (horizontal distance from the nozzle axis) and a height `py` (vertical distance above the ground plane). Because the rocket exhaust plume is axially symmetric, the 2D cross-section can be revolved around the vertical axis to produce a physically meaningful 3D scene.

### Coordinate conversion

Each particle is assigned a random azimuthal angle `theta` in [0, 2*pi) at initialisation. This angle is fixed for the lifetime of the particle. The 2D simulation coordinates `(px, py)` are converted to 3D Cartesian coordinates by revolving around the y-axis:

```
x_3d = px * cos(theta)
y_3d = py
z_3d = px * sin(theta)
```

The same rotation is applied to velocities:

```
vx_3d = pvx * cos(theta)
vy_3d = pvy
vz_3d = pvx * sin(theta)
```

Here `px` is the radial distance from the nozzle axis (which can be positive or negative in the 2D simulation, corresponding to opposite sides of the plume), `py` is the height above the surface, and `theta` sets the particle's position around the plume's circumference.

This produces a 3D cone/fan of debris that is uniformly distributed in azimuth, consistent with the assumption that the plume has no preferred horizontal direction.

### Where the conversion is applied

- **`data/trajectory.csv`** contains pre-computed 3D coordinates. The `x`, `y`, `z`, `vx`, `vy`, `vz` columns are already in 3D Cartesian space — no post-processing needed.
- **`data/results.csv`** stores the raw 2D final state plus the `theta` column. To reconstruct 3D positions from the final state, apply the formulas above using `final_x` as `px`, `peak_height` or 0 as `py`, and the `theta` value from the same row.

### Visualisation tools

The trajectory data can be loaded into:

- **Python (Plotly / PyVista)**: Read `trajectory.csv` with pandas, group by `frame`, and render as an animated 3D scatter plot. Use the `diameter` column to scale point sizes.
- **Blender**: Write a Python script that reads frames and creates keyframed icosphere instances per particle. The `diameter` column maps to object scale.
- **ParaView**: Import the CSV as a table, apply a "Table to Points" filter (columns `x`, `y`, `z`), and use the frame/time columns for animation.

---

## Performance and Scaling

### Benchmarks (AMD RX 6600 XT, ROCm 6.3.1)

| N | Wall-Clock Time | GPU VRAM Usage |
|---|---|---|
| 5,000 | ~3.5 min | < 100 MB |
| 100,000 | ~30 min (est.) | ~200 MB |
| 1,000,000 | 3-6 hours (est.) | ~180 MB |

The integration pass scales linearly with N. The collision pass scales with N times the average particles per cell. At very high N, the serial prefix-sum (320K iterations on a single GPU thread) becomes a secondary bottleneck; this could be replaced with a parallel work-efficient scan if needed.

### Memory Budget at 1M Particles

| Allocation | Size |
|---|---|
| 8 double particle arrays | 64 MB |
| 3 int particle arrays | 12 MB |
| 4 double accumulation buffers | 32 MB |
| 2 collision pair arrays (N * 8 pairs) | 64 MB |
| Grid arrays (320K cells + N flat) | ~6 MB |
| **Total** | **~178 MB** |

This fits comfortably within the 8 GB VRAM of an RX 6600 XT.

---

## Frequently Asked Questions

### Why are there zero collisions in my run?

At moderate particle counts (5K-50K), the arc-based spawn distribution spreads particles across many grid cells. By the time the gas accelerates them, they occupy different spatial regions and rarely share the same 0.05 m cell. Collisions become significant at higher densities (>100K particles) or with a tighter spawn arc.

### Why do all particles stay active for the full 30 seconds?

The continuous gas plume (V_GAS = 2000 m/s) exerts drag that keeps particles above the stop threshold (0.05 m/s) for the entire simulation. Particles only deactivate when they bounce to rest, which requires the drag to weaken enough for gravity and friction to dominate. With the current plume parameters, most particles travel tens of kilometers before this happens.

### Can I run this without a GPU?

Yes. Compile with `g++ -O3 -std=c++17 -fopenmp src/lunar_sim.cpp -o simulator_cpu`. OpenMP target regions fall back to host execution. Performance will be significantly lower (10-50x slower depending on CPU core count and N), but the physics are identical.

### How do I change the plume parameters?

All physical constants are defined at the top of `src/lunar_sim.cpp`. Key tuning knobs:

- `V_GAS`: Exhaust velocity. Lower values reduce particle travel distance.
- `RHO_G0` / `R_REF`: Control how quickly plume density falls off with distance.
- `COR`: Coefficient of restitution. Lower = more energy lost per bounce.
- `DT`: Timestep. Smaller values improve accuracy at the cost of runtime.

Recompile after changing any constant.

### What happens if the collision pair buffer overflows?

`MAX_COLLISIONS` is set to `N * 8`. If more pairs are detected in a single timestep, the counter continues to increment but pairs beyond the limit are not stored. The simulation continues with the pairs that were captured. This is a graceful degradation, not a crash. You can increase the multiplier from 8 to 16 or higher if needed.

### Why is the grid only 20 m wide if particles travel kilometers?

The spatial hash grid covers the immediate vicinity of the lander where particle density is highest and collisions are most likely. Once particles leave this 40 m x 20 m zone, they are in free ballistic flight — still integrated by RK4 but no longer checked for collisions. This is physically appropriate: at large distances the gas density is negligible and particles are widely separated.

### Is the simulation deterministic?

Yes, when run on the same hardware with the same N. The random seed is fixed at 42, and the OpenMP reduction operations produce deterministic results for a given thread decomposition. Running on different GPU architectures or with different thread counts may produce slightly different floating-point reduction orderings.

### How does the adaptive substepping work?

Within each 0.01 s global timestep, the integrator computes the acceleration magnitude. If it would cause a velocity change exceeding 100 m/s in the remaining substep, the substep is shortened to `100 / |a|` seconds. This prevents numerical explosion for tiny particles (diameter ~ 1 um) near the plume origin where drag forces are extreme.

---

## Project Structure

```
Regolith_Simulation/
  src/
    lunar_sim.cpp         # Complete simulation source (single file)
  scripts/
    generate_plots.py     # Reproduces analysis PNGs
    animate_3d.py         # Reproduces animated 3D tracking GIF
  data/
    results.csv           # Final destination output file upon script run
    trajectory.csv        # Snapshot file constructed via execution output
  figures/
    ...                   # Precompiled representations generated from python analysis
  README.md               # This file
  figures.md              # Gallery of analysis visualization outputs
  Visualization.md        # Post-processing requirements
  run_simulation.sh       # Build + run script (compile -> execution -> data move)
```
