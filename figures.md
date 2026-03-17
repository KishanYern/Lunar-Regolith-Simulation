# Simulation Visualization Outputs

This document contains the visual data generated from the Lunar Regolith Simulation using the parameters defined in `Visualization.md`.

## 1. Particle Size Distribution
![Size Distribution](figures/fig1_size_distribution.png)  
Shows the initial generated particle sizes adhering to a log-normal distribution, artificially clamped down to boundaries defined in the simulation initialization step.

## 2. Landing Map
![Landing Map](figures/fig2_landing_map.png)  
Shows the final spatial positioning scaled horizontally via distance from the landing pad. Particles are styled by a logarithmic interpretation of their mass, allowing easy contrast of distance traversal vs mass.

## 3. Local Distribution Scale
![Local Distribution](figures/fig3_local_distribution.png)  
Detailed view focusing tightly on the immediate vicinity spanning `0m` to `5000m`.

## 4. Long-Tail Distribution Scale
![Long Tail Distribution](figures/fig4_long_tail_distribution.png)  
Highlights the extensive distances traversed by micron-scale lightweight dust using a logarithmic distance graph extending outwards entirely up to ~3000 kilometers!

## 5. Temporal Accumulation Curve
![Temporal Accumulation Curve](figures/fig5_temporal_accumulation.png)  
Depicts the stochastic start rate of the particles, displaying an active accumulation plateauing completely sharply upon hitting the predefined `5` second injection cutoff threshold.

## 6. Vertical Envelope Analysis
![Envelope Analysis](figures/fig6_envelope_analysis.png)  
Depicts the full vertical hazard height (max/mean/min) against horizontal travel distance to demonstrate where in the spatial volume orbiting payloads might be hit.
