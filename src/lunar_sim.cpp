#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

// =====================================================================
// Physical constants — host & device visible
// =====================================================================
const double RHO_PARTICLE      = 2700.0;
const double EJECTION_RADIUS   = 6.0;
const double SNAPSHOT_INTERVAL = 0.1;   // seconds between trajectory snapshots
const int    MAX_TRACERS       = 10000; // max particles tracked in trajectory output

#pragma omp declare target

const double G_LUNAR  = 1.63;
const double DT       = 0.01;
const double RHO_G0   = 1.0;
const double R_REF    = 1.0;
const double V_GAS    = 2000.0;
const double PI       = 3.14159265358979323846;
const double COR      = 0.4;
const double FRICTION = 0.8;
const double V_STOP   = 0.05;
const double MU_GAS   = 1.5e-5;
const double V_ESCAPE = 2376.0;  // lunar escape velocity (m/s)

// Grid constants — must be device-visible for kernel use
const double COLLISION_DOMAIN = 20.0;   // half-width of domain (m)
const double GRID_SIZE        = 0.05;   // cell side length (m)
const int    GRID_W           = 800;    // (2 * COLLISION_DOMAIN) / GRID_SIZE
const int    GRID_H           = 400;    // COLLISION_DOMAIN / GRID_SIZE
const int    NUM_CELLS        = GRID_W * GRID_H;  // 320 000

#pragma omp end declare target

// =====================================================================
// Device-callable utilities
// =====================================================================
#pragma omp declare target

inline int get_grid_index(double x, double y)
{
    if (x < -COLLISION_DOMAIN || x >= COLLISION_DOMAIN ||
        y >= COLLISION_DOMAIN || y < 0.0)
        return -1;
    int ix = (int)((x + COLLISION_DOMAIN) / GRID_SIZE);
    int iy = (int)(y / GRID_SIZE);
    if (ix < 0)       ix = 0;
    if (ix >= GRID_W) ix = GRID_W - 1;
    if (iy < 0)       iy = 0;
    if (iy >= GRID_H) iy = GRID_H - 1;
    return iy * GRID_W + ix;
}

void get_acceleration_vals(double xi, double yi, double vxi, double vyi,
                           double m,  double diam,
                           double &ax, double &ay)
{
    double r = sqrt(xi*xi + yi*yi);
    if (r < 0.1) r = 0.1;

    double rho_eff = RHO_G0 * pow(R_REF / r, 2.0);
    double vgx     = V_GAS * (xi / r);
    double vgy     = V_GAS * (yi / r);
    double vrx     = vgx - vxi;
    double vry     = vgy - vyi;
    double vr_mag  = sqrt(vrx*vrx + vry*vry);

    double area   = PI * (diam * 0.5) * (diam * 0.5);
    double re     = (rho_eff * (vr_mag + 1e-9) * diam) / MU_GAS;
    double cd     = 24.0/re + 6.0/(1.0 + sqrt(re)) + 0.4;
    double f_drag = 0.5 * rho_eff * vr_mag * vr_mag * cd * area;
    double f_lift = 0.20 * f_drag;
    double inv_vr = 1.0 / (vr_mag + 1e-9);

    ax = f_drag * (vrx * inv_vr) / m;
    ay = (f_drag * (vry * inv_vr) + f_lift - m * G_LUNAR) / m;
}

// RK4 adaptive-substep integrator for a single particle (SoA index i).
void update_state(int i,
                  double *__restrict__ px,  double *__restrict__ py,
                  double *__restrict__ pvx, double *__restrict__ pvy,
                  const double *__restrict__ pmass,
                  const double *__restrict__ pdiam,
                  int    *__restrict__ pactive,
                  int    *__restrict__ pimp,
                  double *__restrict__ ppeak)
{
    double t_acc = 0.0;
    while (t_acc < DT) {
        double ax, ay;
        get_acceleration_vals(px[i], py[i], pvx[i], pvy[i],
                              pmass[i], pdiam[i], ax, ay);

        double a_mag = sqrt(ax*ax + ay*ay);
        double ldt   = DT - t_acc;
        if (a_mag > 0.0) {
            double safe = 100.0 / a_mag;
            if (safe < ldt) ldt = safe;
        }

        double k1vx = pvx[i], k1vy = pvy[i], k1ax = ax, k1ay = ay;

        double k2ax, k2ay;
        get_acceleration_vals(px[i]  + k1vx*ldt*0.5, py[i]  + k1vy*ldt*0.5,
                              pvx[i] + k1ax*ldt*0.5, pvy[i] + k1ay*ldt*0.5,
                              pmass[i], pdiam[i], k2ax, k2ay);
        double k2vx = pvx[i] + k1ax*ldt*0.5;
        double k2vy = pvy[i] + k1ay*ldt*0.5;

        double k3ax, k3ay;
        get_acceleration_vals(px[i]  + k2vx*ldt*0.5, py[i]  + k2vy*ldt*0.5,
                              pvx[i] + k2ax*ldt*0.5, pvy[i] + k2ay*ldt*0.5,
                              pmass[i], pdiam[i], k3ax, k3ay);
        double k3vx = pvx[i] + k2ax*ldt*0.5;
        double k3vy = pvy[i] + k2ay*ldt*0.5;

        double k4ax, k4ay;
        get_acceleration_vals(px[i]  + k3vx*ldt, py[i]  + k3vy*ldt,
                              pvx[i] + k3ax*ldt, pvy[i] + k3ay*ldt,
                              pmass[i], pdiam[i], k4ax, k4ay);
        double k4vx = pvx[i] + k3ax*ldt;
        double k4vy = pvy[i] + k3ay*ldt;

        px[i]  += (ldt/6.0)*(k1vx + 2.0*k2vx + 2.0*k3vx + k4vx);
        py[i]  += (ldt/6.0)*(k1vy + 2.0*k2vy + 2.0*k3vy + k4vy);
        pvx[i] += (ldt/6.0)*(k1ax + 2.0*k2ax + 2.0*k3ax + k4ax);
        pvy[i] += (ldt/6.0)*(k1ay + 2.0*k2ay + 2.0*k3ay + k4ay);

        if (py[i] > ppeak[i]) ppeak[i] = py[i];

        if (py[i] <= 0.0) {
            py[i]  = 0.001;
            pvy[i] = COR * fabs(pvy[i]);   // always bounce upward
            pvx[i] *= FRICTION;
            pimp[i] = 1;
            if (sqrt(pvx[i]*pvx[i] + pvy[i]*pvy[i]) < V_STOP) {
                pactive[i] = 0;
                pvx[i] = 0.0; pvy[i] = 0.0; py[i] = 0.0;
                break;
            }
        }
        t_acc += ldt;
    }
}

#pragma omp end declare target

// =====================================================================
// main
// =====================================================================
int main(int argc, char *argv[])
{
    int N = 5000;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N < 1) N = 1;

    const int MAX_COLLISIONS = N * 8;

    // --- SoA particle arrays ---
    int    *pid      = new int   [N];
    double *px       = new double[N];
    double *py       = new double[N];
    double *pvx      = new double[N];
    double *pvy      = new double[N];
    double *pdiam    = new double[N];
    double *pmass    = new double[N];
    double *pt_start = new double[N];
    double *ppeak    = new double[N];
    int    *pactive       = new int   [N];
    int    *pimp          = new int   [N];
    int    *pescaped      = new int   [N];
    double *pescape_speed = new double[N];
    double *pescape_time  = new double[N];
    double *ptheta        = new double[N]; // azimuthal angle for 3D reconstruction (host-only)

    // --- Tracer particles for trajectory output ---
    int n_tracers = std::min(N, MAX_TRACERS);
    int *tracer_ids = new int[n_tracers];

    // --- Grid arrays (counting-sort / prefix-sum) ---
    int *grid_count  = new int[NUM_CELLS];
    int *grid_offset = new int[NUM_CELLS + 1];
    int *grid_flat   = new int[N];

    // --- Collision pair arrays ---
    int *col_a     = new int[MAX_COLLISIONS];
    int *col_b     = new int[MAX_COLLISIONS];
    int *col_count = new int[1];
    col_count[0]   = 0;

    // --- Impulse / positional-correction accumulation buffers ---
    double *dvx   = new double[N];
    double *dvy   = new double[N];
    double *dposx = new double[N];
    double *dposy = new double[N];

    // --- Initialise particles ---
    // Distribute spawn positions along an arc at r = EJECTION_RADIUS so
    // dense packing at a single point is avoided (critical at large N).
    std::mt19937 gen(42);
    std::lognormal_distribution<double>  size_dist(-10.0, 2.0);
    std::uniform_real_distribution<double> t_dist(0.0, 5.0);
    std::uniform_real_distribution<double> angle_dist(-PI * 0.45, PI * 0.45);
    std::uniform_real_distribution<double> jitter(-0.05, 0.05);
    std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * PI);

    for (int i = 0; i < N; ++i) {
        double d = size_dist(gen);
        if (d < 1e-6) d = 1e-6;
        if (d > 0.01)  d = 0.01;

        double angle = angle_dist(gen);
        double rx    = EJECTION_RADIUS * std::cos(angle) + jitter(gen);
        double ry    = 0.005 + std::abs(jitter(gen)) * 0.1;

        pid[i]      = i;
        pdiam[i]    = d;
        pmass[i]    = RHO_PARTICLE * (4.0/3.0) * PI * std::pow(d * 0.5, 3.0);
        px[i]       = rx;
        py[i]       = ry < 0.001 ? 0.001 : ry;
        pvx[i]      = 0.0;
        pvy[i]      = 0.0;
        pt_start[i] = t_dist(gen);
        ppeak[i]    = 0.0;
        pactive[i]       = 0;
        pimp[i]          = 0;
        pescaped[i]      = 0;
        pescape_speed[i] = 0.0;
        pescape_time[i]  = 0.0;
        ptheta[i]        = theta_dist(gen);
    }

    // --- Select tracer particles (Fisher-Yates partial shuffle) ---
    {
        std::vector<int> all_ids(N);
        for (int i = 0; i < N; ++i) all_ids[i] = i;
        for (int i = 0; i < n_tracers; ++i) {
            std::uniform_int_distribution<int> pick(i, N - 1);
            std::swap(all_ids[i], all_ids[pick(gen)]);
        }
        std::copy(all_ids.begin(), all_ids.begin() + n_tracers, tracer_ids);
        std::sort(tracer_ids, tracer_ids + n_tracers);
    }

    std::cout << "Starting GPU-Accelerated Regolith Simulation (N=" << N
              << ", tracers=" << n_tracers << ")\n";

    double       current_time      = 0.0;
    const double max_time          = 30.0;
    int          global_stopped    = 0;
    double       last_print_time   = -1.0;
    double       last_snapshot_time = -SNAPSHOT_INTERVAL; // triggers frame 0 immediately
    int          snapshot_count    = 0;

    // --- Open trajectory file for 3D visualisation ---
    std::FILE *trajf = std::fopen("trajectory.csv", "w");
    if (trajf) {
        std::setvbuf(trajf, nullptr, _IOFBF, 1 << 16);
        std::fprintf(trajf, "frame,time,id,x,y,z,vx,vy,vz,diameter,active\n");
    } else {
        std::cerr << "Warning: cannot open trajectory.csv for writing.\n";
    }

    // Map everything to the device once for the entire simulation.
    // Arrays modified on device are tofrom; read-only ones are to; purely
    // intermediate GPU buffers are alloc (no host transfer cost).
    #pragma omp target data \
        map(tofrom: px[0:N], py[0:N], pvx[0:N], pvy[0:N])    \
        map(tofrom: pactive[0:N], pimp[0:N], ppeak[0:N])                    \
        map(tofrom: pescaped[0:N], pescape_speed[0:N], pescape_time[0:N]) \
        map(to:     pdiam[0:N], pmass[0:N], pt_start[0:N])                \
        map(alloc:  grid_count[0:NUM_CELLS],                    \
                    grid_offset[0:NUM_CELLS+1],                 \
                    grid_flat[0:N])                             \
        map(alloc:  col_a[0:MAX_COLLISIONS],                    \
                    col_b[0:MAX_COLLISIONS],                    \
                    col_count[0:1])                             \
        map(alloc:  dvx[0:N], dvy[0:N], dposx[0:N], dposy[0:N])
    {
        // Initialise device-only scalar on the GPU
        #pragma omp target
        col_count[0] = 0;

        int global_escaped = 0;

        while (current_time < max_time && global_stopped < N) {

            // =============================================================
            // Step 1 — Activate newly eligible particles; RK4 integrate
            // =============================================================
            int currently_active = 0;
            int loop_stopped     = 0;
            int loop_escaped     = 0;

            #pragma omp target teams distribute parallel for \
                reduction(+: currently_active, loop_stopped, loop_escaped)
            for (int i = 0; i < N; ++i) {
                if (!pactive[i] && !pimp[i] && !pescaped[i] && current_time >= pt_start[i])
                    pactive[i] = 1;

                if (pactive[i]) {
                    update_state(i, px, py, pvx, pvy,
                                 pmass, pdiam, pactive, pimp, ppeak);

                    if (pactive[i]) {
                        double spd_sq = pvx[i]*pvx[i] + pvy[i]*pvy[i];
                        if (spd_sq >= V_ESCAPE * V_ESCAPE) {
                            pescaped[i]      = 1;
                            pescape_speed[i] = sqrt(spd_sq);
                            pescape_time[i]  = current_time;
                            pactive[i]       = 0;
                        }
                    }

                    currently_active++;
                    if (!pactive[i] && !pescaped[i]) loop_stopped++;
                    if (pescaped[i]) loop_escaped++;
                } else if (pescaped[i]) {
                    loop_escaped++;
                } else if (pimp[i]) {
                    loop_stopped++;
                }
            }
            global_stopped = loop_stopped;
            global_escaped = loop_escaped;

            // =============================================================
            // Step 2 — Build grid via counting sort + exclusive prefix sum
            //
            //   Pass 1 : count particles per cell (parallel, atomic)
            //   Pass 2 : exclusive prefix sum → grid_offset[]  (serial)
            //   Pass 3 : scatter indices into grid_flat[]  (parallel, atomic)
            // =============================================================

            // Pass 1a — zero cell counts
            #pragma omp target teams distribute parallel for
            for (int c = 0; c < NUM_CELLS; ++c)
                grid_count[c] = 0;

            // Pass 1b — accumulate counts
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < N; ++i) {
                if (!pactive[i]) continue;
                int cell = get_grid_index(px[i], py[i]);
                if (cell >= 0) {
                    #pragma omp atomic update
                    grid_count[cell]++;
                }
            }

            // Pass 2 — exclusive prefix sum (serial, single device thread)
            // Simultaneously resets grid_count[] to 0 for reuse as a write
            // cursor in Pass 3.
            #pragma omp target
            {
                int run = 0;
                for (int c = 0; c <= NUM_CELLS; ++c) {
                    grid_offset[c] = run;
                    if (c < NUM_CELLS) {
                        run += grid_count[c];
                        grid_count[c] = 0;
                    }
                }
            }

            // Pass 3 — scatter particle indices
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < N; ++i) {
                if (!pactive[i]) continue;
                int cell = get_grid_index(px[i], py[i]);
                if (cell >= 0) {
                    int slot;
                    #pragma omp atomic capture
                    slot = grid_count[cell]++;
                    grid_flat[grid_offset[cell] + slot] = i;
                }
            }

            // =============================================================
            // Step 3 — Detect collision pairs
            //
            // Detection is a read-only snapshot of positions.  Pairs that
            // are overlapping AND approaching are stored in col_a/col_b.
            // =============================================================
            #pragma omp target
            col_count[0] = 0;

            #pragma omp target teams distribute parallel for
            for (int cell = 0; cell < NUM_CELLS; ++cell) {
                int cnt  = grid_offset[cell + 1] - grid_offset[cell];
                if (cnt < 2) continue;
                int base = grid_offset[cell];

                for (int ii = 0; ii < cnt; ++ii) {
                    for (int jj = ii + 1; jj < cnt; ++jj) {
                        int ai = grid_flat[base + ii];
                        int bi = grid_flat[base + jj];

                        double dx  = px[bi] - px[ai];
                        double dy  = py[bi] - py[ai];
                        double dsq = dx*dx + dy*dy;
                        double md  = (pdiam[ai] + pdiam[bi]) * 0.5;
                        if (dsq >= md * md) continue;

                        double dv = sqrt(dsq);
                        if (dv == 0.0) continue;

                        double nx  = dx / dv, ny = dy / dv;
                        double rvx = pvx[bi] - pvx[ai];
                        double rvy = pvy[bi] - pvy[ai];
                        if (rvx*nx + rvy*ny >= 0.0) continue; // separating

                        int slot;
                        #pragma omp atomic capture
                        slot = col_count[0]++;
                        if (slot < MAX_COLLISIONS) {
                            col_a[slot] = ai;
                            col_b[slot] = bi;
                        }
                    }
                }
            }

            // Bring collision count to host to bound the resolution pass.
            // This is a 4-byte transfer and is negligible vs. the kernels.
            #pragma omp target update from(col_count[0:1])
            int n_col = col_count[0];
            if (n_col > MAX_COLLISIONS) n_col = MAX_COLLISIONS;

            if (n_col > 0) {
                // =========================================================
                // Step 4 — Zero per-particle accumulation buffers
                // =========================================================
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < N; ++i) {
                    dvx[i] = 0.0; dvy[i] = 0.0;
                    dposx[i] = 0.0; dposy[i] = 0.0;
                }

                // =========================================================
                // Step 5 — Accumulate impulses & positional corrections
                //
                // Each thread works on a distinct pair (no two threads share
                // the same pair).  Multiple pairs can involve the same
                // particle, so atomic updates to the accumulation buffers
                // are required.  Because reads and writes target separate
                // buffers, the reads from px/pvx are always from the
                // consistent post-integration snapshot.
                // =========================================================
                #pragma omp target teams distribute parallel for
                for (int k = 0; k < n_col; ++k) {
                    int ai = col_a[k], bi = col_b[k];

                    double dx  = px[bi] - px[ai];
                    double dy  = py[bi] - py[ai];
                    double dv  = sqrt(dx*dx + dy*dy);
                    if (dv == 0.0) continue;

                    double nx  = dx / dv, ny = dy / dv;
                    double rvx = pvx[bi] - pvx[ai];
                    double rvy = pvy[bi] - pvy[ai];
                    double vn  = rvx*nx + rvy*ny;
                    if (vn >= 0.0) continue;

                    double ma = pmass[ai], mb = pmass[bi];
                    double j  = -(1.0 + COR) * vn / (1.0/ma + 1.0/mb);
                    double jx = j * nx, jy = j * ny;

                    #pragma omp atomic update
                    dvx[ai]   -= jx / ma;
                    #pragma omp atomic update
                    dvy[ai]   -= jy / ma;
                    #pragma omp atomic update
                    dvx[bi]   += jx / mb;
                    #pragma omp atomic update
                    dvy[bi]   += jy / mb;

                    // Separate overlapping particles equally
                    double ov = (pdiam[ai] + pdiam[bi]) * 0.5 - dv;
                    double mx = 0.5 * ov * nx;
                    double my = 0.5 * ov * ny;

                    #pragma omp atomic update
                    dposx[ai] -= mx;
                    #pragma omp atomic update
                    dposy[ai] -= my;
                    #pragma omp atomic update
                    dposx[bi] += mx;
                    #pragma omp atomic update
                    dposy[bi] += my;
                }

                // =========================================================
                // Step 6 — Apply accumulated deltas; clamp to ground plane
                // =========================================================
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < N; ++i) {
                    if (!pactive[i]) continue;
                    pvx[i] += dvx[i];
                    pvy[i] += dvy[i];
                    px[i]  += dposx[i];
                    py[i]  += dposy[i];
                    if (py[i] < 0.001) py[i] = 0.001;
                }
            }

            current_time += DT;

            // =============================================================
            // Trajectory snapshot — pull tracer positions to host at interval
            // =============================================================
            if (trajf && current_time - last_snapshot_time >= SNAPSHOT_INTERVAL) {
                #pragma omp target update from(px[0:N], py[0:N], pvx[0:N], pvy[0:N], pactive[0:N])

                int frame = snapshot_count++;
                for (int t = 0; t < n_tracers; ++t) {
                    int i  = tracer_ids[t];
                    double ct = std::cos(ptheta[i]);
                    double st = std::sin(ptheta[i]);
                    std::fprintf(trajf,
                        "%d,%.4f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%d\n",
                        frame, current_time, pid[i],
                        px[i] * ct, py[i], px[i] * st,
                        pvx[i] * ct, pvy[i], pvx[i] * st,
                        pdiam[i], pactive[i]);
                }
                last_snapshot_time = current_time;
            }

            if (currently_active == 0 && current_time > 6.0) break;

            if (current_time - last_print_time >= 1.0) {
                last_print_time = current_time;
                int pct = (int)(100.0 * current_time / max_time);
                std::cout << "t=" << current_time
                          << "s  active=" << currently_active
                          << "  stopped=" << global_stopped
                          << "  escaped=" << global_escaped
                          << "  collisions=" << n_col
                          << "  (" << pct << "%)" << std::endl;
            }
        }
    } // end target data — tofrom arrays are copied back to host here

    if (trajf) {
        std::fclose(trajf);
        std::cout << "Trajectory written to trajectory.csv ("
                  << snapshot_count << " frames, " << n_tracers << " tracers)\n";
    }

    // =====================================================================
    // Buffered CSV output — avoids per-line syscall overhead at 1M particles
    // =====================================================================
    std::FILE *outf = std::fopen("results.csv", "w");
    if (!outf) {
        std::cerr << "Error: cannot open results.csv for writing.\n";
    } else {
        std::setvbuf(outf, nullptr, _IOFBF, 1 << 16); // 64 KB I/O buffer
        std::fprintf(outf, "id,diameter,mass,final_x,peak_height,t_start,escaped,escape_speed,escape_time,theta\n");
        for (int i = 0; i < N; ++i) {
            std::fprintf(outf, "%d,%.6e,%.6e,%.6f,%.6f,%.4f,%d,%.6e,%.4f,%.6f\n",
                         pid[i], pdiam[i], pmass[i],
                         px[i], ppeak[i], pt_start[i],
                         pescaped[i], pescape_speed[i], pescape_time[i],
                         ptheta[i]);
        }
        std::fclose(outf);
    }

    std::cout << "Complete. Results written to results.csv (N=" << N << ")\n";

    // =====================================================================
    // Escaped-particle summary
    // =====================================================================
    {
        int    n_esc      = 0;
        double spd_sum    = 0.0, spd_min = 1e30, spd_max = 0.0;
        double diam_sum   = 0.0, diam_min = 1e30, diam_max = 0.0;
        double mass_sum   = 0.0, mass_min = 1e30, mass_max = 0.0;

        for (int i = 0; i < N; ++i) {
            if (!pescaped[i]) continue;
            ++n_esc;
            spd_sum  += pescape_speed[i];
            spd_min   = std::min(spd_min,  pescape_speed[i]);
            spd_max   = std::max(spd_max,  pescape_speed[i]);
            diam_sum += pdiam[i];
            diam_min  = std::min(diam_min, pdiam[i]);
            diam_max  = std::max(diam_max, pdiam[i]);
            mass_sum += pmass[i];
            mass_min  = std::min(mass_min, pmass[i]);
            mass_max  = std::max(mass_max, pmass[i]);
        }

        std::cout << "\n=== Escaped Particles Summary ===\n";
        std::cout << "  Total escaped : " << n_esc << " / " << N << "\n";
        if (n_esc > 0) {
            std::cout << "  Escape speed (m/s) : "
                      << "mean=" << spd_sum/n_esc
                      << "  min=" << spd_min
                      << "  max=" << spd_max << "\n";
            std::cout << "  Diameter (m)       : "
                      << "mean=" << diam_sum/n_esc
                      << "  min=" << diam_min
                      << "  max=" << diam_max << "\n";
            std::cout << "  Mass (kg)          : "
                      << "mean=" << mass_sum/n_esc
                      << "  min=" << mass_min
                      << "  max=" << mass_max << "\n";
        }
        std::cout << "=================================\n";
    }

    // --- Cleanup ---
    delete[] pid;
    delete[] px;    delete[] py;
    delete[] pvx;   delete[] pvy;
    delete[] pdiam; delete[] pmass;
    delete[] pt_start; delete[] ppeak;
    delete[] pactive; delete[] pimp;
    delete[] pescaped; delete[] pescape_speed; delete[] pescape_time;
    delete[] ptheta;
    delete[] tracer_ids;

    delete[] grid_count; delete[] grid_offset; delete[] grid_flat;

    delete[] col_a; delete[] col_b; delete[] col_count;

    delete[] dvx; delete[] dvy; delete[] dposx; delete[] dposy;

    return 0;
}
