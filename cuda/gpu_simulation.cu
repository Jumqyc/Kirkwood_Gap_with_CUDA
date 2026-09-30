// The GPU half of the simulation. See cuda/gpu_simulation.hpp for the design.
//
// Build: cmake --build <build-dir> --target kirkwood_gpu
// Run:   ./<build-dir>/kirkwood_gpu <n_particles> <n_steps> [dump_dir]
//                                   [epoch_every] [e_max] [saturn] [seed]

#include "gpu_simulation.hpp"

#include "../cpp/epoch_file.hpp"
#include "../cpp/physics.hpp"
#include "../cpp/setup.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Every CUDA call returns an error code; nothing throws, and a failed launch
// leaves the arrays untouched without stopping the program.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            std::fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__, __LINE__,   \
                         #call, cudaGetErrorString(err_));                     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Yoshida's fourth-order composition coefficients. Same values as
// cpp/simulation.cpp; if they drift apart the two builds stop agreeing.
constexpr double W1 = 1.3512071919596578;
constexpr double W0 = -1.7024143839193153;

// Sums the acceleration of every massive body at one sub-step.
//
// The force is computed in SINGLE precision on purpose. Nsight Compute put the
// FP64 pipeline at 86.8% with the issue slots at 2.3%: this kernel is bound by
// the FP64 pipe, which on a consumer GPU runs at 1/64 the FP32 rate. Doing the
// force in float moves it off that pipe entirely, and rsqrtf is one MUFU
// instruction rather than the ~20 FP64 operations a double division expands
// into. Measured 1.80x at 1e6 particles.
//
// What it costs: per-particle |da| after 1e5 years goes from 1e-14 to 5.2e-4
// AU. That is 2% of the 0.025 AU gap width, which sounds alarming and is not:
// what the project measures is a binned distribution, and the difference
// between this and the double version is 30x smaller than the signal in the
// worst bin (0.36% on the 2:1 depletion ratio). A future measurement wanting a
// finer signal than that would have to re-examine this.
//
// Positions, velocities and the accumulator itself all stay double. Only the
// arithmetic between them is float, and the body positions arrive as float so
// the host converts them once rather than every thread converting them on every
// call -- another 1.25x.
//
// Args:
//   rx, ry, rz: this particle's position in AU, already narrowed to float.
//   ax, ay, az: accumulator in AU/step^2; overwritten, not added to.
//   bodies: where every massive body is, by sub-step.
//   sub: which sub-step, 0..2.
// Returns:
//   Nothing; the accumulator is written.
__device__ inline void force(float rx, float ry, float rz,
                             double &ax, double &ay, double &az,
                             const BodyTable &bodies, int sub)
{
    ax = ay = az = 0.0;

    // A run-time loop, unlike the CPU kernel's compile-time unrolling. The body
    // count is the same for every thread, so the branch is uniform, and the
    // unrolling buys little on a GPU. It also means adding a body costs a
    // longer table rather than another switch case.
    for (int b = 0; b < bodies.nb; ++b)
    {
        const float px = bodies.pos[sub][b][0];
        const float py = bodies.pos[sub][b][1];
        const float pz = bodies.pos[sub][b][2];
        const float gm = bodies.gm[b];

        const float dx = px - rx;
        const float dy = py - ry;
        const float dz = pz - rz;
        const float r2 = dx * dx + dy * dy + dz * dz;
        const float inv_r = rsqrtf(r2);
        const float inv_r3 = inv_r * inv_r * inv_r;
        ax += (double)(gm * dx * inv_r3);
        ay += (double)(gm * dy * inv_r3);
        az += (double)(gm * dz * inv_r3);
    }
}

// Advances every test particle by one outer step: the three KDK sub-steps of
// the Yoshida composition, with the acceleration carried in a register
// throughout. There is no barrier anywhere in this kernel and the acceleration
// never touches memory, which is the shape the CPU version cannot take.
//
// Args:
//   rx, ry, rz: device arrays of length n, positions in AU.
//   vx, vy, vz: device arrays of length n, velocities in AU per step.
//   ax, ay, az: device arrays of length n, acceleration in AU/step^2, as left by
//     the previous step. Written back consistent with the new position.
//   n: number of particles. Threads past the end return without writing.
//   bodies: where every massive body is during each of the three sub-steps, in
//     AU. By value: kernel arguments travel with the launch, so this costs no
//     copy and nothing lands in the inner loop.
// Returns:
//   Nothing; the nine arrays are updated in place.
__global__ void particle_step(double *rx, double *ry, double *rz,
                              double *vx, double *vy, double *vz,
                              double *ax, double *ay, double *az, int n,
                              BodyTable bodies)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double rxs = rx[i], rys = ry[i], rzs = rz[i];
    double vxs = vx[i], vys = vy[i], vzs = vz[i];
    double axs = ax[i], ays = ay[i], azs = az[i];

    // Each sub-step narrows the position once, right after its drift. Narrowing
    // once per force() call instead, or worse hoisting it above the sub-steps,
    // is silent: the force is then evaluated at a stale position and the
    // physics is simply wrong (measured |da| ~ 1 AU, with no crash to warn you).

    // Sub-step 1, coefficient W1.
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;
    rxs += W1 * vxs;
    rys += W1 * vys;
    rzs += W1 * vzs;
    force((float)rxs, (float)rys, (float)rzs, axs, ays, azs, bodies, 0);
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;

    // Sub-step 2, coefficient W0. It is negative: the middle sub-step
    // integrates backwards, which is what cancels the h^3 error term.
    vxs += 0.5 * W0 * axs;
    vys += 0.5 * W0 * ays;
    vzs += 0.5 * W0 * azs;
    rxs += W0 * vxs;
    rys += W0 * vys;
    rzs += W0 * vzs;
    force((float)rxs, (float)rys, (float)rzs, axs, ays, azs, bodies, 1);
    vxs += 0.5 * W0 * axs;
    vys += 0.5 * W0 * ays;
    vzs += 0.5 * W0 * azs;

    // Sub-step 3, coefficient W1 again.
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;
    rxs += W1 * vxs;
    rys += W1 * vys;
    rzs += W1 * vzs;
    force((float)rxs, (float)rys, (float)rzs, axs, ays, azs, bodies, 2);
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;

    rx[i] = rxs;
    ry[i] = rys;
    rz[i] = rzs;
    vx[i] = vxs;
    vy[i] = vys;
    vz[i] = vzs;
    ax[i] = axs;
    ay[i] = ays;
    az[i] = azs;
}

// Fills the acceleration array from the positions, once, before the first step.
// particle_step() starts by reading the acceleration left by the previous step,
// and on the very first step there is no previous step, so this seeds it.
//
// Args:
//   rx, ry, rz: device arrays of length n, positions in AU.
//   ax, ay, az: device arrays of length n, written.
//   n: number of particles.
//   bodies: where the massive bodies are at t = 0.
// Returns:
//   Nothing.
__global__ void initial_accel(const double *rx, const double *ry,
                              const double *rz, double *ax, double *ay,
                              double *az, int n, BodyTable bodies)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    force((float)rx[i], (float)ry[i], (float)rz[i], ax[i], ay[i], az[i],
          bodies, 0);
}

namespace
{
constexpr int BLOCK = 256;

int grid_for(std::int64_t n)
{
    return static_cast<int>((n + BLOCK - 1) / BLOCK);
}

// The constant parts of the table: how many bodies there are and what G*M is
// for each. Body 0 is the Sun, pinned at the origin.
// Args:
//   table: written.
//   planets: heliocentric bodies in solar masses; body b + 1 is planets[b].
// Returns:
//   Nothing.
// Throws:
//   std::runtime_error if there are more bodies than ph::MAX_BODIES.
void fill_gm(BodyTable &table, const std::vector<Planet> &planets)
{
    const std::size_t nb = 1 + planets.size();
    if (nb > ph::MAX_BODIES)
        throw std::runtime_error("raise ph::MAX_BODIES");
    table.nb = static_cast<int>(nb);
    table.gm[0] = static_cast<float>(ph::G * ph::M_Sun);
    for (std::size_t b = 0; b < planets.size(); ++b)
        table.gm[b + 1] = static_cast<float>(ph::G * planets[b].mass);
}

// Records where the bodies are right now into one sub-step of the table. The
// Sun is body 0 and stays at the origin, so it is simply left at zero.
// Args:
//   table: written at `sub`.
//   planets: heliocentric bodies in AU.
//   sub: which sub-step, 0..2.
// Returns:
//   Nothing.
void record_positions(BodyTable &table, const std::vector<Planet> &planets,
                      int sub)
{
    for (std::size_t b = 0; b < planets.size(); ++b)
    {
        table.pos[sub][b + 1][0] = static_cast<float>(planets[b].r.x);
        table.pos[sub][b + 1][1] = static_cast<float>(planets[b].r.y);
        table.pos[sub][b + 1][2] = static_cast<float>(planets[b].r.z);
    }
}

// Advances the massive bodies by one kick-drift-kick sub-step of length w (in
// units of dt), under Sun-only gravity, and records the position the test
// particles must feel DURING that sub-step -- the position after the drift and
// before the closing kick, which is where the CPU version calls particle_acc().
//
// This mirrors Simulation::leapfrog in cpp/simulation.cpp. If the two drift
// apart the GPU build stops matching the CPU reference.
//
// Args:
//   planets: heliocentric bodies in AU and AU/step; mutated in place.
//   w: sub-step length as a multiple of dt; may be negative.
//   table: written at `sub`.
//   sub: which sub-step, 0..2.
// Returns:
//   Nothing.
void advance_bodies(std::vector<Planet> &planets, double w, BodyTable &table,
                    int sub)
{
    for (Planet &p : planets)
    {
        p.v.x += 0.5 * w * p.a.x;
        p.v.y += 0.5 * w * p.a.y;
        p.v.z += 0.5 * w * p.a.z;
        p.r.x += w * p.v.x;
        p.r.y += w * p.v.y;
        p.r.z += w * p.v.z;
    }

    record_positions(table, planets, sub);

    // Same Sun-only gravity the CPU build's plant_acc() applies. Shared rather
    // than rewritten: a second copy of the force law is a second thing that can
    // drift.
    sun_only_acceleration(planets);

    for (Planet &p : planets)
    {
        p.v.x += 0.5 * w * p.a.x;
        p.v.y += 0.5 * w * p.a.y;
        p.v.z += 0.5 * w * p.a.z;
    }
}
} // namespace

GpuSimulation::GpuSimulation(std::vector<Planet> planets, std::int64_t n,
                             const double *rx, const double *ry,
                             const double *rz, const double *vx,
                             const double *vy, const double *vz,
                             std::uint64_t n_step)
    : planets_(std::move(planets)), n_(n), tot_step_(n_step)
{
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&rx_, bytes));
    CUDA_CHECK(cudaMalloc(&ry_, bytes));
    CUDA_CHECK(cudaMalloc(&rz_, bytes));
    CUDA_CHECK(cudaMalloc(&vx_, bytes));
    CUDA_CHECK(cudaMalloc(&vy_, bytes));
    CUDA_CHECK(cudaMalloc(&vz_, bytes));
    CUDA_CHECK(cudaMalloc(&ax_, bytes));
    CUDA_CHECK(cudaMalloc(&ay_, bytes));
    CUDA_CHECK(cudaMalloc(&az_, bytes));

    h_rx_.assign(rx, rx + n);
    h_ry_.assign(ry, ry + n);
    h_rz_.assign(rz, rz + n);
    h_vx_.assign(vx, vx + n);
    h_vy_.assign(vy, vy + n);
    h_vz_.assign(vz, vz + n);

    CUDA_CHECK(cudaMemcpy(rx_, h_rx_.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ry_, h_ry_.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rz_, h_rz_.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vx_, h_vx_.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vy_, h_vy_.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vz_, h_vz_.data(), bytes, cudaMemcpyHostToDevice));

    // Nothing has moved yet, so all three sub-steps see the same positions.
    BodyTable at_zero;
    fill_gm(at_zero, planets_);
    for (int sub = 0; sub < 3; ++sub)
        record_positions(at_zero, planets_, sub);

    initial_accel<<<grid_for(n), BLOCK>>>(rx_, ry_, rz_, ax_, ay_, az_,
                                          static_cast<int>(n), at_zero);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

GpuSimulation::~GpuSimulation()
{
    // cudaFree on a null pointer is a no-op, so this is safe even if the
    // constructor threw partway through.
    cudaFree(rx_);
    cudaFree(ry_);
    cudaFree(rz_);
    cudaFree(vx_);
    cudaFree(vy_);
    cudaFree(vz_);
    cudaFree(ax_);
    cudaFree(ay_);
    cudaFree(az_);
}

void GpuSimulation::launch(const BodyTable &bodies)
{
    particle_step<<<grid_for(n_), BLOCK>>>(rx_, ry_, rz_, vx_, vy_, vz_, ax_,
                                           ay_, az_, static_cast<int>(n_),
                                           bodies);

    // A launch is asynchronous. The first check catches a launch that never
    // started; the second waits, so the next step and any copy see completed
    // data.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GpuSimulation::dump(const std::string &path) const
{
    const std::size_t bytes = static_cast<std::size_t>(n_) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_rx_.data(), rx_, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ry_.data(), ry_, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rz_.data(), rz_, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx_.data(), vx_, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy_.data(), vy_, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz_.data(), vz_, bytes, cudaMemcpyDeviceToHost));

    write_epoch(path, step_, planets_, static_cast<std::size_t>(n_),
                h_rx_.data(), h_ry_.data(), h_rz_.data(),
                h_vx_.data(), h_vy_.data(), h_vz_.data());
}

void GpuSimulation::run(const std::string &dump_dir, std::uint64_t epoch_every)
{
    // Step 0 is the initial condition, matching the CPU build.
    if (!dump_dir.empty())
        dump(dump_dir + "/epoch_0.bin");

    for (std::uint64_t s = 1; s <= tot_step_; ++s)
    {
        // The host advances the bodies through the three sub-steps and hands
        // the kernel a table of where they were.
        BodyTable bodies;
        fill_gm(bodies, planets_);
        advance_bodies(planets_, W1, bodies, 0);
        advance_bodies(planets_, W0, bodies, 1);
        advance_bodies(planets_, W1, bodies, 2);

        launch(bodies);
        step_ = s;

        if (!dump_dir.empty() && (s % epoch_every == 0 || s == tot_step_))
            dump(dump_dir + "/epoch_" + std::to_string(s) + ".bin");
    }
}
