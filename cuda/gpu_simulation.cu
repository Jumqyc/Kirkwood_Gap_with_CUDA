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
// leaves the arrays untouched without stopping the program. See cuda/hello.cu.
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

// Adds one body's gravitational acceleration to the accumulator.
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
// arithmetic between them is float.
//
// The perturber's position and gm arrive as float too, so the host converts
// them once instead of every thread converting them on every call: another
// 1.25x.
__device__ inline void acceleration(double rx, double ry, double rz,
                                    double &ax, double &ay, double &az,
                                    float px, float py, float pz, float gm)
{
    const float frx = (float)rx, fry = (float)ry, frz = (float)rz;
    const float dx = px - frx;
    const float dy = py - fry;
    const float dz = pz - frz;
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float inv_r = rsqrtf(r2);
    const float inv_r3 = inv_r * inv_r * inv_r;
    ax += (double)(gm * dx * inv_r3);
    ay += (double)(gm * dy * inv_r3);
    az += (double)(gm * dz * inv_r3);
}

// Advances every test particle by one outer step: the three KDK sub-steps of
// the Yoshida composition, with the acceleration carried in a register
// throughout. There is no barrier anywhere in this kernel and the acceleration
// never touches memory, which is the shape the CPU version cannot take.
//
// Args:
//   rx, ry, rz: device arrays of length n, positions in AU.
//   vx, vy, vz: device arrays of length n, velocities in AU per step.
//   ax, ay, az: device arrays of length n, the acceleration in AU/step^2, as
//     left by the previous step. Written back consistent with the new position.
//   n: number of particles. Threads past the end return without writing.
//   p0*, p1*, p2*: the perturber's position during sub-steps 1, 2 and 3, in AU,
//     passed by value. Kernel arguments travel with the launch, so this costs
//     no copy. The Sun sits at the origin and needs no argument.
//   gm_sun, gm_planet: G*M in AU^3 / step^2.
// Returns:
//   Nothing; the nine arrays are updated in place.
__global__ void particle_step(double *rx, double *ry, double *rz,
                              double *vx, double *vy, double *vz,
                              double *ax, double *ay, double *az, int n,
                              float p0x, float p0y, float p0z,
                              float p1x, float p1y, float p1z,
                              float p2x, float p2y, float p2z,
                              float gm_sun, float gm_planet)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double rxs = rx[i], rys = ry[i], rzs = rz[i];
    double vxs = vx[i], vys = vy[i], vzs = vz[i];
    double axs = ax[i], ays = ay[i], azs = az[i];

    // Each sub-step keeps its own single-precision copy of the position, made
    // once after that sub-step's drift. Making it once per acceleration() call
    // instead, or worse hoisting it out of the sub-steps, is silent: the force
    // is then evaluated at a stale position and the physics is simply wrong
    // (measured |da| ~ 1 AU, with no crash to warn you).

    // Sub-step 1, coefficient W1, perturber at p0.
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;
    rxs += W1 * vxs;
    rys += W1 * vys;
    rzs += W1 * vzs;
    {
        const float frx = (float)rxs, fry = (float)rys, frz = (float)rzs;
        axs = ays = azs = 0.0;
        acceleration(frx, fry, frz, axs, ays, azs, 0.0f, 0.0f, 0.0f, gm_sun);
        acceleration(frx, fry, frz, axs, ays, azs, p0x, p0y, p0z, gm_planet);
    }
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;

    // Sub-step 2, coefficient W0 (negative: the middle sub-step integrates
    // backwards, which is what cancels the h^3 error term), perturber at p1.
    vxs += 0.5 * W0 * axs;
    vys += 0.5 * W0 * ays;
    vzs += 0.5 * W0 * azs;
    rxs += W0 * vxs;
    rys += W0 * vys;
    rzs += W0 * vzs;
    {
        const float frx = (float)rxs, fry = (float)rys, frz = (float)rzs;
        axs = ays = azs = 0.0;
        acceleration(frx, fry, frz, axs, ays, azs, 0.0f, 0.0f, 0.0f, gm_sun);
        acceleration(frx, fry, frz, axs, ays, azs, p1x, p1y, p1z, gm_planet);
    }
    vxs += 0.5 * W0 * axs;
    vys += 0.5 * W0 * ays;
    vzs += 0.5 * W0 * azs;

    // Sub-step 3, coefficient W1 again, perturber at p2.
    vxs += 0.5 * W1 * axs;
    vys += 0.5 * W1 * ays;
    vzs += 0.5 * W1 * azs;
    rxs += W1 * vxs;
    rys += W1 * vys;
    rzs += W1 * vzs;
    {
        const float frx = (float)rxs, fry = (float)rys, frz = (float)rzs;
        axs = ays = azs = 0.0;
        acceleration(frx, fry, frz, axs, ays, azs, 0.0f, 0.0f, 0.0f, gm_sun);
        acceleration(frx, fry, frz, axs, ays, azs, p2x, p2y, p2z, gm_planet);
    }
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
//   px, py, pz: the perturber's position at t = 0, in AU.
//   gm_sun, gm_planet: G*M in AU^3 / step^2.
// Returns:
//   Nothing.
__global__ void initial_accel(const double *rx, const double *ry,
                              const double *rz, double *ax, double *ay,
                              double *az, int n, float px, float py, float pz,
                              float gm_sun, float gm_planet)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const double x = rx[i], y = ry[i], z = rz[i];
    double axs = 0.0, ays = 0.0, azs = 0.0;
    acceleration(x, y, z, axs, ays, azs, 0.0f, 0.0f, 0.0f, gm_sun);
    acceleration(x, y, z, axs, ays, azs, px, py, pz, gm_planet);
    ax[i] = axs;
    ay[i] = ays;
    az[i] = azs;
}

namespace
{
constexpr int BLOCK = 256;

int grid_for(std::int64_t n)
{
    return static_cast<int>((n + BLOCK - 1) / BLOCK);
}

// Advances the massive bodies by one kick-drift-kick sub-step of length w (in
// units of dt), under Sun-only gravity, and returns the position the test
// particles must feel DURING that sub-step -- the position after the drift and
// before the closing kick, which is where the CPU version calls particle_acc().
//
// This mirrors Simulation::leapfrog in cpp/simulation.cpp. If the two drift
// apart the GPU build stops matching the CPU reference.
//
// Args:
//   planets: massive bodies, mutated in place.
//   w: sub-step length as a multiple of dt; may be negative.
// Returns:
//   The first body's position in AU, or the origin if there is none.
Vec3 planet_substep(std::vector<Planet> &planets, double w)
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

    const Vec3 felt = planets.empty() ? Vec3{0.0, 0.0, 0.0} : planets[0].r;

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
    return felt;
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
    const std::size_t dbytes = static_cast<std::size_t>(n) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&rx_, bytes));
    CUDA_CHECK(cudaMalloc(&ry_, bytes));
    CUDA_CHECK(cudaMalloc(&rz_, bytes));
    CUDA_CHECK(cudaMalloc(&vx_, bytes));
    CUDA_CHECK(cudaMalloc(&vy_, bytes));
    CUDA_CHECK(cudaMalloc(&vz_, bytes));
    CUDA_CHECK(cudaMalloc(&ax_, dbytes));
    CUDA_CHECK(cudaMalloc(&ay_, dbytes));
    CUDA_CHECK(cudaMalloc(&az_, dbytes));

    // The host draws in double; the device holds float. The narrowing happens
    // here, once, and identically to what the CPU build keeps.
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

    initial_accel<<<grid_for(n), BLOCK>>>(
        rx_, ry_, rz_, ax_, ay_, az_, static_cast<int>(n),
        static_cast<float>(planets_.empty() ? 0.0 : planets_[0].r.x),
        static_cast<float>(planets_.empty() ? 0.0 : planets_[0].r.y),
        static_cast<float>(planets_.empty() ? 0.0 : planets_[0].r.z),
        static_cast<float>(ph::G * ph::M_Sun),
        static_cast<float>(ph::G * (planets_.empty() ? 0.0 : planets_[0].mass)));
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

void GpuSimulation::launch(const std::array<Vec3, 3> &perturber)
{
    particle_step<<<grid_for(n_), BLOCK>>>(
        rx_, ry_, rz_, vx_, vy_, vz_, ax_, ay_, az_, static_cast<int>(n_),
        static_cast<float>(perturber[0].x), static_cast<float>(perturber[0].y),
        static_cast<float>(perturber[0].z),
        static_cast<float>(perturber[1].x), static_cast<float>(perturber[1].y),
        static_cast<float>(perturber[1].z),
        static_cast<float>(perturber[2].x), static_cast<float>(perturber[2].y),
        static_cast<float>(perturber[2].z),
        static_cast<float>(ph::G * ph::M_Sun),
        static_cast<float>(ph::G * (planets_.empty() ? 0.0 : planets_[0].mass)));

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
        // The host advances the planets through the three sub-steps and hands
        // the kernel the three positions. Three Vec3 is 72 bytes, well inside
        // the kernel-argument limit.
        std::array<Vec3, 3> felt{};
        felt[0] = planet_substep(planets_, W1);
        felt[1] = planet_substep(planets_, W0);
        felt[2] = planet_substep(planets_, W1);

        launch(felt);
        step_ = s;

        if (!dump_dir.empty() && (s % epoch_every == 0 || s == tot_step_))
            dump(dump_dir + "/epoch_" + std::to_string(s) + ".bin");
    }
}
