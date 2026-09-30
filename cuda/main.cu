// GPU port of the particle half of the Kirkwood simulation.
//
// Design, and why it is shaped this way:
//
//   * The massive bodies stay on the HOST. Their motion does not depend on the
//     test particles (the restricted problem), so the host integrates them and
//     hands the kernel their positions AS ARGUMENTS. Kernel arguments travel
//     with the launch, so this costs no cudaMemcpy at all -- a per-step copy of
//     48 bytes would cost several microseconds of latency, more than the
//     arithmetic it feeds.
//
//   * One thread owns one particle for the WHOLE outer step. The acceleration
//     never leaves a register, and there is no barrier anywhere in the kernel.
//     That is the thing the CPU version cannot do: it splits each step into 12
//     OpenMP regions and pays a barrier for each one.
//
//   * The acceleration array does not exist on the device. It is recomputed at
//     the start of every step instead, which costs one extra force evaluation
//     per step. A first-version trade: keeping it would need either a device
//     array (24 more bytes per particle per step) or a kernel that spans many
//     steps.
//
// Culling is deliberately absent: with e_max = 0.1 nothing is ever removed, so
// leaving it out changes no result and postpones the compaction logic.
//
// The epoch file layout is defined in cpp/simulation.hpp (Simulation::dump) and
// duplicated below on purpose. The CPU build is the reference this port is
// checked against, and refactoring the reference while using it to check
// something else is how you lose the ability to tell which side is wrong. Fold
// the two together once they agree.
//
// The unit system comes from cpp/physics.hpp, which both builds share.
//
// Build: cmake --build <build-dir> --target kirkwood_gpu
// Run:   ./<build-dir>/kirkwood_gpu <n_particles> <n_steps> [dump_dir]
//                                   [epoch_every] [e_max] [saturn=0] [seed]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cpp/physics.hpp"

// Every CUDA call returns an error code; nothing throws, and a failed launch
// leaves the arrays untouched without stopping the program. See cuda/hello.cu.
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        const cudaError_t err_ = (call);                                     \
        if (err_ != cudaSuccess)                                             \
        {                                                                    \
            std::fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__, __LINE__, \
                         #call, cudaGetErrorString(err_));                   \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Yoshida's fourth-order composition coefficients. Same values as
// cpp/simulation.cpp; if they drift apart the two builds stop agreeing.
constexpr double W1 = 1.3512071919596578;
constexpr double W0 = -1.7024143839193153;

// A local Vec3 rather than cpp/vec3.hpp: that header also defines SoAVec3,
// whose OpenMP pragmas nvcc would warn about. This is the whole of what the
// host side needs.
struct Vec3
{
    double x, y, z;
};

// A massive body, heliocentric, Sun pinned at the origin.
struct Planet
{
    double mass;
    Vec3 r, v, a;
};

// Advances the massive bodies by one kick-drift-kick sub-step of length w (in
// units of dt), under Sun-only gravity, and returns the position the test
// particles must feel DURING that sub-step -- the position after the drift and
// before the closing kick, which is where the CPU version calls particle_acc().
//
// Args:
//   planets: massive bodies, mutated in place.
//   w: sub-step length as a multiple of dt; may be negative.
// Returns:
//   The perturber's position in AU, or the Sun's (the origin) if there is none.
//
// This mirrors Simulation::leapfrog in cpp/simulation.cpp. If the two drift
// apart, the GPU build stops matching the reference.
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

    for (Planet &p : planets)
    {
        const double r2 = p.r.x * p.r.x + p.r.y * p.r.y + p.r.z * p.r.z;
        const double r3 = r2 * std::sqrt(r2);
        p.a.x = -ph::G * ph::M_Sun * p.r.x / r3;
        p.a.y = -ph::G * ph::M_Sun * p.r.y / r3;
        p.a.z = -ph::G * ph::M_Sun * p.r.z / r3;
    }
    for (Planet &p : planets)
    {
        p.v.x += 0.5 * w * p.a.x;
        p.v.y += 0.5 * w * p.a.y;
        p.v.z += 0.5 * w * p.a.z;
    }
    return felt;
}


__device__ inline void acceleration(double rx, double ry, double rz,
                              double&ax, double&ay, double&az,
                              double px, double py, double pz,
                            double gm)
{
    const double dx = px-rx;
    const double dy = py-ry;
    const double dz = pz-rz;
    double d = std::sqrt(dx*dx+dy*dy+dz*dz);
    d = d*d*d;
    d = 1/d;
    ax += gm*dx*d;
    ay += gm*dy*d;
    az += gm*dz*d;
}


__global__ void particle_step(double *rx, double *ry, double *rz,
                              double *vx, double *vy, double *vz,
                              double *ax, double *ay, double *az,
                              int n,
                              double p0x, double p0y, double p0z,
                              double p1x, double p1y, double p1z,
                              double p2x, double p2y, double p2z,
                              double gm_sun, double gm_planet)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    
    double rxs = rx[i]; double rys = ry[i]; double rzs = rz[i];
    double vxs = vx[i]; double vys = vy[i]; double vzs = vz[i];
    double axs = ax[i]; double ays = ay[i]; double azs = az[i];
    
    // first leap frog with W0:
    // p.v += 0.5*W1 *p.a
    // p.r += W1*p.a
    vxs += 0.5*W1*axs, vys += 0.5*W1*ays, vzs += 0.5*W1*azs;
    rxs += W1*vxs, rys += W1*vys, rzs += W1*vzs;
    axs = ays = azs = 0;
    acceleration(rxs,rys,rzs,axs,ays,azs,0,0,0,gm_sun); // sun
    acceleration(rxs,rys,rzs,axs,ays,azs,p0x,p0y,p0z,gm_planet); // Jupiter
    vxs += 0.5*W1*axs, vys += 0.5*W1*ays, vzs += 0.5*W1*azs;

    // second frog leap

    vxs += 0.5*W0*axs, vys += 0.5*W0*ays, vzs += 0.5*W0*azs;
    rxs += W0*vxs, rys += W0*vys, rzs += W0*vzs;
    axs = ays = azs = 0;
    acceleration(rxs,rys,rzs,axs,ays,azs,0,0,0,gm_sun); // sun
    acceleration(rxs,rys,rzs,axs,ays,azs,p1x,p1y,p1z,gm_planet); // Jupiter
    vxs += 0.5*W0*axs, vys += 0.5*W0*ays, vzs += 0.5*W0*azs;

    // third frog leap
    vxs += 0.5*W1*axs, vys += 0.5*W1*ays, vzs += 0.5*W1*azs;
    rxs += W1*vxs, rys += W1*vys, rzs += W1*vzs;
    axs = ays = azs = 0;
    acceleration(rxs,rys,rzs,axs,ays,azs,0,0,0,gm_sun); // sun
    acceleration(rxs,rys,rzs,axs,ays,azs,p2x,p2y,p2z,gm_planet); // Jupiter
    vxs += 0.5*W1*axs, vys += 0.5*W1*ays, vzs += 0.5*W1*azs;

    rx[i] = rxs; ry[i] = rys; rz[i] = rzs;
    vx[i] = vxs; vy[i] = vys; vz[i] = vzs;
    ax[i] = axs; ay[i] = ays; az[i] = azs;


}

__global__ void initial_accel(double *rx, double *ry, double *rz,
                                double *ax, double *ay, double *az,
                                int n,
                                double px, double py, double pz,
                                double gm_sun, double gm_planet)
  {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;

      const double x = rx[i], y = ry[i], z = rz[i];
      double axs = 0, ays = 0, azs = 0;
      acceleration(x, y, z, axs, ays, azs, 0, 0, 0, gm_sun); 
      acceleration(x, y, z, axs, ays, azs, px, py, pz, gm_planet);
      ax[i] = axs; ay[i] = ays; az[i] = azs;
  }


    // Launches particle_step for the whole particle set. Complete: this is
    // boilerplate, and every line of it is explained in cuda/hello.cu.
static void launch_particle_step(double *rx, double *ry, double *rz,
                                     double *vx, double *vy, double *vz, 
                                     double *ax, double*ay,double* az,
                                     int n,
                                     const std::array<Vec3, 3> &p)
{
    // 256 threads per block is the usual starting point. The grid is rounded
    // up, which is why the kernel needs its bounds check.
    const int block = 256;
    const int grid = (n + block - 1) / block;

    particle_step<<<grid, block>>>(rx, ry, rz, vx, vy, vz, ax,ay,az,
                                    n,
                                   p[0].x, p[0].y, p[0].z,
                                   p[1].x, p[1].y, p[1].z,
                                   p[2].x, p[2].y, p[2].z,
                                   ph::G * ph::M_Sun, ph::G * ph::M_Jupiter);

    // A launch is asynchronous. The first check catches a launch that never
    // started; the second waits for the kernel to finish, so the next step --
    // and any copy -- sees completed data.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
// YOUR PART ENDS HERE
// ===========================================================================

// Writes one epoch record. The layout is the one documented in
// cpp/simulation.hpp; duplicated here, see the note at the top of this file.
static void write_epoch(const std::string &path, std::uint64_t step,
                        const std::vector<Planet> &planets,
                        const std::vector<double> &rx,
                        const std::vector<double> &ry,
                        const std::vector<double> &rz,
                        const std::vector<double> &vx,
                        const std::vector<double> &vy,
                        const std::vector<double> &vz)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("dump: cannot open " + path);

    const std::uint32_t magic_version[2] = {0x4B49524Bu, 1u};
    const std::uint64_t n_planet = planets.size();
    const double constants[2] = {ph::dt, ph::G};
    f.write(reinterpret_cast<const char *>(magic_version), sizeof(magic_version));
    f.write(reinterpret_cast<const char *>(&n_planet), sizeof(n_planet));
    f.write(reinterpret_cast<const char *>(constants), sizeof(constants));

    const std::uint64_t n_alive = rx.size();
    const std::uint64_t record[2] = {n_alive, step};
    f.write(reinterpret_cast<const char *>(record), sizeof(record));

    std::vector<double> body(6 * planets.size());
    for (std::size_t b = 0; b < planets.size(); ++b)
    {
        body[6 * b + 0] = planets[b].r.x;
        body[6 * b + 1] = planets[b].r.y;
        body[6 * b + 2] = planets[b].r.z;
        body[6 * b + 3] = planets[b].v.x;
        body[6 * b + 4] = planets[b].v.y;
        body[6 * b + 5] = planets[b].v.z;
    }
    auto put = [&f](const std::vector<double> &v)
    {
        f.write(reinterpret_cast<const char *>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(double)));
    };
    put(body);
    put(rx);
    put(ry);
    put(rz);
    put(vx);
    put(vy);
    put(vz);

    f.close();
    if (!f)
        throw std::runtime_error("dump: write failed for " + path);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0]
                  << " <n_particles> <n_steps> [dump_dir] [epoch_every] "
                     "[e_max] [saturn] [seed]\n"
                  << "  saturn must be 0: this port integrates Jupiter only.\n"
                  << "  culling is not implemented, so nothing is ever removed.\n";
        return 1;
    }

    const std::int64_t n = std::stoll(argv[1]);
    const std::int64_t n_step = std::stoll(argv[2]);
    const std::string dump_dir = (argc > 3) ? argv[3] : "data";
    const double e_max = (argc > 5) ? std::stod(argv[5]) : 0.1;
    const bool with_saturn = (argc > 6) && std::stoi(argv[6]) != 0;
    const std::uint64_t seed = (argc > 7) ? std::stoull(argv[7]) : 114514;
    const std::uint64_t epoch_every =
        (argc > 4 && std::stoull(argv[4]) > 0)
            ? std::stoull(argv[4])
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(n_step) / 100);

    if (with_saturn)
    {
        std::cerr << "saturn=1 is not supported by the GPU port yet\n";
        return 1;
    }
    if (!dump_dir.empty())
        std::filesystem::create_directories(dump_dir);

    // Jupiter, on a circular orbit, Sun-only gravity. Same construction as
    // cpp/main.cpp.
    constexpr double JUPITER_A_AU = 5.203;
    std::vector<Planet> planets(1);
    planets[0].mass = ph::M_Jupiter;
    planets[0].r = {JUPITER_A_AU, 0.0, 0.0};
    planets[0].v = {0.0, std::sqrt(ph::G * ph::M_Sun / JUPITER_A_AU), 0.0};
    planets[0].a = {-ph::G * ph::M_Sun / (JUPITER_A_AU * JUPITER_A_AU), 0.0,
                    0.0};

    // Initial conditions. The draw order and the placements must match
    // cpp/main.cpp exactly, or the two builds start from different particles
    // and nothing downstream can be compared.
    std::vector<double> h_rx(n), h_ry(n), h_rz(n);
    std::vector<double> h_vx(n), h_vy(n), h_vz(n);
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> angle(0.0,
                                                     2.0 * std::numbers::pi);
        std::uniform_real_distribution<double> radius(2.0, 3.5);
        std::uniform_real_distribution<double> eccentricity(0, e_max);
        for (std::int64_t i = 0; i < n; ++i)
        {
            const double th = angle(rng);
            double r = radius(rng);
            const double e = eccentricity(rng);
            const double speed =
                std::sqrt(ph::G * ph::M_Sun / r * (1 + e) / (1 - e));
            r *= (1 - e);
            h_rx[i] = r * std::cos(th);
            h_ry[i] = r * std::sin(th);
            h_rz[i] = 0.0;
            h_vx[i] = -speed * std::sin(th);
            h_vy[i] = speed * std::cos(th);
            h_vz[i] = 0.0;
        }
    }

    // Six device arrays, one per coordinate. SoA on the device as well as on
    // the host: a warp then reads 32 consecutive doubles with one transaction.
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
    double *drx = nullptr, *dry = nullptr, *drz = nullptr;
    double *dvx = nullptr, *dvy = nullptr, *dvz = nullptr;
    double *dax = nullptr, *day = nullptr, *daz = nullptr;
    CUDA_CHECK(cudaMalloc(&drx, bytes));
    CUDA_CHECK(cudaMalloc(&dry, bytes));
    CUDA_CHECK(cudaMalloc(&drz, bytes));
    CUDA_CHECK(cudaMalloc(&dvx, bytes));
    CUDA_CHECK(cudaMalloc(&dvy, bytes));
    CUDA_CHECK(cudaMalloc(&dvz, bytes));
    CUDA_CHECK(cudaMalloc(&dax, bytes));
    CUDA_CHECK(cudaMalloc(&day, bytes));
    CUDA_CHECK(cudaMalloc(&daz, bytes));
    CUDA_CHECK(cudaMemcpy(drx, h_rx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dry, h_ry.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(drz, h_rz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvz, h_vz.data(), bytes, cudaMemcpyHostToDevice));

    // Brings the particle state back and writes one epoch file. Only called at
    // epoch boundaries, so the 48 MB round trip is paid ~100 times over a run
    // rather than once per step.
    auto dump_epoch = [&](std::uint64_t at_step)
    {
        CUDA_CHECK(cudaMemcpy(h_rx.data(), drx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ry.data(), dry, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rz.data(), drz, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vx.data(), dvx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vy.data(), dvy, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vz.data(), dvz, bytes, cudaMemcpyDeviceToHost));
        write_epoch(dump_dir + "/epoch_" + std::to_string(at_step) + ".bin",
                    at_step, planets, h_rx, h_ry, h_rz, h_vx, h_vy, h_vz);
    };

    const std::uint64_t total = static_cast<std::uint64_t>(n_step);

    // init 

    const int block = 256;
    const int grid = (static_cast<int>(n) + block - 1) / block;
    initial_accel<<<grid, block>>>(drx, dry, drz,
                                     dax, day, daz,
                                     static_cast<int>(n),
                                     planets[0].r.x, planets[0].r.y, planets[0].r.z,
                                     ph::G * ph::M_Sun,
                                     ph::G * ph::M_Jupiter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto t0 = std::chrono::steady_clock::now();

    if (!dump_dir.empty())
        dump_epoch(0);

    for (std::uint64_t s = 1; s <= total; ++s)
    {
        // The host advances the planets through the three sub-steps and keeps
        // the three positions the particles must feel. Three Vec3 is 72 bytes,
        // well inside the 4 KB kernel-argument limit.
        std::array<Vec3, 3> felt;
        felt[0] = planet_substep(planets, W1);
        felt[1] = planet_substep(planets, W0);
        felt[2] = planet_substep(planets, W1);

        launch_particle_step(drx, dry, drz, dvx, dvy, dvz, dax,day,daz,static_cast<int>(n),
                             felt);

        if (!dump_dir.empty() && (s % epoch_every == 0 || s == total))
            dump_epoch(s);
    }

    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();

    std::cout << "gpu n_particle=" << n << " n_step=" << n_step
              << " e_max=" << e_max << " saturn=0"
              << " seed=" << seed
              << " seconds=" << sec
              << " ns_per_particle_step=" << (sec / n / n_step * 1e9)
              << " dump_dir=" << (dump_dir.empty() ? "-" : dump_dir) << "\n";

    CUDA_CHECK(cudaFree(drx));
    CUDA_CHECK(cudaFree(dry));
    CUDA_CHECK(cudaFree(drz));
    CUDA_CHECK(cudaFree(dvx));
    CUDA_CHECK(cudaFree(dvy));
    CUDA_CHECK(cudaFree(dvz));
    return 0;
}
