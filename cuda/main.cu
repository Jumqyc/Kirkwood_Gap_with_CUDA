// The GPU build's command line. Everything except the backend lives in the
// shared headers, so this file is only plumbing -- the CPU build's main.cpp is
// the same shape.

#include "../cpp/args.hpp"
#include "../cpp/setup.hpp"
#include "gpu_simulation.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    const Args a = parse_args(argc, argv);

    if (!a.dump_dir.empty())
        std::filesystem::create_directories(a.dump_dir);

    const std::vector<Planet> planets = make_planets(a.with_saturn);

    std::vector<double> rx(a.n_particle), ry(a.n_particle), rz(a.n_particle);
    std::vector<double> vx(a.n_particle), vy(a.n_particle), vz(a.n_particle);
    fill_particles(a.n_particle, a.e_max, a.seed, rx.data(), ry.data(),
                   rz.data(), vx.data(), vy.data(), vz.data());

    GpuSimulation s(planets, a.n_particle, rx.data(), ry.data(), rz.data(),
                    vx.data(), vy.data(), vz.data(), a.n_step);

    const auto t0 = std::chrono::steady_clock::now();
    s.run(a.dump_dir, a.epoch_every);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    // threads=0: this build has no OpenMP.
    print_summary("gpu", a, seconds, 0);
    return 0;
}
