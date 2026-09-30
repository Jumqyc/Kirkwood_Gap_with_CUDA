#include "args.hpp"
#include "setup.hpp"
#include "simulation.hpp"

#include <chrono>
#include <filesystem>
#include <omp.h>
#include <vector>

int main(int argc, char **argv)
{
    const Args a = parse_args(argc, argv);

    // The dumps share the wall clock with the integration below, so a timing
    // run has to pass "" or it is measuring the disk, not the kernel.
    if (!a.dump_dir.empty())
        std::filesystem::create_directories(a.dump_dir);

    const std::vector<Planet> planets = make_planets(a.with_saturn);

    Particles particles(a.n_particle);
    fill_particles(a.n_particle, a.e_max, a.seed,
                   particles.r.x.data(), particles.r.y.data(),
                   particles.r.z.data(), particles.v.x.data(),
                   particles.v.y.data(), particles.v.z.data());

    Simulation s(planets, particles, a.n_step);

    const auto t0 = std::chrono::steady_clock::now();
    s.run(a.dump_dir, a.epoch_every);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    print_summary("cpu", a, seconds, omp_get_max_threads());
    return 0;
}
