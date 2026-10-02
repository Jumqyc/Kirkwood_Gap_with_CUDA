#include "args.hpp"
#include "epoch_file.hpp"
#include "setup.hpp"
#include "simulation.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

// The body of main, so the exit path is in one place.
//
// Resuming is automatic: if the output directory already holds epoch files, the
// run continues from the newest one instead of starting over. Same rule as the
// GPU build, so a driver script that restarts on failure works against either.
// Delete the directory (or point dump_dir elsewhere) to start fresh; the
// resumption is announced on stdout.
static int run_backend(const Args &a)
{
    // The dumps share the wall clock with the integration below, so a timing
    // run has to pass "" or it is measuring the disk, not the kernel.
    if (!a.dump_dir.empty())
        std::filesystem::create_directories(a.dump_dir);

    std::vector<Planet> planets = make_planets(a.with_saturn, a.eccentric);
    Particles particles(a.n_particle);

    std::uint64_t start_step = 0;
    const std::string previous =
        a.dump_dir.empty() ? std::string{} : newest_epoch(a.dump_dir);

    if (previous.empty())
    {
        fill_particles(a.n_particle, a.e_max, a.seed,
                       particles.r.x.data(), particles.r.y.data(),
                       particles.r.z.data(), particles.v.x.data(),
                       particles.v.y.data(), particles.v.z.data());
    }
    else
    {
        const EpochState s = read_epoch(previous);
        if (static_cast<std::int64_t>(s.rx.size()) != a.n_particle)
            throw std::runtime_error(
                previous + " holds " + std::to_string(s.rx.size()) +
                " particles but this run asks for " +
                std::to_string(a.n_particle));
        restore_planets(planets, s.planets);
        particles.r.x = s.rx;
        particles.r.y = s.ry;
        particles.r.z = s.rz;
        particles.v.x = s.vx;
        particles.v.y = s.vy;
        particles.v.z = s.vz;
        start_step = s.step;
        std::cout << "resuming step=" << start_step << " from " << previous << "\n";
    }

    Simulation s(planets, particles, a.n_step, start_step);

    const auto t0 = std::chrono::steady_clock::now();
    s.run(a.dump_dir, a.epoch_every);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    print_summary("cpu", a, seconds, omp_get_max_threads());
    return 0;
}

int main(int argc, char **argv)
{
    // Without this a bad epoch file, or a dump that cannot be written, ends in
    // std::terminate and an abort with no message.
    try
    {
        return run_backend(parse_args(argc, argv));
    }
    catch (const std::exception &e)
    {
        std::cerr << "kirkwood: " << e.what() << "\n";
        return 1;
    }
}
