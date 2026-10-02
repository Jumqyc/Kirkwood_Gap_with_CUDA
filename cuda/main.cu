// The GPU build's command line. Everything except the backend lives in the
// shared headers, so this file is only plumbing -- the CPU build's main.cpp is
// the same shape.

#include "../cpp/args.hpp"
#include "../cpp/epoch_file.hpp"
#include "../cpp/setup.hpp"
#include "gpu_simulation.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// The body of main, so the exit path is in one place.
//
// Resuming is automatic: if the output directory already holds epoch files, the
// run continues from the newest one instead of starting over. That is what makes
// a wrapper script able to restart a multi-hour run after a crash -- re-issuing
// the same command is the whole recovery procedure. Delete the directory (or
// point dump_dir elsewhere) to start fresh, and note that the resumption is
// always announced on stdout, never silent.
static int run_backend(const Args &a)
{
    if (!a.dump_dir.empty())
        std::filesystem::create_directories(a.dump_dir);

    std::vector<Planet> planets = make_planets(a.with_saturn, a.eccentric);

    std::vector<double> rx(a.n_particle), ry(a.n_particle), rz(a.n_particle);
    std::vector<double> vx(a.n_particle), vy(a.n_particle), vz(a.n_particle);

    std::uint64_t start_step = 0;
    const std::string previous =
        a.dump_dir.empty() ? std::string{} : newest_epoch(a.dump_dir);

    if (previous.empty())
    {
        fill_particles(a.n_particle, a.e_max, a.seed, rx.data(), ry.data(),
                       rz.data(), vx.data(), vy.data(), vz.data());
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
        rx = s.rx;
        ry = s.ry;
        rz = s.rz;
        vx = s.vx;
        vy = s.vy;
        vz = s.vz;
        start_step = s.step;
        std::cout << "resuming step=" << start_step << " from " << previous << "\n";
    }

    GpuSimulation s(planets, a.n_particle, rx.data(), ry.data(), rz.data(),
                    vx.data(), vy.data(), vz.data(), a.n_step, start_step,
                    a.wisdom_holman);

    const auto t0 = std::chrono::steady_clock::now();
    s.run(a.dump_dir, a.epoch_every);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    // threads=0: this build has no OpenMP.
    print_summary("gpu", a, seconds, 0);
    return 0;
}

int main(int argc, char **argv)
{
    // Without this a bad epoch file, or a dump that cannot be written, ends in
    // std::terminate and an abort with no message. The failure modes here are
    // mostly the filesystem, so they deserve a sentence.
    try
    {
        return run_backend(parse_args(argc, argv));
    }
    catch (const std::exception &e)
    {
        std::cerr << "kirkwood_gpu: " << e.what() << "\n";
        return 1;
    }
}
