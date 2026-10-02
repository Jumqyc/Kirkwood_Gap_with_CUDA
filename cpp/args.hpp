#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

// The command line, parsed in one place so the CPU and the GPU binary always
// mean the same thing by the same arguments. They are interchangeable on
// purpose: a run is reproduced by re-issuing its command line against whichever
// build you want.
struct Args
{
    std::int64_t n_particle = 0;
    std::int64_t n_step = 0;
    std::string dump_dir = "data";

    // Steps between epoch dumps. 0 means "target about 100 epochs", which keeps
    // the file count bounded however long the run is.
    std::uint64_t epoch_every = 0;

    double e_max = 0.1;
    bool with_saturn = false;
    std::uint64_t seed = 114514;

    // Give the perturbers their real eccentricities. False keeps them circular,
    // which was this project's first model and produces none of the
    // higher-order gaps: without precessing perihelia there is no nu5 or nu6.
    bool eccentric = false;

    // Integrate with the Wisdom-Holman mapping instead of the Yoshida-4
    // leapfrog composition. The drift becomes an exact Kepler advance, so
    // the step size is set by the perturbation rather than by perihelion.
    bool wisdom_holman = false;
};

inline void print_usage(const char *argv0)
{
    std::cerr << "usage: " << argv0
              << " <n_particles> <n_steps> [dump_dir] [epoch_every] "
                 "[e_max] [saturn] [seed] [eccentric] [wisdom_holman]\n"
              << "  dump_dir     defaults to 'data'; \"\" skips the dumps, "
                 "which is what a timing run wants.\n"
              << "  epoch_every  steps between dumps; 0 targets ~100 epochs, "
                 "which is the default.\n"
              << "  e_max        upper bound of the initial eccentricity, "
                 "default 0.1.\n"
              << "  saturn       1 adds Saturn, default 0.\n"
              << "  seed         default 114514.\n"
              << "  eccentric    1 gives the perturbers their real "
                 "eccentricities, default 0.\n"
              << "  wisdom_holman  1 uses the Wisdom-Holman mapping, default 0 "
                 "(Yoshida-4).\n";
}

// Args:
//   argc, argv: the process's arguments.
// Returns:
//   The parsed command line. Exits with status 1 after printing the usage if
//   there are too few arguments.
inline Args parse_args(int argc, char **argv)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        std::exit(1);
    }

    Args a;
    a.n_particle = std::stoll(argv[1]);
    a.n_step = std::stoll(argv[2]);
    a.dump_dir = (argc > 3) ? argv[3] : "data";
    a.e_max = (argc > 5) ? std::stod(argv[5]) : 0.1;
    a.with_saturn = (argc > 6) && std::stoi(argv[6]) != 0;
    a.seed = (argc > 7) ? std::stoull(argv[7]) : 114514;
    a.eccentric = (argc > 8) && std::stoi(argv[8]) != 0;
    a.wisdom_holman = (argc > 9) && std::stoi(argv[9]) != 0;

    const std::uint64_t requested = (argc > 4) ? std::stoull(argv[4]) : 0;
    a.epoch_every = requested > 0
                        ? requested
                        : std::max<std::uint64_t>(
                              1, static_cast<std::uint64_t>(a.n_step) / 100);
    return a;
}

// The one line both builds print when they finish, so a driver script can parse
// a run instead of scraping prose. `backend` is "cpu" or "gpu"; dump_dir=- means
// nothing was written.
inline void print_summary(const char *backend, const Args &a, double seconds,
                          int threads)
{
    std::cout << "backend=" << backend
              << " n_particle=" << a.n_particle
              << " n_step=" << a.n_step
              << " e_max=" << a.e_max
              << " saturn=" << (a.with_saturn ? 1 : 0)
              << " eccentric=" << (a.eccentric ? 1 : 0)
              << " integrator=" << (a.wisdom_holman ? "wh" : "yoshida4")
              << " seed=" << a.seed
              << " threads=" << threads
              << " seconds=" << seconds
              << " ns_per_particle_step="
              << (seconds / a.n_particle / a.n_step * 1e9)
              << " dump_dir=" << (a.dump_dir.empty() ? "-" : a.dump_dir)
              << "\n";
}
