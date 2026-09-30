#include "simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <omp.h>
#include <random>
#include <string>

double energy(const Planet &p)
{
    double e = 0;
    e += 0.5 * p.mass * (p.v.x * p.v.x + p.v.y * p.v.y + p.v.z * p.v.z);
    e -= ph::G * ph::M_Sun * p.mass / std::sqrt(p.r.x * p.r.x + p.r.y * p.r.y + p.r.z * p.r.z);
    return e;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0]
                  << " <n_particles> <n_steps> [dump_dir] [epoch_every] "
                     "[e_max] [saturn] [seed]\n"
                  << "  dump_dir     defaults to 'data'; \"\" skips the dumps, "
                     "which is what a timing run wants.\n"
                  << "  epoch_every  steps between dumps; 0 targets ~100 "
                     "epochs, which is the default.\n"
                  << "  e_max        upper bound of the initial eccentricity, "
                     "default 0.1.\n"
                  << "  saturn       1 adds Saturn at 9.537 AU, default 0.\n"
                  << "  seed         default 114514.\n";
        return 1;
    }

    const std::int64_t n_particle = std::stoll(argv[1]);
    const std::int64_t n_step = std::stoll(argv[2]);
    const std::string dump_dir = (argc > 3) ? argv[3] : "data";
    const double e_max = (argc > 5) ? std::stod(argv[5]) : 0.1;
    const bool with_saturn = (argc > 6) && std::stoi(argv[6]) != 0;
    const std::uint64_t seed = (argc > 7) ? std::stoull(argv[7]) : 114514;

    // Targeting ~100 epochs keeps the file count bounded however long the run
    // is. run() rounds this down to a whole number of cull intervals, so the
    // actual count comes out at or just under 100. 0 means "pick the default",
    // which a driver script can pass without having to compute it.
    const std::uint64_t epoch_every_requested = (argc > 4) ? std::stoull(argv[4]) : 0;
    const std::uint64_t epoch_every =
        epoch_every_requested > 0
            ? epoch_every_requested
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(n_step) / 100);

    // The dumps share the wall clock with the integration below, so a timing
    // run has to pass "" or it is measuring the disk, not the kernel.
    if (!dump_dir.empty())
        std::filesystem::create_directories(dump_dir);

    // Semimajor axes in AU. The Kirkwood gaps sit at fixed fractions of the
    // perturber's axis -- 2.50 AU is the 3:1 of 5.203 -- so a planet placed
    // anywhere else puts every resonance outside the test-particle disk and no
    // gap can form there at all.
    constexpr double JUPITER_A_AU = 5.203;
    constexpr double SATURN_A_AU = 9.537;

    // A circular orbit in the heliocentric frame, starting on the +x axis.
    const auto circular = [](double mass, double a_au)
    {
        Planet body;
        body.mass = mass;
        body.r = Vec3({a_au, 0, 0});
        body.v = Vec3({0, std::sqrt(ph::G * ph::M_Sun / a_au), 0});
        body.a = Vec3({0, 0, 0});
        return body;
    };

    // Note: plant_acc() gives the planets Sun-only gravity, so Jupiter and
    // Saturn do not pull on each other here. That is the restricted problem,
    // and it is enough for a secular resonance: the test particles feel every
    // body through particle_acc(), so nu6 is present in their dynamics.
    std::vector<Planet> planets{circular(ph::M_Jupiter, JUPITER_A_AU)};
    if (with_saturn)
        planets.push_back(circular(ph::M_Saturn, SATURN_A_AU));

    Particles particles(n_particle);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * std::numbers::pi);
    std::uniform_real_distribution<double> radius(2.0, 3.5);
    std::uniform_real_distribution<double> eccentricity(0, e_max);

    for (std::size_t i = 0; i < (std::size_t)n_particle; ++i)
    {
        double th = angle(rng);
        double r = radius(rng);
        const double e = eccentricity(rng);

        // Placed at perihelion, so r is scaled down by (1 - e) while the speed
        // carries the (1 + e) / (1 - e) of the vis-viva speed there.
        double v = std::sqrt(ph::G * ph::M_Sun / r * (1 + e) / (1 - e));
        r *= (1 - e);
        particles.r.x[i] = r * std::cos(th);
        particles.r.y[i] = r * std::sin(th);
        particles.r.z[i] = 0.0;
        particles.v.x[i] = -v * std::sin(th);
        particles.v.y[i] = v * std::cos(th);
        particles.v.z[i] = 0.0;
    }

    auto s = Simulation(planets, particles, n_step);
    auto t0 = std::chrono::steady_clock::now();
    s.run(dump_dir, epoch_every);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // One line of key=value so a driver script can parse the run instead of
    // scraping prose. dump_dir=- means nothing was written.
    std::cout << "n_particle=" << n_particle
              << " n_step=" << n_step
              << " e_max=" << e_max
              << " saturn=" << (with_saturn ? 1 : 0)
              << " seed=" << seed
              << " threads=" << omp_get_max_threads()
              << " seconds=" << sec
              << " ns_per_particle_step=" << (sec / n_particle / n_step * 1e9)
              << " dump_dir=" << (dump_dir.empty() ? "-" : dump_dir)
              << "\n";

    return 0;
}
