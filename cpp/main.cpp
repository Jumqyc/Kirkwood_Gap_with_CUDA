#include "simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
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
                  << " <n_particles> <n_steps> [dump_dir] [epoch_every]\n"
                  << "  dump_dir     defaults to 'data'; pass \"\" to skip the "
                     "dumps when timing.\n"
                  << "  epoch_every  steps between dumps, defaults to whatever "
                     "gives ~100 epochs.\n";
        return 1;
    }

    // The dumps share the wall clock with the integration below, so a timing
    // run has to pass "" or it is measuring the disk, not the kernel.
    const std::string dump_dir = (argc > 3) ? argv[3] : "data";
    if (!dump_dir.empty())
        std::filesystem::create_directories(dump_dir);

    // Jupiter's semimajor axis in AU. The Kirkwood gaps sit at fixed fractions
    // of this -- 2.50 AU is the 3:1 -- so a perturber anywhere else puts every
    // resonance outside the test-particle disk and no gap can form.
    constexpr double JUPITER_A_AU = 5.203;

    Planet Jupiter;
    Jupiter.mass = ph::M_Jupiter;
    Jupiter.r = Vec3({JUPITER_A_AU, 0, 0});
    Jupiter.v = Vec3({0, std::sqrt(ph::G * ph::M_Sun / JUPITER_A_AU), 0});
    Jupiter.a = Vec3({0, 0, 0});
    std::vector<Planet> planets;
    planets.push_back(Jupiter);

    std::int64_t n_particle = std::stoi(argv[1]);
    Particles particles(n_particle);
    std::int64_t n_step = std::stoi(argv[2]);

    const std::uint64_t seed = 114514;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * std::numbers::pi);
    std::uniform_real_distribution<double> radius(2.0, 3.5);
    std::uniform_real_distribution<double> eccentricity(0, 0.3);

    for (std::size_t i = 0; i < (std::size_t)n_particle; ++i)
    {
        double th = angle(rng);
        double r = radius(rng);
        const double e = eccentricity(rng);

        double v = std::sqrt(ph::G * ph::M_Sun / r * (1 + e) / (1 - e));
        r *= (1 - e);
        particles.r.x[i] = r * std::cos(th);
        particles.r.y[i] = r * std::sin(th);
        particles.r.z[i] = 0.0;
        particles.v.x[i] = -v * std::sin(th);
        particles.v.y[i] = v * std::cos(th);
        particles.v.z[i] = 0.0;
    }

    // Targeting ~100 epochs keeps the file count bounded however long the run
    // is. run() rounds this down to a whole number of cull intervals, so the
    // actual count comes out at or just under 100.
    const std::uint64_t epoch_every =
        (argc > 4) ? std::stoull(argv[4])
                   : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(n_step) / 100);

    auto s = Simulation(
        planets,
        particles,
        n_step);
    // double E0 = energy(Jupiter);
    auto t0 = std::chrono::steady_clock::now();
    s.run(dump_dir, epoch_every);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Seed:  " << seed
              << "  Total time use:  " << sec
              << "  Each particle needs  " << (sec / n_particle / n_step * 1e9) << "ns"
              << "threads" << omp_get_max_threads()
              << std::endl;

    return 0;
}
