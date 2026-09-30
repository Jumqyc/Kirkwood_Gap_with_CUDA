#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include "body.hpp"
#include "physics.hpp"

// Where the massive bodies are and where the test particles start. Both builds
// call these, and that is the point: a CPU run and a GPU run that agree must
// start from bit-identical particles, and the only way to keep them identical
// is to draw them from one function.

// A Kirkwood gap sits at a fixed fraction of the perturber's axis -- 2.50 AU is
// the 3:1 of 5.203 -- so a planet placed anywhere else puts every resonance
// outside the test-particle disk and no gap can form there at all.
inline constexpr double JUPITER_A_AU = 5.203;
inline constexpr double SATURN_A_AU = 9.537;

// A circular orbit in the heliocentric frame, starting on the +x axis.
// Args:
//   mass: body mass in solar masses.
//   a_au: orbital radius in AU.
// Returns:
//   The body, with its acceleration left at zero -- the simulation computes it.
inline Planet circular_body(double mass, double a_au)
{
    Planet body;
    body.mass = mass;
    body.r = Vec3{a_au, 0.0, 0.0};
    body.v = Vec3{0.0, std::sqrt(ph::G * ph::M_Sun / a_au), 0.0};
    body.a = Vec3{0.0, 0.0, 0.0};
    return body;
}

// Fills in the Sun-only acceleration of every body, from its current position.
//
// The bodies come out of make_planets() already consistent, because a build
// that integrates them on the host -- the GPU one -- has no plant_acc() to call
// and would otherwise take its first kick with a zero acceleration. That is a
// one-off O(h^2) error, silent, and worth 0.05 AU of Jupiter after a few
// hundred steps.
//
// Args:
//   planets: bodies to update in place.
// Returns:
//   Nothing.
inline void sun_only_acceleration(std::vector<Planet> &planets)
{
    for (Planet &p : planets)
    {
        const double r2 = p.r.x * p.r.x + p.r.y * p.r.y + p.r.z * p.r.z;
        const double r3 = r2 * std::sqrt(r2);
        p.a.x = -ph::G * ph::M_Sun * p.r.x / r3;
        p.a.y = -ph::G * ph::M_Sun * p.r.y / r3;
        p.a.z = -ph::G * ph::M_Sun * p.r.z / r3;
    }
}

// The massive bodies of the restricted problem.
//
// Their acceleration is Sun-only, so Jupiter and Saturn do not pull on each
// other here. That is enough for a secular resonance: the test particles feel
// every body, so nu6 is present in their dynamics.
//
// Args:
//   with_saturn: add Saturn at 9.537 AU.
// Returns:
//   Jupiter, and Saturn if asked for, with their accelerations already set.
inline std::vector<Planet> make_planets(bool with_saturn)
{
    std::vector<Planet> planets{circular_body(ph::M_Jupiter, JUPITER_A_AU)};
    if (with_saturn)
        planets.push_back(circular_body(ph::M_Saturn, SATURN_A_AU));
    sun_only_acceleration(planets);
    return planets;
}

// The initial test particles: a disk uniform in semimajor axis on [2.0, 3.5] AU
// with eccentricities uniform on [0, e_max], every one placed at perihelion.
//
// Args:
//   n: number of particles.
//   e_max: upper bound of the initial eccentricity.
//   seed: RNG seed, saved with the run.
//   rx..vz: output arrays of length n; positions in AU, velocities in AU per
//     step. Their element type may be float or double -- the draws are made in
//     double either way, so a float build starts from the same values rounded.
// Returns:
//   Nothing; the six arrays are filled.
template <typename T>
void fill_particles(std::int64_t n, double e_max, std::uint64_t seed,
                    T *rx, T *ry, T *rz, T *vx, T *vy, T *vz)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * std::numbers::pi);
    std::uniform_real_distribution<double> radius(2.0, 3.5);
    std::uniform_real_distribution<double> eccentricity(0, e_max);

    for (std::int64_t i = 0; i < n; ++i)
    {
        const double th = angle(rng);
        double r = radius(rng);
        const double e = eccentricity(rng);

        // Placed at perihelion, so r is scaled down by (1 - e) while the speed
        // carries the (1 + e) / (1 - e) of the vis-viva speed there.
        const double v = std::sqrt(ph::G * ph::M_Sun / r * (1 + e) / (1 - e));
        r *= (1 - e);
        rx[i] = r * std::cos(th);
        ry[i] = r * std::sin(th);
        rz[i] = 0.0;
        vx[i] = -v * std::sin(th);
        vy[i] = v * std::cos(th);
        vz[i] = 0.0;
    }
}
