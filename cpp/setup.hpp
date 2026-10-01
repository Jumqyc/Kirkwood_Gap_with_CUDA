#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "body.hpp"
#include "physics.hpp"

// Where the massive bodies are and where the test particles start. Both builds
// call these, and that is the point: a CPU run and a GPU run that agree must
// start from bit-identical particles, and the only way to keep them identical
// is to draw them from one function.

// A Kirkwood gap sits at a fixed fraction of the perturber's axis -- 2.50 AU is
// the 3:1 of 5.20 -- so a planet placed anywhere else puts every resonance
// outside the test-particle disk and no gap can form there at all. The axes and
// eccentricities come from ph, which cites them.
inline constexpr double JUPITER_A_AU = ph::A_JUPITER_AU;
inline constexpr double SATURN_A_AU = ph::A_SATURN_AU;

// A body on an orbit of the given elements, placed at perihelion on the +x axis
// with its velocity along +y.
//
// Eccentricity matters here and is not a refinement. A circular perturber's
// perihelion does not precess, so there is no g5 or g6 secular frequency, so
// the nu5 and nu6 resonances -- which are what make the 3:1, 4:1, 5:2 and 7:3
// commensurabilities chaotic -- are absent. See ph::E_JUPITER.
//
// Arg:
//   mass: body mass in solar masses.
//   a_au: semimajor axis in AU; must be > 0.
//   e: eccentricity; must satisfy 0 <= e < 1.
// Returns:
//   The body, with its acceleration left at zero -- the simulation computes it.
inline Planet body_at_perihelion(double mass, double a_au, double e)
{
    Planet body;
    body.mass = mass;
    body.r = Vec3{a_au * (1.0 - e), 0.0, 0.0};
    body.v = Vec3{0.0,
                  std::sqrt(ph::G * ph::M_Sun / a_au * (1.0 + e) / (1.0 - e)),
                  0.0};
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
// other here. The test particles feel every body, but with circular orbits that
// is not enough for the secular resonances the gaps are attributed to: nu5 and
// nu6 exist because the planetary perihelia precess, and a circular orbit's
// perihelion does not. Pass eccentric = true for the model the literature
// actually uses.
//
// Args:
//   with_saturn: add Saturn.
//   eccentric: use the real eccentricities from ph. False reproduces the
//     circular runs this project started with, which are worth keeping as a
//     control -- they are the model that does NOT produce the higher-order
//     gaps.
// Returns:
//   Jupiter, and Saturn if asked for, with their accelerations already set.
inline std::vector<Planet> make_planets(bool with_saturn, bool eccentric)
{
    const double ej = eccentric ? ph::E_JUPITER : 0.0;
    const double es = eccentric ? ph::E_SATURN : 0.0;
    std::vector<Planet> planets{
        body_at_perihelion(ph::M_Jupiter, JUPITER_A_AU, ej)};
    if (with_saturn)
        planets.push_back(body_at_perihelion(ph::M_Saturn, SATURN_A_AU, es));
    sun_only_acceleration(planets);
    return planets;
}

// Puts a stored epoch's body positions and velocities back into this run's
// bodies, keeping the masses this build knows.
//
// The epoch format does not store masses -- they belong to the setup, not to the
// state -- so a resumed run takes them from make_planets() and only the state
// from the file. The acceleration is recomputed, because the format does not
// store that either; it is a function of the positions.
//
// Args:
//   planets: this run's bodies, with masses; positions and velocities are
//     overwritten and the acceleration is refilled.
//   stored: the bodies as read back from an epoch file; their masses are
//     ignored.
// Returns:
//   Nothing.
// Throws:
//   std::runtime_error if the counts differ, which means the epoch came from a
//   run with a different set of bodies and cannot be continued here.
inline void restore_planets(std::vector<Planet> &planets,
                            const std::vector<Planet> &stored)
{
    if (planets.size() != stored.size())
        throw std::runtime_error(
            "the epoch holds " + std::to_string(stored.size()) +
            " bodies but this run has " + std::to_string(planets.size()) +
            "; resume with the same saturn setting");
    for (std::size_t b = 0; b < planets.size(); ++b)
    {
        planets[b].r = stored[b].r;
        planets[b].v = stored[b].v;
    }
    sun_only_acceleration(planets);
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
