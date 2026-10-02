#pragma once

#include <cstddef>
#include <cstdint>

// The unit system and the constants derived in it.
//
// Included by both builds, so the constants have exactly one copy. Depends on
// nothing else in the project.

namespace ph
{
// We will set M_sun = AU = day = 1
constexpr double M_Sun = 1;                         // Sun mass
constexpr double M_Jupiter = 0.000954500308024994;  // Jupiter mass
constexpr double M_Saturn = 0.00028564603527829116; // Saturn mass

// One time step in days, overridable at compile time so several step sizes can
// be built from one source tree. 10 is the value the README's results used.
#ifndef PH_DT
#define PH_DT 10
#endif
constexpr double dt = PH_DT;

// G in AU^3 / (Msun * dt^2). The step is folded in, so every quantity in the
// project is per step rather than per day.
constexpr double G = 0.00029592190063820864 * dt * dt;

constexpr std::int32_t separation = 1024;

// Perihelion below which a test particle counts as removed, in AU. 1.524 is
// Mars's semimajor axis.
constexpr double REMOVE_BELOW_AU = 1.524;

// The largest number of massive bodies the force kernels are built for. The CPU
// kernel unrolls to this count with a template; the GPU kernel loops at run time
// and needs only the array bound.
constexpr std::size_t MAX_BODIES = 5;

// The giant planets' orbits: semimajor axis in AU, eccentricity dimensionless.
// J2000 mean elements from the JPL planetary fact sheet
// (https://nssdc.gsfc.nasa.gov/planetary/factsheet/), to four decimals.
//
// A circular perturber has a fixed perihelion, so its secular forcing carries no
// g5 or g6 frequency and the nu5 and nu6 resonances do not exist. Those are what
// make the 3:1, 4:1, 5:2 and 7:3 commensurabilities chaotic (Morbidelli & Moons
// 1993; Moons & Morbidelli 1995).
constexpr double A_JUPITER_AU = 5.2028;
constexpr double E_JUPITER = 0.0489;
constexpr double A_SATURN_AU = 9.5388;
constexpr double E_SATURN = 0.0565;
} // namespace ph
