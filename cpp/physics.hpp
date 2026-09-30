#pragma once

#include <cstddef>
#include <cstdint>

// The unit system and the constants derived in it.
//
// This header exists so the CUDA port can include the constants without pulling
// in the CPU simulation, which needs OpenMP. A constant written down twice is a
// constant that will eventually disagree with itself, so there is exactly one
// copy and both builds read it from here.
//
// Nothing in here may depend on anything else in the project.

namespace ph
{
// We will set M_sun = AU = day = 1
constexpr double M_Sun = 1;                         // Sun mass
constexpr double M_Jupiter = 0.000954500308024994;  // Jupiter mass
constexpr double M_Saturn = 0.00028564603527829116; // Saturn mass

constexpr double dt = 10; // one time step in days

// G in units of AU^3 / (Msun * dt^2). The step size is folded in here, so every
// quantity below is per *step*, not per day.
constexpr double G = 0.00029592190063820864 * dt * dt;

constexpr std::int32_t separation = 1024;

// A test particle is removed once it comes inside this radius, in AU. 1.524 is
// Mars's semimajor axis, so reaching it means the orbit is Mars-crossing.
constexpr double REMOVE_BELOW_AU = 1.524;

// The largest number of massive bodies the force kernels are built for. The CPU
// kernel unrolls to this count with a template, so raising it means one more
// switch line in particle_acc(); the GPU kernel loops at run time and only
// needs the array bound. Going over is a thrown error, never a silent
// truncation.
constexpr std::size_t MAX_BODIES = 5;
} // namespace ph
