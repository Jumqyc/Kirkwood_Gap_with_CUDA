#pragma once

// Exact two-body (Kepler) propagation, for the Wisdom-Holman drift.
//
// In a kick-drift-kick scheme the drift is where the accuracy is decided. The
// leapfrog drift is a straight line, r += v*dt, which is wrong for a particle
// in the Sun's gravity well and worst near perihelion -- it is what forces our
// step size down to 20 days at e = 0.65. Replacing it with an exact Kepler
// advance removes that limit and lets dt be set by how fast the PERTURBATION
// changes instead.
//
// Universal variables, not the classical eccentric-anomaly form. The classical
// route needs the eccentricity vector, which is degenerate as e -> 0; our
// particles are drawn with e uniform on [0, 0.1], so e near zero is common and
// dividing by it would produce NaN on the very first step. Universal variables
// use only r, v and the Stumpff functions, and cover elliptic, parabolic and
// hyperbolic orbits in one formulation.
//
// Everything is in the project's units: AU, and steps rather than days, so mu
// carries G*dt^2 and a "time" argument is a number of steps.

#include <cmath>

#ifdef __CUDACC__
#define PH_HD __host__ __device__
#else
#define PH_HD
#endif

namespace ph
{

// The Stumpff functions c2 and c3, which stand in for (1-cos x)/x^2 and
// (x-sin x)/x^3 and stay finite as psi -> 0 where those would not.
//
// Args:
//   psi: the universal-variable argument, dimensionless; any sign.
//   c2, c3: written.
// Returns:
//   Nothing.
PH_HD inline void stumpff(double psi, double &c2, double &c3)
{
    if (psi > 1e-6)
    {
        const double s = std::sqrt(psi);
        c2 = (1.0 - std::cos(s)) / psi;
        c3 = (s - std::sin(s)) / (psi * s);
    }
    else if (psi < -1e-6)
    {
        const double s = std::sqrt(-psi);
        c2 = (std::cosh(s) - 1.0) / (-psi);
        c3 = (std::sinh(s) - s) / ((-psi) * s);
    }
    else
    {
        // Series about zero. The closed forms lose all their significant
        // digits here: both numerators vanish like psi^2.
        c2 = 0.5 - psi / 24.0 + psi * psi / 720.0;
        c3 = 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0;
    }
}

// Advances one particle along its exact Kepler orbit about the origin.
//
// The particle is one of many and feels only the central body here; the other
// bodies are the perturbation and belong to the kick, not the drift. Passing
// them in as well would double-count the central body, which produces a
// plausible-looking orbit that is simply a different one.
//
// Args:
//   rx, ry, rz: position in AU; overwritten with the position after dt.
//   vx, vy, vz: velocity in AU per step; overwritten likewise.
//   mu: G * M_central in AU^3 / step^2.
//   dt: time to advance, in steps. May be negative, which retraces the orbit.
// Returns:
//   Nothing.
// Throws:
//   Nothing. Callers that need to know about unbound orbits should check the
//   specific orbital energy themselves.
PH_HD inline void kepler_drift(double &rx, double &ry, double &rz,
                               double &vx, double &vy, double &vz,
                               double mu, double dt)
{
    const double r0 = std::sqrt(rx * rx + ry * ry + rz * rz);
    const double v2 = vx * vx + vy * vy + vz * vz;
    const double rv = rx * vx + ry * vy + rz * vz; // r . v, in AU^2/step

    const double sqrt_mu = std::sqrt(mu);
    const double alpha = 2.0 / r0 - v2 / mu; // 1/a, in 1/AU
    const double sigma0 = rv / sqrt_mu;      // dimensionless

    // Newton iteration on the universal anomaly chi. The starting guess is
    // Vallado's; for an ellipse it is within a few percent and three or four
    // steps converge. The count is fixed rather than convergence-tested
    // because a data-dependent loop diverges across a warp on the GPU, where
    // one thread needing an extra pass costs the whole warp that pass.
    double chi = sqrt_mu * std::abs(alpha) * dt;
    if (alpha > 0.0)
        chi = sqrt_mu * alpha * dt;
    for (int i = 0; i < 8; ++i)
    {
        const double psi = chi * chi * alpha;
        double c2 = 0.0, c3 = 0.0;
        stumpff(psi, c2, c3);

        const double r = chi * chi * c2 + sigma0 * chi * (1.0 - psi * c3)
                         + r0 * (1.0 - psi * c2);
        const double f = sigma0 * chi * chi * c2
                         + (1.0 - alpha * r0) * chi * chi * chi * c3
                         + r0 * chi - sqrt_mu * dt;
        // dF/dchi is r(chi) itself -- the standard identity, and the reason r is
        // computed here rather than only after convergence. Writing out the
        // derivative by hand instead gets one term wrong in a way that still
        // converges to about 1e-8, which is small enough to look like rounding.
        const double step = f / r;
        chi -= step;
        if (std::fabs(step) < 1e-13 * (1.0 + std::fabs(chi)))
            break;
    }

    const double psi = chi * chi * alpha;
    double c2 = 0.0, c3 = 0.0;
    stumpff(psi, c2, c3);

    // Lagrange f and g, and their time derivatives.
    const double f = 1.0 - chi * chi * c2 / r0;
    const double g = dt - chi * chi * chi * c3 / sqrt_mu;

    const double r_new = chi * chi * c2 + sigma0 * chi * (1.0 - psi * c3)
                         + r0 * (1.0 - psi * c2);

    const double fdot = sqrt_mu / (r0 * r_new) * chi * (psi * c3 - 1.0);
    const double gdot = 1.0 - chi * chi * c2 / r_new;

    const double nx = f * rx + g * vx;
    const double ny = f * ry + g * vy;
    const double nz = f * rz + g * vz;
    const double nvx = fdot * rx + gdot * vx;
    const double nvy = fdot * ry + gdot * vy;
    const double nvz = fdot * rz + gdot * vz;

    rx = nx; ry = ny; rz = nz;
    vx = nvx; vy = nvy; vz = nvz;
}

} // namespace ph

#undef PH_HD
