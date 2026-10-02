#pragma once

// Exact two-body (Kepler) propagation, for the Wisdom-Holman drift.
//
// Universal variables rather than the classical eccentric-anomaly form: those
// need the eccentricity vector, which is degenerate as e -> 0, and our
// particles are drawn with e uniform on [0, 0.1]. This form needs only r, v and
// the Stumpff functions, and covers all conic sections.
//
// Units are the project's: AU, and steps rather than days, so mu carries
// G*dt^2 and a time argument counts steps.

#include <cmath>

#ifdef __CUDACC__
#define PH_HD __host__ __device__
#else
#define PH_HD
#endif

namespace ph
{

// sin and cos from one call, sinh and cosh from one exp.
//
// Args:
//   x: angle or argument, dimensionless.
//   s, c: written with sin(x) and cos(x), or sinh(x) and cosh(x).
// Returns:
//   Nothing.
PH_HD inline void sin_cos(double x, double &s, double &c)
{
#ifdef __CUDACC__
    ::sincos(x, &s, &c);
#else
    s = std::sin(x);
    c = std::cos(x);
#endif
}

PH_HD inline void sinh_cosh(double x, double &s, double &c)
{
    const double e = std::exp(x);
    const double inv = 1.0 / e;
    s = 0.5 * (e - inv);
    c = 0.5 * (e + inv);
}

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
        double sn = 0.0, cs = 0.0;
        sin_cos(s, sn, cs);
        c2 = (1.0 - cs) / psi;
        c3 = (s - sn) / (psi * s);
    }
    else if (psi < -1e-6)
    {
        const double s = std::sqrt(-psi);
        double sh = 0.0, ch = 0.0;
        sinh_cosh(s, sh, ch);
        c2 = (ch - 1.0) / (-psi);
        c3 = (sh - s) / ((-psi) * s);
    }
    else
    {
        // Series about zero: the closed forms above lose all their significant
        // digits here, both numerators vanishing like psi^2.
        c2 = 0.5 - psi / 24.0 + psi * psi / 720.0;
        c3 = 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0;
    }
}

// Advances one particle along its exact Kepler orbit about the origin.
//
// Only the central body acts here. The other bodies are the perturbation and
// belong to the kick.
//
// Args:
//   rx, ry, rz: position in AU; overwritten with the position after dt.
//   vx, vy, vz: velocity in AU per step; overwritten likewise.
//   mu: G * M_central in AU^3 / step^2.
//   dt: time to advance, in steps. May be negative, which retraces the orbit.
// Returns:
//   Nothing.
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

    // Newton iteration on the universal anomaly chi, from Vallado's starting
    // guess. The exit is a convergence test, not a fixed count.
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
        // dF/dchi is r(chi): the standard identity.
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
