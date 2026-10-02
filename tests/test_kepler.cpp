// Checks ph::kepler_drift against answers that are known in closed form.
//
// No reference integrator is involved. Each case is either an exact analytic
// solution (the circular orbit) or a quantity the two-body problem conserves
// exactly (energy, angular momentum), so a failure here is a failure of the
// drift and not of some other code it is being compared to.
//
// Build: cmake --build <build-dir> --target test_kepler
// Run:   ./<build-dir>/test_kepler

#include "../cpp/kepler.hpp"
#include "../cpp/physics.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{

int failures = 0;

// Reports one check. Args: what it was, the measured error, the tolerance.
void check(const char *what, double error, double tolerance)
{
    const bool ok = error <= tolerance;
    if (!ok)
        ++failures;
    std::printf("  %-46s %.3e  (tol %.0e)  %s\n", what, error, tolerance,
                ok ? "ok" : "FAIL");
}

// A circular orbit of radius a about the origin: position and velocity after
// any time are known exactly, so this is an absolute reference rather than a
// consistency check.
void circular()
{
    const double a = 2.5;
    const double mu = ph::G * ph::M_Sun;
    const double v = std::sqrt(mu / a);
    const double n = std::sqrt(mu / (a * a * a)); // radians per step

    double rx = a, ry = 0.0, rz = 0.0;
    double vx = 0.0, vy = v, vz = 0.0;

    const int periods = 500;
    const double total = 2.0 * M_PI / n * periods;
    const int steps = 20000;
    const double dt = total / steps;
    for (int i = 0; i < steps; ++i)
        ph::kepler_drift(rx, ry, rz, vx, vy, vz, mu, dt);

    const double angle = n * total;
    const double ex = a * std::cos(angle), ey = a * std::sin(angle);
    const double evx = -v * std::sin(angle), evy = v * std::cos(angle);

    const double dr = std::hypot(rx - ex, ry - ey);
    const double dv = std::hypot(vx - evx, vy - evy);
    std::printf("\ncircular orbit, radius %.1f AU, %d periods in %d steps\n",
                a, periods, steps);
    check("position error after 500 periods (AU)", dr, 1e-9);
    check("velocity error after 500 periods (AU/step)", dv, 1e-9);
}

// The two-body energy and angular momentum are conserved exactly; the drift
// must not change them. An eccentric orbit is used because that is where the
// leapfrog drift this replaces does its worst.
void conservation(double e)
{
    const double a = 2.5;
    const double mu = ph::G * ph::M_Sun;
    const double rp = a * (1.0 - e);
    const double vp = std::sqrt(mu * (1.0 + e) / rp);

    double rx = rp, ry = 0.0, rz = 0.0;
    double vx = 0.0, vy = vp, vz = 0.0;

    const double energy0 = 0.5 * (vx * vx + vy * vy) - mu / rx;
    const double angmom0 = rx * vy - ry * vx;

    const double period = 2.0 * M_PI * std::sqrt(a * a * a / mu);
    const int steps = 2000;
    const double dt = period / steps;
    for (int i = 0; i < steps; ++i)
        ph::kepler_drift(rx, ry, rz, vx, vy, vz, mu, dt);

    const double energy1 = 0.5 * (vx * vx + vy * vy)
                           - mu / std::sqrt(rx * rx + ry * ry);
    const double angmom1 = rx * vy - ry * vx;

    std::printf("\neccentric orbit, a = %.1f AU, e = %.2f, one period in %d steps\n",
                a, e, steps);
    check("relative energy change", std::fabs(energy1 - energy0) / std::fabs(energy0),
          1e-12);
    check("relative angular momentum change",
          std::fabs(angmom1 - angmom0) / std::fabs(angmom0), 1e-12);
    check("closes after one period (AU)",
          std::hypot(rx - rp, ry), 1e-9);
}

// e = 0 is the case the classical eccentric-anomaly formulation cannot handle,
// and it is common in our initial conditions: e is drawn uniform on [0, 0.1],
// so some particles start essentially circular. Nothing here should be NaN.
void near_circular()
{
    const double a = 2.5;
    const double mu = ph::G * ph::M_Sun;
    const double v = std::sqrt(mu / a) * 0.999999; // almost, not exactly, circular

    double rx = a, ry = 0.0, rz = 0.0;
    double vx = 0.0, vy = v, vz = 0.0;
    for (int i = 0; i < 1000; ++i)
        ph::kepler_drift(rx, ry, rz, vx, vy, vz, mu, 1.0);

    const bool finite = std::isfinite(rx) && std::isfinite(ry)
                        && std::isfinite(vx) && std::isfinite(vy);
    const double r = std::hypot(rx, ry);
    std::printf("\nquasi-circular orbit, e ~ 1e-6, 1000 steps of 1\n");
    check("no NaN or inf (0 = clean)", finite ? 0.0 : 1.0, 0.0);
    check("radius stayed near a (AU)", std::fabs(r - a), 1e-6);
}

// Time reversal: the drift is an exact solution, so stepping forward and back
// must return to the same place. This is the property the Wisdom-Holman split
// rests on.
void reversibility()
{
    const double a = 2.5, e = 0.6;
    const double mu = ph::G * ph::M_Sun;
    const double rp = a * (1.0 - e);
    const double vp = std::sqrt(mu * (1.0 + e) / rp);

    const double rx0 = rp, ry0 = 0.0, rz0 = 0.0;
    const double vx0 = 0.0, vy0 = vp, vz0 = 0.0;
    double rx = rx0, ry = ry0, rz = rz0;
    double vx = vx0, vy = vy0, vz = vz0;

    for (int i = 0; i < 500; ++i)
        ph::kepler_drift(rx, ry, rz, vx, vy, vz, mu, 3.0);
    for (int i = 0; i < 500; ++i)
        ph::kepler_drift(rx, ry, rz, vx, vy, vz, mu, -3.0);

    std::printf("\nforward 500 steps of 3, then back 500\n");
    check("returns to the start (AU)",
          std::hypot(std::hypot(rx - rx0, ry - ry0), rz - rz0), 1e-10);
    check("velocity returns too (AU/step)",
          std::hypot(std::hypot(vx - vx0, vy - vy0), vz - vz0), 1e-10);
}

} // namespace

int main()
{
    std::printf("kepler_drift, units AU and steps, mu = G * M_Sun\n");
    circular();
    conservation(0.05);
    conservation(0.60);
    near_circular();
    reversibility();
    std::printf("\n%s\n", failures == 0 ? "all checks passed"
                                        : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
