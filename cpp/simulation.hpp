#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <omp.h>

#include "vec3.hpp"


namespace ph{
// We will set M_sun = AU = day = 1
constexpr double M_Sun = 1;                         // Sun mass
constexpr double M_Jupiter = 0.000954500308024994;  // Jupiter mass
constexpr double M_Saturn = 0.00028564603527829116; // Saturn mass

constexpr double dt = 10; // one time step in days

constexpr double G = 0.00029592190063820864 * dt * dt; // G in units of AU^3 / (Msun * dt^2)

constexpr std::int32_t separation = 1024;

// A test particle is removed once it comes inside this radius, in AU. 1.524 is
// Mars's semimajor axis, so reaching it means the orbit is Mars-crossing.
constexpr double REMOVE_BELOW_AU = 1.524;
}

// A massive body, in the heliocentric frame with the Sun pinned at the origin.
// Note: v and a are per *step*, not per day -- dt is folded into G above, so a
// velocity written here means AU per step and changes meaning if dt changes.
struct Planet
{
    double mass;
    Vec3 r, v, a;
};

// Massless test particles in SoA layout.
// Invariant: r, v and a always have the same length.
struct Particles
{
    SoAVec3 r, v, a;
    explicit Particles(std::size_t n = 0) : r(n), v(n), a(n) {};

    /// @brief Return the number of total particles
    /// @return
    std::size_t len() const { return r.x.size(); };

    /// @brief erase the i th element
    /// @param i index of the element to be erased
    void erase(std::size_t i)
    {
        r.erase(i);
        v.erase(i);
        a.erase(i);
    };
};

class Simulation
{
public:
    Simulation(
        std::vector<Planet> planets_,
        Particles particles_,
        uint64_t n_step);
    ~Simulation() {};

    // Return a reference to the internal arrays; no copy, no allocation.
    const Particles &get_particles() const { return this->particles; }
    const std::vector<Planet> &get_planets() const { return this->planets; }

    // Writes one epoch record to `path`. Creates or truncates the file; it
    // never appends, so one file holds exactly one epoch.
    // Args:
    //   path: output file path; its directory must already exist.
    // Returns: nothing.
    // Throws: std::runtime_error if the file cannot be opened, or if any
    //   write or the final flush fails. ofstream sets failbit rather than
    //   throwing, and a short write would leave a file that the reader
    //   silently misreads as a smaller epoch -- so this check is required.
    //
    // Layout, little-endian, no padding. `n_planet` and `n_alive` are the two
    // counts a reader needs; everything after them is fixed-width f64.
    //   u32 magic = 0x4B49524B ("KIRK"), u32 version = 1
    //   u64 n_planet
    //   f64 dt_days, f64 G
    //   u64 n_alive, u64 step
    //   f64 planet[6 * n_planet]    r.x r.y r.z v.x v.y v.z, interleaved
    //   f64 particle[6 * n_alive]   x[n] y[n] z[n] vx[n] vy[n] vz[n], i.e. six
    //                               consecutive blocks, matching the SoA
    //                               layout already in memory
    // `G` is in AU^3 / (M_sun * step^2), consistent with the per-step
    // velocities. The orbital-element formulas cancel the step unit, so a
    // reader never needs to convert to days.
    void dump(const std::string &path) const;

    // Simulated time since the start of the run, in years.
    double time() const;

    // Integrates to tot_step, advancing the whole system in place.
    // Args:
    //   dump_dir: directory receiving the `epoch_<step>.bin` files. Pass ""
    //     to run without writing anything -- the dumps share the wall clock
    //     with the integration, so a timing run must exclude them.
    //   epoch_every: requested number of steps between dumps. It is rounded
    //     down to a whole number of `separation` intervals, because a dump has
    //     to land on a cull boundary: culling is a discrete event, so splitting
    //     between two culls would record a different population than the same
    //     run at a different stride. Step 0 and the final step are always
    //     dumped, whatever the stride.
    // Returns: nothing.
    void run(const std::string &dump_dir, std::uint64_t epoch_every);

private:
    Particles particles;
    std::vector<Planet> planets; // vectors of planets
    std::uint64_t step;          // steps taken so far
    std::uint64_t tot_step;      // total steps to take
    void plant_acc();            // update planet acceleration
    void particle_acc();         // update particle acceleration
    void one_step();              // one step in simulation
    void forward(std::size_t step){for(std::size_t i=0;i<step;i++) one_step();}
    void cull();                 // delete all particles within the mars orbit
};
