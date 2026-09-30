#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "body.hpp"
#include "physics.hpp"
#include "vec3.hpp"

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
    // Throws: std::runtime_error on an I/O failure.
    // The layout is defined once, in cpp/epoch_file.hpp.
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
    void yoshida_step();         // one yoshida_step
    void leapfrog(double h);             // one leapfrog step in simulation
    void forward(std::size_t step){for(std::size_t i=0;i<step;i++) yoshida_step();}
    void cull();                 // delete all particles within the mars orbit
};
