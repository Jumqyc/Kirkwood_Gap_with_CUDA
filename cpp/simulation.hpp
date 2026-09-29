#pragma once

#include <cstdint>
#include <vector>

#include "vec3.hpp"


// We will set M_sun = AU = day = 1
constexpr double M_Sun = 1;
constexpr double M_Jupiter = 0.000954500308024994;
constexpr double M_Saturn = 0.00028564603527829116;

constexpr double dt = 1; // in days

constexpr double G = 0.00029592190063820864 * dt * dt;
// G in units of AU^3 / (Msun * dt^2)

// A massive body, in the heliocentric frame with the Sun pinned at the origin.
// Note: v and a are per *step*, not per day -- dt is folded into G above, so a
// velocity written here means AU per step and changes meaning if dt changes.
struct Planet
{
    double mass;
    Vec3 r,v,a;
};

class Simulation{
    public: 
    Simulation(
        std::vector<Planet> planets_,
        Particles particles_,
        uint64_t n_step
    );
    ~Simulation(){};

    // Return a reference to the internal arrays; no copy, no allocation.
    const Particles& get_particles() const {return this->particles;}
    const std::vector<Planet>& get_planets() const {return this->planets;}

    double time();

    
    void forward();

    private:
    Particles particles;
    std::vector<Planet> planets;
    std::uint64_t step;
    std::uint32_t n_particles;
    void plant_acc();
    void particle_acc();
};
