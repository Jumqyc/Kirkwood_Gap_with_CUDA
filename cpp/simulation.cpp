#include "simulation.hpp"

#include <cmath>
#include <algorithm>


Simulation::Simulation(
    std::vector<Planet> planets_,
    Particles particles_,
    uint64_t n_step){
    this->planets = planets_;
    this->particles = particles_;
    this->step = 0;

    this->n_particles = this->particles.r.x.size();

    this->plant_acc(); 
    this->particle_acc();
};

double Simulation::time(){
    constexpr double step_duration = dt/365;
    return step_duration * this->step; // in years
};

void Simulation::plant_acc(){
    for (Planet &p: this->planets){
        double r = std::sqrt(
            p.r.x * p.r.x + p.r.y * p.r.y + p.r.z * p.r.z
        );
        double r3 = r*r*r;
        p.a.x = -G*M_Sun* p.r.x/r3;
        p.a.y = -G*M_Sun* p.r.y/r3;
        p.a.z = -G*M_Sun* p.r.z/r3;
    }
}

namespace {
void gravity_kernel(
    double* __restrict ax, 
    double* __restrict ay, 
    double* __restrict az,
    const double* __restrict rx, 
    const double* __restrict ry, 
    const double* __restrict rz,
    std::size_t n, 
    double px,
    double py, 
    double pz, 
    double gm){
    for (std::size_t i = 0; i < n; ++i){
        const double dx = px - rx[i], dy = py - ry[i], dz = pz - rz[i];
        const double r  = std::sqrt(dx*dx + dy*dy + dz*dz);
        const double r3 = r*r*r;
        ax[i] += gm*dx/r3;
        ay[i] += gm*dy/r3;
        az[i] += gm*dz/r3;
    }
}
}

void Particles::add_gravity(Vec3 const &pos, double mass){
    gravity_kernel( a.x.data(), 
                    a.y.data(), 
                    a.z.data(),
                    r.x.data(), 
                    r.y.data(), 
                    r.z.data(),
                    r.x.size(), 
                    pos.x, 
                    pos.y, 
                    pos.z, 
                    G*mass );
}

void Simulation::particle_acc(){
    std::fill(this->particles.a.x.begin(), this->particles.a.x.end(), 0.0);
    std::fill(this->particles.a.y.begin(), this->particles.a.y.end(), 0.0);
    std::fill(this->particles.a.z.begin(), this->particles.a.z.end(), 0.0);

    this->particles.add_gravity(Vec3({0,0,0}),1);
    for (Planet const &p: this->planets){
        this->particles.add_gravity(p.r, p.mass);
    };

};


void Simulation::forward(){
    // leapfrog

    for (Planet &p : this->planets){
        p.v.add_scaled(0.5,p.a);
        p.r.add_scaled(1,p.v);
    };
    particles.v.add_scaled(0.5,particles.a);
    particles.r.add_scaled(1,particles.v);

    this->plant_acc(); this->particle_acc();

    for (Planet &p : this->planets){
        p.v.add_scaled(0.5,p.a);
    };
    particles.v.add_scaled(0.5,particles.a);

    this->step += 1;

}