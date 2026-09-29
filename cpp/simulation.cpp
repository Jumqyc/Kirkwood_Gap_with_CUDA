#include "simulation.hpp"


Simulation::Simulation(
    std::vector<Planet> planets_,
        Particles particles_,
    uint64_t n_step
){
    this->planets = planets_;
    this->particles = particles_;
    this->step = 0;

    this->n_particles = this->particles.r.x.size();

};

double Simulation::time(){
    constexpr double step_duration = dt/365;
    return step_duration * this->step; // in years
};

void Simulation::plant_acc(){
    for (Planet p: this->planets){
        double r = std::sqrt(
            p.r.x * p.r.x + p.r.y * p.r.y + p.r.z * p.r.z
        );
        double r3 = r*r*r;
        p.a.x = G*M_Sun* p.r.x/r3;
        p.a.y = G*M_Sun* p.r.y/r3;
        p.a.z = G*M_Sun* p.r.z/r3;
    }
}

void Simulation::particle_acc(){
    for (uint32_t i = 0; i< this->n_particles ; i++){

        // sun gravity
        double r3 = std::sqrt(
            this->particles.r.x[i]*this->particles.r.x[i]
            + this->particles.r.y[i]*this->particles.r.y[i]
            + this->particles.r.z[i]*this->particles.r.z[i]
        );
        r3 = r3*r3*r3;

        this->particles.a.x[i] = -G*M_Sun * particles.r.x[i]/r3;
        this->particles.a.y[i] = -G*M_Sun * particles.r.y[i]/r3;
        this->particles.a.z[i] = -G*M_Sun * particles.r.z[i]/r3;

        // planet gravity

        for (Planet p : this->planets){
            double dx = p.r.x - this->particles.r.x[i];
            double dy = p.r.y - this->particles.r.y[i];
            double dz = p.r.z - this->particles.r.z[i];

            r3 = std::sqrt(dx*dx+dy*dy+dz*dz);
            r3 = r3*r3*r3;

            this->particles.a.x[i] += G*M_Sun * dx/r3;
            this->particles.a.y[i] += G*M_Sun * dy/r3;
            this->particles.a.z[i] += G*M_Sun * dz/r3;
        }
    };
};


void Simulation::forward(){
    // naive version:

    this->step += 1;

    for (Planet p : this->planets){
        p.r.x += p.v.x;
        p.r.y += p.v.y;
        p.r.z += p.v.z;

        p.v.x += p.a.x;
        p.v.y += p.a.y;
        p.v.z += p.a.z;
    }

    for (uint32_t i = 0; i < this->n_particles ; i++){
        particles.r.x[i] += particles.v.x[i];
        particles.r.y[i] += particles.v.y[i];
        particles.r.z[i] += particles.v.z[i];

        particles.v.x[i] += particles.a.x[i];
        particles.v.y[i] += particles.a.y[i];
        particles.v.z[i] += particles.a.z[i];
    }

}