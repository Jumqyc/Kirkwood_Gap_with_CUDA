#include "simulation.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>


Simulation::Simulation(
    std::vector<Planet> planets_,
    Particles particles_,
    uint64_t n_step){
    this->planets = planets_;
    this->particles = particles_;
    this->tot_step = n_step;
    this->step   = 0;
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


// Upper bound on the number of massive bodies: the Sun plus the planets.
// Raising it means one more switch line in particle_acc.
constexpr std::size_t MAX_BODIES = 5;


namespace {

// Bodies live in two flat arrays: body_pos holds three coordinates each, at
// body_pos[3*B + {0,1,2}], and body_gm[B] holds G*mass.

// One body's contribution at particle i, accumulated into
// (accel_x, accel_y, accel_z). B is a compile-time constant, so
// body_pos[3*B+...] and body_gm[B] are resolved while compiling rather than
// looked up at run time.
template <std::size_t B>
inline void one_body(double& accel_x, 
                    double& accel_y, 
                    double& accel_z,
                    const double* __restrict body_pos, 
                    const double* __restrict body_gm,
                    const double* __restrict rx, 
                    const double* __restrict ry,
                    const double* __restrict rz, 
                    std::size_t i){
    const double dx = body_pos[3*B]   - rx[i];
    const double dy = body_pos[3*B+1] - ry[i];
    const double dz = body_pos[3*B+2] - rz[i];
    const double dist  = std::sqrt(dx*dx + dy*dy + dz*dz);
    const double gm_over_r3  = body_gm[B]/(dist*dist*dist);
    accel_x += gm_over_r3*dx;
    accel_y += gm_over_r3*dy;
    accel_z += gm_over_r3*dz;
}

// Expands to one call per body: straight-line code, no loop. A body loop here
// would become the innermost loop, and the vectorizer only vectorizes the
// innermost one -- that is what killed the earlier attempt.
template <std::size_t... B>
inline void all_bodies(std::index_sequence<B...>,
                       double& accel_x, 
                       double& accel_y, 
                       double& accel_z,
                       const double* __restrict body_pos, 
                       const double* __restrict body_gm,
                       const double* __restrict rx, 
                       const double* __restrict ry,
                       const double* __restrict rz, 
                       std::size_t i){
    ( one_body<B>(accel_x, 
                    accel_y, 
                    accel_z, 
                    body_pos, 
                    body_gm, 
                    rx, 
                    ry, 
                    rz, 
                    i), ... );
}

// NB bodies, fixed at compile time, so all_bodies unrolls completely.
// The expansion sits in its own function on purpose: written directly in the
// loop body below, GCC 13 refuses to vectorize the OpenMP SIMD loop.
template <std::size_t NB>
void gravity_all(double* __restrict ax, 
                 double* __restrict ay, 
                 double* __restrict az,
                 const double* __restrict rx, 
                 const double* __restrict ry, 
                 const double* __restrict rz,
                 std::size_t n, 
                 const double* __restrict body_pos, 
                 const double* __restrict body_gm){
    #pragma omp parallel for simd
    for (std::size_t i = 0; i < n; ++i){
        double accel_x = 0.0, accel_y = 0.0, accel_z = 0.0;
        all_bodies(std::make_index_sequence<NB>{}, accel_x, accel_y, accel_z, body_pos, body_gm, rx, ry, rz, i);
        ax[i] = accel_x;
        ay[i] = accel_y;
        az[i] = accel_z;
    }
}

} // namespace


void Simulation::particle_acc(){
    const std::size_t n = this->particles.len();

    double* __restrict ax = this->particles.a.x.data();
    double* __restrict ay = this->particles.a.y.data();
    double* __restrict az = this->particles.a.z.data();
    const double* __restrict rx = this->particles.r.x.data();
    const double* __restrict ry = this->particles.r.y.data();
    const double* __restrict rz = this->particles.r.z.data();

    // The Sun first, then the planets. Both are stack arrays, so this costs
    // nothing per step and allocates nothing.
    const std::size_t nb = 1 + this->planets.size();
    if (nb > MAX_BODIES) throw std::runtime_error("raise MAX_BODIES");
    std::array<double, 3*MAX_BODIES> body_pos{};
    std::array<double, MAX_BODIES>   body_gm{};
    body_gm[0] = G*M_Sun;
    for (std::size_t b = 0; b < this->planets.size(); ++b){
        const Planet& pl = this->planets[b];
        body_pos[3*(b+1)+0] = pl.r.x;
        body_pos[3*(b+1)+1] = pl.r.y;
        body_pos[3*(b+1)+2] = pl.r.z;
        body_gm[b+1] = G*pl.mass;
    }

    // Unrolling needs a compile-time body count, so the run-time count has to
    // be turned into one here. One line per supported count.
    switch (nb){
        case 1: gravity_all<1>(ax,ay,az,rx,ry,rz,n,body_pos.data(),body_gm.data()); break;
        case 2: gravity_all<2>(ax,ay,az,rx,ry,rz,n,body_pos.data(),body_gm.data()); break;
        case 3: gravity_all<3>(ax,ay,az,rx,ry,rz,n,body_pos.data(),body_gm.data()); break;
        case 4: gravity_all<4>(ax,ay,az,rx,ry,rz,n,body_pos.data(),body_gm.data()); break;
        case 5: gravity_all<5>(ax,ay,az,rx,ry,rz,n,body_pos.data(),body_gm.data()); break;
    }
};

// Removes the particles that have wandered inside Mars's orbit. Walking
// backwards matters: erase() shifts everything after i down by one.
void Simulation::cull(){
    constexpr double r2_max = REMOVE_BELOW_AU*REMOVE_BELOW_AU;
    for (std::size_t i = this->particles.len(); i-- > 0;){
        const double& x = this->particles.r.x[i];
        const double& y = this->particles.r.y[i];
        const double& z = this->particles.r.z[i];
        if (x*x + y*y + z*z < r2_max) this->particles.erase(i);
    }
}

void Simulation::run(){
    const std::uint64_t bursts = this->tot_step / separation;

    for (std::uint64_t s = 0; s < bursts; ++s){
        for (std::int32_t j = 0; j < separation; ++j) forward();
        this->cull();
    }

    // n_step is not necessarily a multiple of separation.
    for (std::uint64_t k = bursts*separation; k < this->tot_step; ++k) forward();
    this->cull();
}


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
