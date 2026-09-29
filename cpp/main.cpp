#include "simulation.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <string>


double energy(const Planet &p){
    double e = 0;
    e += 0.5 * p.mass * (
        p.v.x*p.v.x + p.v.y*p.v.y + p.v.z*p.v.z
    );
    e -= G*M_Sun*p.mass / std::sqrt(p.r.x*p.r.x+p.r.y*p.r.y+p.r.z*p.r.z);
    return e;
}


int main(int argc, char **argv){
    if (argc < 3){
        std::cerr << "usage: " << argv[0] << " <n_particles>  <n_steps>\n";
        return 1;
    }


    Planet Jupiter;
    Jupiter.mass = M_Jupiter;
    Jupiter.r = Vec3({14,0,0});
    Jupiter.v = Vec3({0,std::sqrt(G*M_Sun/14),0});
    Jupiter.a = Vec3({0,0,0});
    std::vector<Planet> planets;
    planets.push_back(Jupiter); 

    
    std::int64_t n_particle = std::stoi(argv[1]);
    Particles particles(n_particle);
    std::int64_t n_step = std::stoi(argv[2]);

    const std::uint64_t seed = 114514;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * std::numbers::pi);
    std::uniform_real_distribution<double> radius(2.0,3.5);
    std::uniform_real_distribution<double> eccentricity(0,0.1);

    for (std::size_t i = 0; i < (std::size_t)n_particle; ++i) {
        double th = angle(rng);
        double r = radius(rng);
        const double e = eccentricity(rng);
      
        double v = std::sqrt(G*M_Sun/r * (1+e)/(1-e));
        r *= (1-e);
        particles.r.x[i] =  r * std::cos(th);
        particles.r.y[i] =  r * std::sin(th);
        particles.r.z[i] =  0.0;
        particles.v.x[i] = -v * std::sin(th);
        particles.v.y[i] =  v * std::cos(th);
        particles.v.z[i] =  0.0;
    }

    auto s = Simulation(
        planets,
        particles,
        n_step
    );
    // double E0 = energy(Jupiter);
    auto t0 = std::chrono::steady_clock::now();
    s.run();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1-t0).count();

    std::cout << "Seed:  " << seed 
    << "  Total time use:  "  << sec 
    << "  Each particle needs  " << (sec / n_particle / n_step * 1e9)<< "ns" 
    << "threads" << omp_get_max_threads() 
    <<std::endl;
    

    return 0;
}

