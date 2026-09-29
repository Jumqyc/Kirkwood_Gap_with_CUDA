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
    Planet Jupiter;
    Jupiter.mass = M_Jupiter;
    Jupiter.r = Vec3({14,0,0});
    Jupiter.v = Vec3({0,4.697530e-03,0});
    Jupiter.a = Vec3({0,0,0});
    std::vector<Planet> planets;
    planets.push_back(Jupiter); 

    
    int n_particle = std::stoi(argv[1]);
    Particles particles(n_particle);
    int n_step = std::stoi(argv[2]);

    const std::uint64_t seed = 114514;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * std::numbers::pi);

    const double ring = 3.0;
    const double vc = std::sqrt(G * M_Sun / ring);

    for (std::size_t i = 0; i < (std::size_t)n_particle; ++i) {
      const double th = angle(rng);
      particles.r.x[i] =  ring * std::cos(th);
      particles.r.y[i] =  ring * std::sin(th);
      particles.r.z[i] =  0.0;
      particles.v.x[i] = -vc * std::sin(th);
      particles.v.y[i] =  vc * std::cos(th);
      particles.v.z[i] =  0.0;
  }


    auto s = Simulation(
        planets,
        particles,
        n_step
    );
    // double E0 = energy(Jupiter);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0 ; i < n_step; i++){
        s.forward();
    };
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1-t0).count();

    std::cout << "Seed:  " << seed << "  Total time use:  "  << sec << "  Each particle needs  " << (sec / n_particle / n_step * 1e9)<< "ns" <<std::endl;

    return 0;
}

