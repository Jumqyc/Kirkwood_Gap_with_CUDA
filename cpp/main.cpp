#include "simulation.hpp"

#include <iostream>

int main(){
    Planet Jupiter;
    Jupiter.mass = M_Jupiter;
    
    Jupiter.r = Vec3({14,0,0});
    Jupiter.v = Vec3({0,4.597530e-03,0});
    Jupiter.a = Vec3({0,0,0});

    Particles particles(1);
    particles.r.x[0] = -14;
    particles.v.y[0] = -0.01;


    std::vector<Planet> planets;
    planets.push_back(Jupiter); 
    auto s = Simulation(
        planets,
        particles,
        100
    );
    for (int i = 0 ; i < 400000; i++){
        s.forward();
        auto p = s.get_planets()[0];
        if (i%2000 == 0){
        std::cout<< std::sqrt(p.r.x*p.r.x + p.r.y*p.r.y) << std::endl;
    }
    }
    return 0;
}

