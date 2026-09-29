#include <cmath>
#include <cstdint>
#include <vector>


// We will set M_sun = AU = day = 1
constexpr double M_Sun = 1;
constexpr double M_Jupiter = 0.000954500308024994;
constexpr double M_Saturn = 0.00028564603527829116;

constexpr double dt = 1; // in days

constexpr double G = 0.00029592190063820864 * dt * dt; 
// G in units of AU^3 / (Msun * dt^2)

struct SoAVec3 {
    std::vector<double> x, y, z;
    explicit SoAVec3(std::size_t n = 0) : x(n), y(n), z(n) {}
};

struct Particles {
    SoAVec3 r, v, a;
    explicit Particles(std::size_t n = 0) : r(n), v(n), a(n) {}
};

struct Vec3
{
    double x,y,z;
};

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
    Particles get_particles(){return this->particles;};
    std::vector<Planet> get_planets(){return this->planets;};
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
