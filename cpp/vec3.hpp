#pragma once

#include <cstddef>
#include <vector>


// A 3-vector in array-of-structs layout: 24 bytes, lives on the stack, no heap.
// Used for the massive bodies, of which there are only a handful.
struct Vec3
{
    double x,y,z;

    // Adds alpha * other into this, in place. No temporary, no allocation.
    
    void add_scaled(double alpha, const Vec3 &other){
        x += alpha * other.x;
        y += alpha * other.y;
        z += alpha * other.z;
    }
};

// Three parallel arrays of length n, so the force loop reads contiguous doubles
// and vectorizes. Used for the massless test particles.
struct SoAVec3 {
    std::vector<double> x, y, z;

    explicit SoAVec3(std::size_t n = 0) : x(n), y(n), z(n) {}

    void add_scaled(double alpha, const SoAVec3 &other){
        #pragma omp parallel for simd
        for (std::size_t i = 0; i < x.size(); ++i){
            x[i] += alpha * other.x[i];
            y[i] += alpha * other.y[i];
            z[i] += alpha * other.z[i];
        }
    }

    void erase(std::size_t i){
        x.erase(x.begin()+i);
        y.erase(y.begin()+i);
        z.erase(z.begin()+i);
    }
};

// Massless test particles in SoA layout.
// Invariant: r, v and a always have the same length.
struct Particles {
    SoAVec3 r, v, a;
    explicit Particles(std::size_t n = 0) : r(n), v(n), a(n) {};
    std::size_t len() const {return r.x.size();};
    void erase(std::size_t i){
        r.erase(i);
        v.erase(i);
        a.erase(i);};
};
