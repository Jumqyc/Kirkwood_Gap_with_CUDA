#pragma once

#include <cstddef>
#include <vector>


// A 3-vector in array-of-structs layout: 24 bytes, lives on the stack, no heap.
// Used for the massive bodies, of which there are only a handful.
struct Vec3
{
    double x,y,z;

    /// @brief self += alpha * other
    /// @param alpha the scalar
    /// @param other the vector to be added to self
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

    /// @brief self += alpha * other. 
    /// @param alpha the scale
    /// @param other the other vector to be added
    void add_scaled(double alpha, const SoAVec3 &other){
        #pragma omp parallel for simd
        for (std::size_t i = 0; i < x.size(); ++i){
            x[i] += alpha * other.x[i];
            y[i] += alpha * other.y[i];
            z[i] += alpha * other.z[i];
        }
    }

    /// @brief erase the ith element 
    /// @param i  index of the element to be erased
    void erase(std::size_t i){
        x.erase(x.begin()+i);
        y.erase(y.begin()+i);
        z.erase(z.begin()+i);
    }
};


