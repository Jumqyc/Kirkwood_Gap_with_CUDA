#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "body.hpp"
#include "physics.hpp"

// The epoch file: the contract between the simulation and the Python reader.
//
// Both builds write it through this one function, so the layout is defined
// exactly once. python/kirkwood_io.py is the other end of it.
//
// Layout, little-endian, no padding:
//   u32 magic = 0x4B49524B ("KIRK"), u32 version = 1
//   u64 n_planet
//   f64 dt_days, f64 G
//   u64 n_alive, u64 step
//   f64 planet[6 * n_planet]    r.x r.y r.z v.x v.y v.z, interleaved
//   f64 particle[6 * n_alive]   x[n] y[n] z[n] vx[n] vy[n] vz[n], i.e. six
//                               consecutive blocks, matching the SoA layout the
//                               simulation already has in memory
//
// `G` is in AU^3 / (M_sun * step^2), consistent with the per-step velocities.
// The orbital-element formulas cancel the step unit, so a reader never has to
// convert to days.
//
// The state may be float or double in memory -- the GPU keeps positions and
// velocities as float -- but the file is always f64, so the reader has one case
// to handle and a float build cannot silently change the format.
//
// Args:
//   path: output file path; its parent directory must already exist.
//   step: integration steps taken, in units of dt days.
//   planets: massive bodies in the heliocentric frame.
//   n_alive: number of test particles.
//   rx..vz: the six coordinate arrays, each of length n_alive. Positions in AU,
//     velocities in AU per step.
// Returns:
//   Nothing. Creates or truncates the file; it never appends, so one file holds
//   exactly one epoch.
// Throws:
//   std::runtime_error if the file cannot be opened, or if any write or the
//   final flush fails. ofstream sets failbit rather than throwing, and a short
//   write leaves a file the reader misreads as a smaller epoch -- so the check
//   is required, not defensive.
template <typename T>
void write_epoch(const std::string &path, std::uint64_t step,
                 const std::vector<Planet> &planets, std::size_t n_alive,
                 const T *rx, const T *ry, const T *rz,
                 const T *vx, const T *vy, const T *vz)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("write_epoch: cannot open " + path);

    const std::uint32_t magic_version[2] = {0x4B49524Bu, 1u};
    const std::uint64_t n_planet = planets.size();
    const double constants[2] = {ph::dt, ph::G};
    f.write(reinterpret_cast<const char *>(magic_version), sizeof(magic_version));
    f.write(reinterpret_cast<const char *>(&n_planet), sizeof(n_planet));
    f.write(reinterpret_cast<const char *>(constants), sizeof(constants));

    const std::uint64_t record[2] = {n_alive, step};
    f.write(reinterpret_cast<const char *>(record), sizeof(record));

    const auto put = [&f](const double *p, std::size_t count)
    {
        f.write(reinterpret_cast<const char *>(p),
                static_cast<std::streamsize>(count * sizeof(double)));
    };

    // The bodies are AoS in memory, so they need one interleaving pass. There
    // are only a handful of them.
    std::vector<double> body(6 * planets.size());
    for (std::size_t b = 0; b < planets.size(); ++b)
    {
        body[6 * b + 0] = planets[b].r.x;
        body[6 * b + 1] = planets[b].r.y;
        body[6 * b + 2] = planets[b].r.z;
        body[6 * b + 3] = planets[b].v.x;
        body[6 * b + 4] = planets[b].v.y;
        body[6 * b + 5] = planets[b].v.z;
    }
    put(body.data(), body.size());

    for (const T *block : {rx, ry, rz, vx, vy, vz})
    {
        const std::vector<double> tmp(block, block + n_alive);
        put(tmp.data(), tmp.size());
    }

    // failbit is sticky, so this one check after the flush covers every write
    // above. The destructor would otherwise swallow the error and leave a
    // truncated file with a zero exit status.
    f.close();
    if (!f)
        throw std::runtime_error("write_epoch: write failed for " + path);
}
