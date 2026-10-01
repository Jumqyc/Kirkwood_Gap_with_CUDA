#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
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

// The state one epoch file describes. Mirrors Epoch in python/kirkwood_io.py,
// which is the other reader of the same bytes.
//
// Note what is NOT here: the acceleration. It is a function of the positions,
// so a resumed run recomputes it rather than storing it -- which is also why
// the file format did not have to change to support resuming.
struct EpochState
{
    std::uint64_t step = 0; // steps already taken, in units of the FILE's dt
    double dt_days = 0.0;   // the step size the velocities were written in
    double G = 0.0; // AU^3 / (M_sun * step^2), velocities are per step
    std::vector<Planet> planets;
    std::vector<double> rx, ry, rz, vx, vy, vz; // all of length n_alive
};

// Reads one epoch file back.
// Args:
//   path: an epoch file written by write_epoch().
// Returns:
//   The state it holds. Positions in AU, velocities in AU per step of THIS
//   build's dt -- read_epoch converts them if the file used another one.
//   carry zero mass, because the format does not store it -- the caller knows
//   what it put in and keeps its own copy.
// Throws:
//   std::runtime_error if the file cannot be opened, is shorter than its header
//   claims, or carries a magic or version this build does not know. A truncated
//   file is the expected aftermath of a hard kill, and reading it as a valid
//   shorter epoch would corrupt a resumed run silently.
inline EpochState read_epoch(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("read_epoch: cannot open " + path);

    const auto get = [&f](void *p, std::size_t bytes)
    { f.read(reinterpret_cast<char *>(p), static_cast<std::streamsize>(bytes)); };

    std::uint32_t magic = 0, version = 0;
    std::uint64_t n_planet = 0, n_alive = 0;
    EpochState s;

    get(&magic, sizeof(magic));
    get(&version, sizeof(version));
    if (magic != 0x4B49524Bu)
        throw std::runtime_error("read_epoch: not an epoch file: " + path);
    if (version != 1u)
        throw std::runtime_error("read_epoch: version " +
                                 std::to_string(version) +
                                 " is not supported: " + path);
    get(&n_planet, sizeof(n_planet));
    get(&s.dt_days, sizeof(s.dt_days));
    get(&s.G, sizeof(s.G));
    get(&n_alive, sizeof(n_alive));
    get(&s.step, sizeof(s.step));

    std::vector<double> body(6 * n_planet);
    get(body.data(), body.size() * sizeof(double));
    s.planets.resize(n_planet);
    for (std::size_t b = 0; b < n_planet; ++b)
    {
        s.planets[b].r = Vec3{body[6 * b + 0], body[6 * b + 1], body[6 * b + 2]};
        s.planets[b].v = Vec3{body[6 * b + 3], body[6 * b + 4], body[6 * b + 5]};
        s.planets[b].a = Vec3{0.0, 0.0, 0.0};
        s.planets[b].mass = 0.0;
    }

    for (std::vector<double> *block : {&s.rx, &s.ry, &s.rz, &s.vx, &s.vy, &s.vz})
    {
        block->resize(n_alive);
        get(block->data(), block->size() * sizeof(double));
    }

    // A short read sets failbit, which is how a file cut off mid-write gets
    // caught here instead of being accepted as a smaller epoch.
    if (!f)
        throw std::runtime_error("read_epoch: truncated or short: " + path);

    // Velocities are in AU per step, and a step is dt days -- so the same number
    // means a different speed in a build with a different dt, by exactly
    // dt_this_build / dt_file, NOT the other way round -- a velocity is AU per step,
    // so going to a longer step multiplies it. Resuming without this is silent: the
    // dt = 2 binary reading a dt = 10 file would run everything at a fifth of its
    // speed and the orbit would simply be wrong. Accelerations are not stored,
    // and the semimajor axis is derived from the state, so the velocities are the
    // only thing that has to move.
    const double to_this_step = ph::dt / s.dt_days;
    if (to_this_step != 1.0)
    {
        // The step count is in the same unit and needs the same conversion, or
        // the resumed run's clock jumps: 2000 steps of 2 days is 11 years, and
        // read as 2000 steps of 10 days it becomes 55. The dynamics are fine
        // either way, but every time the run reports would be wrong.
        s.step = static_cast<std::uint64_t>(
            static_cast<double>(s.step) / to_this_step + 0.5);
        for (std::vector<double> *block : {&s.vx, &s.vy, &s.vz})
            for (double &value : *block)
                value *= to_this_step;
        for (Planet &p : s.planets)
            p.v = Vec3{p.v.x * to_this_step, p.v.y * to_this_step,
                       p.v.z * to_this_step};
    }
    return s;
}

// Finds the highest-numbered epoch_<step>.bin in a directory.
// Args:
//   dir: directory to scan.
// Returns:
//   The path, or an empty string if the directory holds no epoch files, in
//   which case the caller should start from the beginning.
inline std::string newest_epoch(const std::string &dir)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(dir, ec))
        return {};

    // Ordered by the simulated time, not by the step number in the file name.
    // Those are the same thing only while the step size never changes: a run
    // continued across a dt transition leaves a dt=2 epoch_2000 next to a dt=10
    // epoch_450, and by number alone the older one looks newer, so every restart
    // would resume from it and never move.
    double best_time = -1.0;
    std::string best;
    for (const fs::directory_entry &e : fs::directory_iterator(dir, ec))
    {
        const std::string name = e.path().filename().string();
        if (name.rfind("epoch_", 0) != 0 || e.path().extension() != ".bin")
            continue;
        // Only the header is needed for dt_days and step, and it is 96 bytes.
        std::ifstream f(e.path(), std::ios::binary);
        if (!f)
            continue;
        std::uint32_t head[2] = {};
        std::uint64_t n_planet = 0, step = 0, n_alive = 0;
        double dt_days = 0.0, G = 0.0;
        f.read(reinterpret_cast<char *>(head), sizeof(head));
        f.read(reinterpret_cast<char *>(&n_planet), sizeof(n_planet));
        f.read(reinterpret_cast<char *>(&dt_days), sizeof(dt_days));
        f.read(reinterpret_cast<char *>(&G), sizeof(G));
        f.read(reinterpret_cast<char *>(&n_alive), sizeof(n_alive));
        f.read(reinterpret_cast<char *>(&step), sizeof(step));
        if (!f || head[0] != 0x4B49524Bu)
            continue;
        const double when = static_cast<double>(step) * dt_days;
        if (when > best_time)
        {
            best_time = when;
            best = e.path().string();
        }
    }
    return best;
}
