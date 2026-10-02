#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../cpp/body.hpp"
#include "../cpp/physics.hpp"
#include "../cpp/vec3.hpp"

// Where every massive body is during every sub-step of one outer step, shaped
// the way the kernel wants it.
//
// Passed to the kernel BY VALUE. It is a couple of hundred bytes, kernel
// arguments travel with the launch, and that keeps a device copy out of the
// inner loop -- a per-step cudaMemcpy of even 200 bytes costs more than the
// arithmetic it feeds.
//
// Body 0 is the Sun, pinned at the origin, so its positions are all zero and
// the kernel needs no special case for it.
struct BodyTable
{
    float pos[3][ph::MAX_BODIES][3] = {}; // [sub-step][body][x y z], in AU
    float gm[ph::MAX_BODIES] = {};        // G*M, in AU^3 / step^2
    int nb = 0;                           // bodies actually filled in
};

// The GPU half of the simulation.
//
// Its public interface mirrors the CPU Simulation, so the two are
// interchangeable on the command line. Inside they are nothing alike: the CPU
// version splits every step into a dozen OpenMP regions, this one puts a whole
// outer step in one thread with no barrier anywhere.
//
// Owns nine cudaMalloc'd arrays. The copy operations are deleted on purpose:
// the default copy would duplicate the nine pointers, and the two destructors
// would then cudaFree the same block twice.
//
// Invariant: the device arrays always hold exactly n particles. Culling is not
// implemented here, so n never changes; `step` counts outer steps taken.
class GpuSimulation
{
public:
    // Args:
    //   planets: massive bodies in the heliocentric frame. Integrated on the
    //     host, because their motion does not depend on the test particles --
    //     that is what lets the kernel take their positions as arguments
    //     instead of a device array refreshed every step.
    //   n: number of test particles.
    //   rx..vz: initial positions in AU and velocities in AU per step, each of
    //     length n. Copied to the device and not modified. A resumed run passes
    //     state read back from an epoch file here instead of a fresh draw.
    //   n_step: total outer steps to take, in units of dt days.
    //   start_step: steps already taken before this object existed, i.e. the
    //     step number of the epoch the state came from. run() continues from
    //     start_step + 1.
    //   wisdom_holman: integrate with the Wisdom-Holman mapping rather than the
    //     Yoshida-4 leapfrog composition. The drift becomes an exact Kepler
    //     advance about the Sun and the kick carries only the other bodies, so
    //     the two are not interchangeable step for step -- WH takes far larger
    //     steps for the same accuracy, and the two describe the same physics
    //     from the same initial conditions only in the statistical sense.
    GpuSimulation(std::vector<Planet> planets, std::int64_t n,
                  const double *rx, const double *ry, const double *rz,
                  const double *vx, const double *vy, const double *vz,
                  std::uint64_t n_step, std::uint64_t start_step = 0,
                  bool wisdom_holman = false);
    ~GpuSimulation();

    GpuSimulation(const GpuSimulation &) = delete;
    GpuSimulation &operator=(const GpuSimulation &) = delete;

    // Integrates to n_step, writing one epoch file every `epoch_every` steps,
    // plus step 0 and the final step.
    // Args:
    //   dump_dir: directory for the epoch_<step>.bin files; "" writes nothing.
    //   epoch_every: steps between dumps. Unlike the CPU version this is not
    //     rounded to a cull interval, because there is no culling here. The two
    //     builds therefore write a different NUMBER of files for the same
    //     argument, though the final state at the final step is the same.
    // Returns: nothing.
    void run(const std::string &dump_dir, std::uint64_t epoch_every);

private:
    // Brings the state back from the device and writes one epoch file.
    void dump(const std::string &path) const;

    // Launches one outer step, given where the massive bodies are during each
    // of the three Yoshida sub-steps.
    void launch(const BodyTable &bodies);

    std::vector<Planet> planets_;

    // Positions and velocities are double. Making them float was tried and
    // measured: it is 0.75x SLOWER, not faster, and it costs two orders of
    // magnitude of accuracy for nothing. Whatever limits this kernel, halving
    // the state size and removing every conversion does not help it.
    double *rx_ = nullptr, *ry_ = nullptr, *rz_ = nullptr;
    double *vx_ = nullptr, *vy_ = nullptr, *vz_ = nullptr;

    // The accumulator is double as well. Only the arithmetic BETWEEN the state
    // and the accumulator is float -- that is the one place where it pays.
    double *ax_ = nullptr, *ay_ = nullptr, *az_ = nullptr;

    // Host-side staging for the dumps, allocated once and reused. Mutable
    // because dump() is const -- it does not change the simulation's state --
    // while these buffers are a cache on the way to the file, not state.
    mutable std::vector<double> h_rx_, h_ry_, h_rz_, h_vx_, h_vy_, h_vz_;

    bool wisdom_holman_ = false;
    std::int64_t n_ = 0;
    std::uint64_t step_ = 0;     // outer steps taken so far
    std::uint64_t tot_step_ = 0; // outer steps to take in total
};
