# The build type selects the optimization level; -march=native is always on.
#   cmake -S . -B build-o0 -DCMAKE_BUILD_TYPE=Debug           -> -O0 -g
#   cmake -S . -B build-o2 -DCMAKE_BUILD_TYPE=RelWithDebInfo  -> -O2 -g -DNDEBUG
#   cmake -S . -B build-o3 -DCMAKE_BUILD_TYPE=Release         -> -O3 -DNDEBUG
# -DNDEBUG compiles out assert(). A check that must always run goes behind an
# explicit if, not an assert.

cmake_minimum_required(VERSION 3.20)

project(kirkwood
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kirkwood
  cpp/main.cpp
  cpp/simulation.cpp)

# No -ffast-math: it permits floating-point reassociation, which breaks the
# bit-reproducibility that the later checks compare against.
target_compile_options(kirkwood PRIVATE
  -march=native
  -Wall
  -Wextra)
  find_package(OpenMP REQUIRED)
  target_link_libraries(kirkwood PRIVATE OpenMP::OpenMP_CXX)

# The two builds share cpp/: the unit system, the initial conditions, the
# command line and the epoch file format all live there, because a constant or a
# layout written down twice is one that eventually disagrees with itself. What
# is NOT shared is the integration itself -- cpp/simulation.cpp is the reference
# cuda/gpu_simulation.cu is checked against, and keeping them separate is what
# makes the check meaningful.
#
# Guarded with check_language so the CPU target above still configures on a
# machine with no CUDA toolkit.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)

  # 89 is Ada (RTX 40 series). Pinning the architecture embeds no PTX, which
  # keeps the build fast; the binary then runs only on this architecture.
  set(CMAKE_CUDA_ARCHITECTURES 89)
  # 20, not 17: the shared headers use std::numbers. The CPU target above is on
  # 23; nvcc 12.0 does not implement all of 23, so the two sides differ here.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_executable(kirkwood_gpu
    cuda/main.cu
    cuda/gpu_simulation.cu)
  target_compile_options(kirkwood_gpu PRIVATE -Wall -Wextra)

  # No --use_fast_math. It was tried and measured at 1.00x: the SASS is
  # byte-for-byte identical with and without it, because it mostly affects FP32
  # sqrt and division, and this kernel's hot spot is a plain rsqrtf, which it
  # does not touch. The FP64 force was what actually mattered and it was fixed
  # by writing float in the source.
endif()