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

# The GPU port is deliberately separate: it includes nothing from cpp/. The CPU
# build is the reference the GPU build is checked against, so it stays untouched
# until the two agree.
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
  # 20, not 17: main.cu uses std::numbers. The CPU target above is on 23; nvcc
  # 12.0 does not implement all of 23, so the two sides differ here.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # The smallest complete CUDA program in this project's shape. Build and run
  # this first: if it works, the toolchain is fine, and any later failure is in
  # the code rather than in the setup.
  add_executable(kirkwood_gpu_hello cuda/hello.cu)
  target_compile_options(kirkwood_gpu_hello PRIVATE -Wall -Wextra)

  add_executable(kirkwood_gpu cuda/main.cu)
  target_compile_options(kirkwood_gpu PRIVATE -Wall -Wextra )
  target_compile_options(kirkwood_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()