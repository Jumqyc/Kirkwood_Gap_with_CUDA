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
