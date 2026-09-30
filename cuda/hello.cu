// The smallest complete CUDA program in this project's shape.
//
// Every CUDA program has the same four parts, and they are always in this
// order: allocate device memory, copy the inputs in, launch, copy the results
// back. The kernel in cuda/main.cu is the same four parts with a longer launch.
//
// Build: cmake --build <build-dir> --target kirkwood_gpu_hello
// Run:   ./<build-dir>/kirkwood_gpu_hello

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// A kernel runs on the GPU, once per thread. __global__ means "called from the
// host, executed on the device". Every pointer dereferenced inside must be a
// DEVICE pointer -- a host pointer here is an immediate crash.
__global__ void scale_add(double *v, const double *a, double k, int n)
{
    // threadIdx.x is this thread's index within its block; blockIdx.x is the
    // block's index within the grid. Together they give the global index.
    // The guard is mandatory: the grid is rounded up, so the last block almost
    // always has threads past the end of the array.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] += k * a[i];
}

// Every CUDA call returns an error code. Nothing throws, and a failed kernel
// launch does not stop the program -- it just silently leaves the array
// untouched. Wrapping every call is not optional; this is the single largest
// source of wasted time when starting out.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess)                                              \
        {                                                                     \
            std::fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__, __LINE__,  \
                         #call, cudaGetErrorString(err_));                    \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

int main()
{
    const int n = 1 << 20;
    const double k = 0.5;

    // Host memory and device memory are entirely separate. Nothing is shared,
    // and no pointer is valid on both sides.
    std::vector<double> h_v(n, 1.0), h_a(n, 2.0);
    double *d_v = nullptr, *d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    // 256 threads per block is the usual starting point. The grid then needs
    // one block per 256 elements, rounded up.
    const int block = 256;
    const int grid = (n + block - 1) / block;
    scale_add<<<grid, block>>>(d_v, d_a, k, n);

    // A launch is ASYNCHRONOUS: it returns on the host before the GPU has done
    // anything. Without the next two lines the copy below could read stale
    // data, and a launch failure would never be noticed.
    CUDA_CHECK(cudaGetLastError());      // "did the launch itself fail?"
    CUDA_CHECK(cudaDeviceSynchronize()); // "is it finished?"

    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (h_v[i] != 2.0)
            ++bad;
    std::printf("n=%d  mismatches=%d  (expected 0)\n", n, bad);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_v));
    return bad == 0 ? 0 : 1;
}
