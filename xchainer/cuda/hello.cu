#include "xchainer/cuda/hello.h"

#include <cstdio>

#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

__global__ void HelloCuda() { printf("Hello, CUDA!\n"); }

void HelloCpu() { printf("Hello, World!\n"); }

void Hello() {
    Device device = GetCurrentDevice();
    if (device == Device{"cuda"}) {
        HelloCuda<<<1, 1>>>();
        CheckError(cudaDeviceSynchronize());
    } else {
        HelloCpu();
    }
}

}  // namespace cuda
}  // namespace xchainer
