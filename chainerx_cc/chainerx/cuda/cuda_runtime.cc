#include "chainerx/cuda/cuda_runtime.h"

#include <sstream>
#include <string>

#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {
namespace {

std::string BuildErrorMessage(cudaError_t error) {
    std::ostringstream os;
    os << cudaGetErrorName(error) << ":" << cudaGetErrorString(error);
    return os.str();
}

}  // namespace

RuntimeError::RuntimeError(cudaError_t error) : ChainerxError(BuildErrorMessage(error)), error_(error) {}

void CheckCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        Throw(error);
    }
}

void Throw(cudaError_t error) { throw RuntimeError(error); }

bool IsPointerCudaMemory(const void* ptr) {
    cudaPointerAttributes attr = {};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            return true;
        case cudaErrorInvalidValue:
            return false;
        default:
            CheckCudaError(status);
            break;
    }
    CHAINERX_NEVER_REACH();
}

bool IsPointerManagedMemory(const void* ptr) {
    cudaPointerAttributes attr = {};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
#if CUDART_VERSION < 10000
            return attr.isManaged != 0;
#else  // CUDART_VERSION
            return attr.type == cudaMemoryTypeManaged;
#endif  // CUDART_VERSION
        case cudaErrorInvalidValue:
            return false;
        default:
            CheckCudaError(status);
            break;
    }
    CHAINERX_NEVER_REACH();
}

}  // namespace cuda
}  // namespace chainerx
