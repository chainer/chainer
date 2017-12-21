#include "xchainer/cuda/cuda_runtime.h"

#include <sstream>
#include <string>

namespace xchainer {
namespace cuda {
namespace {

std::string BuildErrorMessage(cudaError_t error) {
    std::ostringstream os;
    os << cudaGetErrorName(error) << ":" << cudaGetErrorString(error);
    return os.str();
}

}  // namespace

CudaRuntimeError::CudaRuntimeError(cudaError_t error) : XchainerError(BuildErrorMessage(error)), error_(error) {}

void CudaCheckError(cudaError_t error) {
    if (error != cudaSuccess) {
        throw CudaRuntimeError(error);
    }
}

void CudaDeviceSynchronize() { CudaCheckError(cudaDeviceSynchronize()); }

}  // namespace cuda
}  // namespace xchainer
