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

RuntimeError::RuntimeError(cudaError_t error) : XchainerError(BuildErrorMessage(error)), error_(error) {}

void CheckError(cudaError_t error) {
    if (error != cudaSuccess) {
        Throw(error);
    }
}

void Throw(cudaError_t error) { throw RuntimeError(error); }

}  // namespace cuda
}  // namespace xchainer
