#include "xchainer/cuda/cuda_backend.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

int CudaBackend::GetDeviceCount() const {
    int count = 0;
    CheckError(cudaGetDeviceCount(&count));
    return count;
}

Device& CudaBackend::GetDevice(int index) {
    if (index < 0) {
        throw std::out_of_range("The index number must be greater than or equal to 0");
    }
    if (index >= GetDeviceCount()) {
        throw std::out_of_range("The index number must be smaller than the number of available devices");
    }
    if (!devices_.HasDevice(index)) {
        devices_.AddDevice(std::make_unique<CudaDevice>(*this, index));
    }
    return devices_.GetDevice(index);
}

std::string CudaBackend::GetName() const { return "cuda"; }

}  // namespace cuda
}  // namespace xchainer
