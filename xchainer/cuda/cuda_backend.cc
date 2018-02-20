#include "xchainer/cuda_backend.h"

#include <cuda_runtime.h>

#include "xchainer/cuda_device.h"

namespace xchainer {
namespace cuda {

int CudaBackend::GetDeviceCount() const {
    int count;
    CheckError(cudaGetDeviceCount(&count));
    return count;
}

Device& CudaBackend::GetDevice(int index) {
    if (!devices_.HasDevice(index)) {
        devices_.AddDevice(std::make_unique<CudaDevice>(*this, index));
    }
    return devices_.GetDevice(index);
}

std::string CudaBackend::GetName() const { return "cuda"; }

}  // namespace cuda
}  // namespace xchainer
