#include "xchainer/cuda/cuda_backend.h"

#include <cuda_runtime.h>

#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

std::string CudaBackend::GetName() const { return kDefaultName; }

int CudaBackend::GetDeviceCount() const {
    int count = 0;
    CheckError(cudaGetDeviceCount(&count));
    return count;
}

std::unique_ptr<Device> CudaBackend::CreateDevice(int index) { return std::make_unique<CudaDevice>(*this, index); }

}  // namespace cuda
}  // namespace xchainer
