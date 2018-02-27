#include "xchainer/cuda/cuda_backend.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

std::unique_ptr<Device> CudaBackend::CreateDevice(int index) {
    int device_count = GetDeviceCount();
    if (index >= device_count) {
        throw std::out_of_range("The index number (= " + std::to_string(index) + ") is not less than the device count (= " +
                                std::to_string(device_count) + ')');
    }
    return std::make_unique<CudaDevice>(*this, index);
}

bool CudaBackend::SupportsTransfer(Device& src_device, Device& dst_device) {
    return &src_device.backend() == this && &dst_device.backend() == this;
}

}  // namespace cuda
}  // namespace xchainer
