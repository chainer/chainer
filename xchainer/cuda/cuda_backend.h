#pragma once

#include <cublas_v2.h>

#include <memory>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "cuda";

    explicit CudaBackend(Context& context);
    ~CudaBackend() override;

    std::string GetName() const override;

    int GetDeviceCount() const override;

    bool SupportsTransfer(Device& src_device, Device& dst_device) override;

    cublasHandle_t cublas_handle() const { return cublas_handle_; }

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
    cublasHandle_t cublas_handle_;
};

}  // namespace cuda
}  // namespace xchainer
