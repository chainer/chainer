#pragma once

#include <memory>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    static constexpr const char* kDefaultName = "cuda";

    std::string GetName() const override;

    int GetDeviceCount() const override;

private:
    std::unique_ptr<Device> CreateDevice(int index) override;
};

}  // namespace cuda
}  // namespace xchainer
