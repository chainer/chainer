#pragma once

#include <string>

#include "xchainer/backend.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    int GetDeviceCount() const override;

    Device& GetDevice(int index) override;

    std::string GetName() const override;

private:
    DeviceList devices_;
};

}  // namespace cuda
}  // namespace xchainer
