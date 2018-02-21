#include "xchainer/native_backend.h"

#include "xchainer/native_device.h"

namespace xchainer {

std::string NativeBackend::GetName() const { return "native"; }

// TODO(sonots): Returns number of CPU cores
int NativeBackend::GetDeviceCount() const { return 4; }

std::unique_ptr<Device> NativeBackend::CreateDevice(int index) { return std::make_unique<NativeDevice>(*this, index); }

}  // namespace xchainer
