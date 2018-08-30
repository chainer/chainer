#include "xchainer/native/native_device.h"

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/native/elementwise.h"

namespace xchainer {
namespace native {

void NativeDevice::Sqrt(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = std::sqrt(x); }
        };
        Elementwise<const T, T>(Impl{}, x, out);
    });
}

}  // namespace native
}  // namespace xchainer
