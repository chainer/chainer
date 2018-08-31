#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"

namespace chainerx {
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
}  // namespace chainerx
