#include "xchainer/native/native_device.h"

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/native/elementwise.h"

namespace xchainer {
namespace native {

void NativeDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1, x2, out);
    });
}

void NativeDevice::Greater(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 > x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1, x2, out);
    });
}

void NativeDevice::Not(const Array& x1, const Array& out) {
    CheckDevicesCompatible(x1, out);
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, bool& out) { out = !x1; }
        };
        Elementwise<const T, bool>(Impl{}, x1, out);
    });
}

}  // namespace native
}  // namespace xchainer
