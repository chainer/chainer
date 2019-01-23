#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/numeric.h"

namespace chainerx {
namespace native {

void NativeDevice::Sqrt(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Sqrt(x); }
        };
        Elementwise<const T, T>(Impl{}, x, out);
    });
}

void NativeDevice::IsNan(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, bool& out) { out = chainerx::IsNan(x); }
        };
        Elementwise<const T, bool>(Impl{}, x, out);
    });
}

void NativeDevice::IsInf(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, bool& out) { out = chainerx::IsInf(x); }
        };
        Elementwise<const T, bool>(Impl{}, x, out);
    });
}

}  // namespace native
}  // namespace chainerx
