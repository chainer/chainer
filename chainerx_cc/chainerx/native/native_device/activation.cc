#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/numeric.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {

void NativeDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
            T x2;
            T pos;
        };
        Elementwise<const T, const T, T>(Impl{static_cast<T>(x2), static_cast<T>(pos)}, x1, neg, out);
    });
}

void NativeDevice::Tanh(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Tanh(x); }
        };
        Elementwise<const T, T>(Impl{}, x, out);
    });
}

}  // namespace native
}  // namespace chainerx
