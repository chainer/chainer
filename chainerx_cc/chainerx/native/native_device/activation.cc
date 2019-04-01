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
    const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
    const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
            T x2;
            T pos;
        };
        Elementwise<const T, const T, T>(Impl{static_cast<T>(x2), static_cast<T>(pos)}, x1_cast, neg_cast, out);
    });
}

void NativeDevice::IfGreaterElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
    const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 > x2 ? pos : neg; }
            T x2;
            T pos;
        };
        Elementwise<const T, const T, T>(Impl{static_cast<T>(x2), static_cast<T>(pos)}, x1_cast, neg_cast, out);
    });
}

void NativeDevice::Tanh(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Tanh(x); }
        };
        Elementwise<const T, T>(Impl{}, x_cast, out);
    });
}

void NativeDevice::Sin(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Sin(x); }
        };
        Elementwise<const T, T>(Impl{}, x_cast, out);
    });
}

void NativeDevice::Cos(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Cos(x); }
        };
        Elementwise<const T, T>(Impl{}, x_cast, out);
    });
}

}  // namespace native
}  // namespace chainerx
