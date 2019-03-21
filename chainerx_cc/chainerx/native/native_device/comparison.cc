#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"

namespace chainerx {
namespace native {

void NativeDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
    const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
    const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
    });
}

void NativeDevice::NotEqual(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
    const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
    const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 != x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
    });
}

void NativeDevice::Greater(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
    const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
    const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 > x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
    });
}

void NativeDevice::GreaterEqual(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
    const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
    const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 >= x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
    });
}

void NativeDevice::LogicalNot(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, bool& out) { out = !x; }
        };
        Elementwise<const T, bool>(Impl{}, x, out);
    });
}

}  // namespace native
}  // namespace chainerx
