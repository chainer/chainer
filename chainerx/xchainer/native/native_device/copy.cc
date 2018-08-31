#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"

namespace chainerx {
namespace native {

void NativeDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T a, T& out) { out = a; }
        };
        Elementwise<const T, T>(Impl{}, a, out);
    });
}

void NativeDevice::AsType(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    auto do_astype = [&](auto in_pt, auto out_pt) {
        using InT = typename decltype(in_pt)::type;
        using OutT = typename decltype(out_pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, InT a, OutT& out) { out = static_cast<OutT>(a); }
        };
        Elementwise<const InT, OutT>(Impl{}, a, out);
    };
    VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
}

}  // namespace native
}  // namespace chainerx
