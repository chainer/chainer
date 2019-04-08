#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace native {

void NativeDevice::Square(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Square(x); }
        };
        Elementwise<const T, T>(Impl{}, x, out);
    });
}

void NativeDevice::Sqrt(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Sqrt(x); }
        };
        Elementwise<const T, T>(Impl{}, x_cast, out);
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

class NativeCeilOp : public CeilOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Ceil(x); }
            };
            Elementwise<const T, T>(Impl{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(CeilOp, NativeCeilOp);

}  // namespace native
}  // namespace chainerx
