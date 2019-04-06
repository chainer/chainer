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
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {
namespace {

template <typename T>
struct NativeUnaryOp {
    T (*func)(T);

    explicit NativeUnaryOp(T (*func)(T)) : func{func} {}

    inline void operator()(int64_t /*i*/, T x, T& out) { out = func(x); }
};

class NativeSinhOp : public SinhOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(NativeUnaryOp<T>{chainerx::Sinh}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(SinhOp, NativeSinhOp);

class NativeCoshOp : public CoshOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(NativeUnaryOp<T>{chainerx::Cosh}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(CoshOp, NativeCoshOp);

class NativeArcsinhOp : public ArcsinhOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(NativeUnaryOp<T>{chainerx::Arcsinh}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(ArcsinhOp, NativeArcsinhOp);

class NativeArccoshOp : public ArccoshOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(NativeUnaryOp<T>{chainerx::Arccosh}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(ArccoshOp, NativeArccoshOp);

}  // namespace
}  // namespace native
}  // namespace chainerx
