#include "chainerx/native/native_device.h"

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/binary.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/routines/binary.h"
#include "chainerx/scalar.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BitwiseAnd)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BitwiseAndAS)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BitwiseOr)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BitwiseOrAS)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BitwiseXor)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(BitwiseXorAS)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LeftShiftAA)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LeftShiftAS)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LeftShiftSA)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(RightShiftAA)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(RightShiftAS)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(RightShiftSA)
}  // namespace internal

namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(BitwiseAndKernel, { out = x1 & x2; }, VisitIntegralDtype);

class NativeBitwiseAndASKernel : public BitwiseAndASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = x1 & x2; }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(BitwiseAndASKernel, NativeBitwiseAndASKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(BitwiseOrKernel, { out = x1 | x2; }, VisitIntegralDtype);

class NativeBitwiseOrASKernel : public BitwiseOrASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = x1 | x2; }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(BitwiseOrASKernel, NativeBitwiseOrASKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(BitwiseXorKernel, { out = x1 ^ x2; }, VisitIntegralDtype);

class NativeBitwiseXorASKernel : public BitwiseXorASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = x1 ^ x2; }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(BitwiseXorASKernel, NativeBitwiseXorASKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(LeftShiftAAKernel, { out = x1 << x2; }, VisitShiftDtype);

class NativeLeftShiftASKernel : public LeftShiftASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitShiftDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = x1 << x2; }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(LeftShiftASKernel, NativeLeftShiftASKernel);

class NativeLeftShiftSAKernel : public LeftShiftSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        VisitShiftDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x2, T& out) { out = x1 << x2; }
                T x1;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(LeftShiftSAKernel, NativeLeftShiftSAKernel);

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(RightShiftAAKernel, { out = x1 >> x2; }, VisitShiftDtype);

class NativeRightShiftASKernel : public RightShiftASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        VisitShiftDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T& out) { out = x1 >> x2; }
                T x2;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(RightShiftASKernel, NativeRightShiftASKernel);

class NativeRightShiftSAKernel : public RightShiftSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        VisitShiftDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x2, T& out) { out = x1 >> x2; }
                T x1;
            };
            Elementwise<const T, T>(Impl{static_cast<T>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(RightShiftSAKernel, NativeRightShiftSAKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
