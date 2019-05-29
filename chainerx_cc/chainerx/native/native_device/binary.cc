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

}  // namespace
}  // namespace native
}  // namespace chainerx
