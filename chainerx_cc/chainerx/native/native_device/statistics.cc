#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/statistics.h"
#include "chainerx/macro.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/native/reduce.h"
#include "chainerx/numeric.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/shape.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(AMax)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(AMin)
}  // namespace internal

namespace native {
namespace {

class NativeAMaxKernel : public AMaxKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        a.device().CheckDevicesCompatible(a, out);

        VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                T Identity() { return NumericLimits<T>::LowestOrInf(); }
                T MapIn(T in, int64_t /*index*/) { return in; }
                void Reduce(T next, T& accum) {
                    if (chainerx::IsNan(next) || accum < next) {
                        accum = next;
                    }
                }
                T MapOut(T accum) { return accum; }
            };
            Reduce<T, T>(a, axis, out, Impl{});
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AMaxKernel, NativeAMaxKernel);

class NativeAMinKernel : public AMinKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        a.device().CheckDevicesCompatible(a, out);

        VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                T Identity() { return NumericLimits<T>::MaxOrInf(); }
                T MapIn(T in, int64_t /*index*/) { return in; }
                void Reduce(T next, T& accum) {
                    if (chainerx::IsNan(next) || accum > next) {
                        accum = next;
                    }
                }
                T MapOut(T accum) { return accum; }
            };
            Reduce<T, T>(a, axis, out, Impl{});
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AMinKernel, NativeAMinKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
