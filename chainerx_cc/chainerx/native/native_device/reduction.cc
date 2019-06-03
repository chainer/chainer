#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/reduction.h"
#include "chainerx/kernels/sorting.h"
#include "chainerx/macro.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/native/reduce.h"
#include "chainerx/numeric.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/routines/math.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {
namespace {

class NativeArgMaxKernel : public ArgMaxKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(std::all_of(axis.begin(), axis.end(), [&a](int8_t i) { return a.shape()[i] > 0; }));
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), false));
        a.device().CheckDevicesCompatible(a, out);

        VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                struct MaxAndArgMax {
                    T max;
                    int64_t argmax;
                };

                MaxAndArgMax Identity() { return {T{}, -1}; }
                MaxAndArgMax MapIn(T in, int64_t index) { return {in, index}; }
                void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
                    if (accum.argmax < 0 || accum.max < next.max) {
                        accum = next;
                    }
                }
                int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
            };
            Reduce<T, int64_t>(a, axis, out, Impl{});
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ArgMaxKernel, NativeArgMaxKernel);

class NativeArgMinKernel : public ArgMinKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(std::all_of(axis.begin(), axis.end(), [&a](int8_t i) { return a.shape()[i] > 0; }));
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), false));
        a.device().CheckDevicesCompatible(a, out);

        VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                struct MinAndArgMin {
                    T min;
                    int64_t argmin;
                };

                MinAndArgMin Identity() { return {T{}, -1}; }
                MinAndArgMin MapIn(T in, int64_t index) { return {in, index}; }
                void Reduce(MinAndArgMin next, MinAndArgMin& accum) {
                    if (accum.argmin < 0 || accum.min > next.min) {
                        accum = next;
                    }
                }
                int64_t MapOut(MinAndArgMin accum) { return accum.argmin; }
            };
            Reduce<T, int64_t>(a, axis, out, Impl{});
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ArgMinKernel, NativeArgMinKernel);

class NativeSumKernel : public SumKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        a.device().CheckDevicesCompatible(a, out);

        auto do_sum = [&a, &axis, &out](auto in_pt, auto out_pt) {
            using In = typename decltype(in_pt)::type;
            using Out = typename decltype(out_pt)::type;
            using Accum = std::conditional_t<std::is_same<Out, Float16>{}, float, Out>;
            struct Impl {
                Accum Identity() { return Accum{0}; }
                Accum MapIn(In in, int64_t /*index*/) { return static_cast<Accum>(in); }
                void Reduce(Accum next, Accum& accum) { accum += next; }
                Out MapOut(Accum accum) { return static_cast<Out>(accum); }
            };
            Reduce<In, Out>(a, axis, out, Impl{});
        };

        VisitDtype(out.dtype(), [a_dtype = a.dtype(), &do_sum](auto out_pt) { VisitDtype(a_dtype, do_sum, out_pt); });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SumKernel, NativeSumKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
