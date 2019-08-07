#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/reduction.h"
#include "chainerx/kernels/sorting.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/native/reduce.h"
#include "chainerx/numeric.h"
#include "chainerx/numeric_limits.h"
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

// TODO(imanishi): Performance improvement.
class NativeCumsumKernel : public CumsumKernel {
public:
    void Call(const Array& a, int8_t axis, const Array& out) override {
        a.device().CheckDevicesCompatible(a, out);
        const Array& a_cast = a.dtype() != out.dtype() ? a.AsType(out.dtype()) : a;

        VisitDtype(out.dtype(), [&a_cast, axis, &out](auto pt) {
            using T = typename decltype(pt)::type;

            IndexableArray<const T> a_iarray{a_cast};
            IndexableArray<T> out_iarray{out};
            Indexer<> out_indexer{out.shape()};
            Indexer<> prev_indexer{a_cast.shape()};

            int64_t axis_dim = a_cast.shape()[axis];

            // left: set of input dimensions lower than the axis
            // right: set of input dimensions higher than the axis
            Shape left_shape{a_cast.shape().begin(), a_cast.shape().begin() + axis};
            Shape right_shape{a_cast.shape().begin() + (axis + 1), a_cast.shape().end()};
            Shape axis_shape{axis_dim};  // always ndim==1

            Indexer<> left_indexer{left_shape};
            Indexer<> right_indexer{right_shape};
            Indexer<> indices_indexer{axis_shape};

            auto it_left = left_indexer.It(0);
            auto it_right = right_indexer.It(0);
            auto it_out = out_indexer.It(0);
            auto it_prev = prev_indexer.It(0);

            // Copy
            for (auto it = out_indexer.It(0); it; ++it) {
                out_iarray[it] = a_iarray[it];
            }

            for (auto it = indices_indexer.It(1); it; ++it) {
                int64_t index = it.raw_index();
                index = index % axis_dim;
                CHAINERX_ASSERT(0 <= index);
                CHAINERX_ASSERT(index < axis_dim);

                it_out.CopyIndex(it, it_left.ndim());

                for (it_left.Restart(); it_left; ++it_left) {
                    it_out.CopyIndex(it_left);

                    for (it_right.Restart(); it_right; ++it_right) {
                        it_out.CopyIndex(it_right, it_left.ndim() + it.ndim());
                        it_prev.CopyIndex(it_out);
                        it_prev.index()[axis] -= 1;
                        native_internal::StorageToDataType<T>(out_iarray[it_out]) +=
                                native_internal::StorageToDataType<T>(out_iarray[it_prev]);
                        it_prev.index()[axis] += 1;
                    }
                }
            }
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(CumsumKernel, NativeCumsumKernel);

class NativeNansumKernel : public NansumKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        a.device().CheckDevicesCompatible(a, out);

        auto do_nansum = [&a, &axis, &out](auto in_pt, auto out_pt) {
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

        VisitDtype(out.dtype(), [a_dtype = a.dtype(), &do_nansum](auto out_pt) { VisitDtype(a_dtype, do_nansum, out_pt); });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(NansumKernel, NativeNansumKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
