#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/kernels/indexing.h"
#include "chainerx/macro.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {
namespace {

class NativeTakeKernel : public TakeKernel {
public:
    void Call(const Array& a, const Array& indices, int8_t axis, const Array& out) override {
        CHAINERX_ASSERT(GetKind(indices.dtype()) == DtypeKind::kInt || GetKind(indices.dtype()) == DtypeKind::kUInt);
        a.device().CheckDevicesCompatible(a, indices, out);

        const Array& indices_cast = indices.dtype() == Dtype::kInt64 ? indices : indices.AsType(Dtype::kInt64);

        VisitDtype(out.dtype(), [&a, &indices_cast, axis, &out](auto pt) {
            using T = typename decltype(pt)::type;

            IndexableArray<const T> a_iarray{a};
            IndexableArray<T> out_iarray{out};
            IndexableArray<const int64_t> indices_iarray{indices_cast};
            Indexer<> a_indexer{a.shape()};
            Indexer<> out_indexer{out.shape()};
            Indexer<> indices_indexer{indices_cast.shape()};

            int64_t axis_dim = a.shape()[axis];

            // left: set of input dimensions lower than the axis
            // right: set of input dimensions higher than the axis
            Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
            Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
            Shape axis_shape{axis_dim};  // always ndim==1
            Indexer<> left_indexer{left_shape};
            Indexer<> right_indexer{right_shape};
            Indexer<> axis_indexer{axis_shape};

            auto it_left = left_indexer.It(0);
            auto it_right = right_indexer.It(0);
            auto it_axis = axis_indexer.It(0);
            auto it_out = out_indexer.It(0);
            auto it_a = a_indexer.It(0);

            for (auto it = indices_indexer.It(0); it; ++it) {
                int64_t index = indices_iarray[it];
                if (index < 0) {
                    index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
                } else {
                    index = index % axis_dim;
                }
                CHAINERX_ASSERT(0 <= index);
                CHAINERX_ASSERT(index < axis_dim);
                it_axis.Restart(index);

                it_out.CopyIndex(it, it_left.ndim());
                it_a.CopyIndex(it_axis, it_left.ndim());

                for (it_left.Restart(); it_left; ++it_left) {
                    it_out.CopyIndex(it_left);
                    it_a.CopyIndex(it_left);

                    for (it_right.Restart(); it_right; ++it_right) {
                        it_out.CopyIndex(it_right, it_left.ndim() + it.ndim());
                        it_a.CopyIndex(it_right, it_left.ndim() + it_axis.ndim());
                        out_iarray[it_out] = a_iarray[it_a];
                    }
                }
            }
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(TakeKernel, NativeTakeKernel);

class NativeAddAtKernel : public AddAtKernel {
public:
    void Call(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) override {
        CHAINERX_ASSERT(a.shape() == out.shape());
        CHAINERX_ASSERT(GetKind(indices.dtype()) == DtypeKind::kInt || GetKind(indices.dtype()) == DtypeKind::kUInt);
        a.device().CheckDevicesCompatible(a, indices, b);

        const Array& indices_cast = indices.dtype() == Dtype::kInt64 ? indices : indices.AsType(Dtype::kInt64);

        VisitDtype(a.dtype(), [&a, &indices_cast, axis, &b, &out](auto pt) {
            using T = typename decltype(pt)::type;

            IndexableArray<const T> a_iarray{a};
            IndexableArray<const T> b_iarray{b};
            IndexableArray<const int64_t> indices_iarray{indices_cast};
            IndexableArray<T> out_iarray{out};
            Indexer<> b_indexer{b.shape()};
            Indexer<> indices_indexer{indices_cast.shape()};
            Indexer<> out_indexer{out.shape()};  // indexer for both out_iarray and a_array

            int64_t axis_dim = a.shape()[axis];

            // left: set of input dimensions lower than the axis
            // right: set of input dimensions higher than the axis
            Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
            Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
            Shape axis_shape{axis_dim};  // always ndim==1
            Indexer<> left_indexer{left_shape};
            Indexer<> right_indexer{right_shape};
            Indexer<> axis_indexer{axis_shape};

            // Copy
            for (auto it = out_indexer.It(0); it; ++it) {
                out_iarray[it] = a_iarray[it];
            }

            auto it_left = left_indexer.It(0);
            auto it_right = right_indexer.It(0);
            auto it_axis = axis_indexer.It(0);
            auto it_out = out_indexer.It(0);
            auto it_b = b_indexer.It(0);

            // Add
            for (auto it = indices_indexer.It(0); it; ++it) {
                int64_t index = indices_iarray[it];
                if (index < 0) {
                    index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
                } else {
                    index = index % axis_dim;
                }
                CHAINERX_ASSERT(0 <= index);
                CHAINERX_ASSERT(index < axis_dim);
                it_axis.Restart(index);

                it_out.CopyIndex(it_axis, it_left.ndim());
                it_b.CopyIndex(it, it_left.ndim());

                for (it_left.Restart(); it_left; ++it_left) {
                    it_out.CopyIndex(it_left);
                    it_b.CopyIndex(it_left);

                    for (it_right.Restart(); it_right; ++it_right) {
                        it_out.CopyIndex(it_right, it_left.ndim() + it_axis.ndim());
                        it_b.CopyIndex(it_right, it_left.ndim() + it.ndim());
                        out_iarray[it_out] += b_iarray[it_b];
                    }
                }
            }
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AddAtKernel, NativeAddAtKernel);

class NativeWhereKernel : public WhereKernel {
public:
    void Call(const Array& condition, const Array& x, const Array& y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, x, y, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        Dtype out_dtype = out.dtype();
        const Array& x_cast = x.dtype() != out_dtype ? x.AsType(out_dtype) : x;
        const Array& y_cast = y.dtype() != out_dtype ? y.AsType(out_dtype) : y;

        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, bool condition, T x, T y, T& out) { out = condition ? x : y; }
            };
            Elementwise<const bool, const T, const T, T>(Impl{}, condition_cast, x_cast, y_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(WhereKernel, NativeWhereKernel);

class NativeWhereAASKernel : public WhereAASKernel {
public:
    void Call(const Array& condition, const Array& x, Scalar y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, x, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        Dtype out_dtype = out.dtype();
        const Array& x_cast = x.dtype() != out_dtype ? x.AsType(out_dtype) : x;

        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                T y;
                void operator()(int64_t /*i*/, bool condition, T x, T& out) { out = condition ? x : y; }
            };
            Elementwise<const bool, const T, T>(Impl{static_cast<T>(y)}, condition_cast, x_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(WhereAASKernel, NativeWhereAASKernel);

class NativeWhereASAKernel : public WhereASAKernel {
public:
    void Call(const Array& condition, Scalar x, const Array& y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, y, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        Dtype out_dtype = out.dtype();
        const Array& y_cast = y.dtype() != out_dtype ? y.AsType(out_dtype) : y;

        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                T x;
                void operator()(int64_t /*i*/, bool condition, T y, T& out) { out = condition ? x : y; }
            };
            Elementwise<const bool, const T, T>(Impl{static_cast<T>(x)}, condition_cast, y_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(WhereASAKernel, NativeWhereASAKernel);

class NativeWhereASSKernel : public WhereASSKernel {
public:
    void Call(const Array& condition, Scalar x, Scalar y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                T x;
                T y;
                void operator()(int64_t /*i*/, bool condition, T& out) { out = condition ? x : y; }
            };
            Elementwise<const bool, T>(Impl{static_cast<T>(x), static_cast<T>(y)}, condition_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(WhereASSKernel, NativeWhereASSKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
