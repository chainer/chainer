#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {

void NativeDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CHAINERX_ASSERT(GetKind(indices.dtype()) == DtypeKind::kInt || GetKind(indices.dtype()) == DtypeKind::kUInt);
    CheckDevicesCompatible(a, indices, out);

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

void NativeDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    CHAINERX_ASSERT(a.shape() == out.shape());
    CHAINERX_ASSERT(GetKind(indices.dtype()) == DtypeKind::kInt || GetKind(indices.dtype()) == DtypeKind::kUInt);
    CheckDevicesCompatible(a, indices, b);

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

}  // namespace native
}  // namespace chainerx
