#include "xchainer/native/native_device.h"

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/macro.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace native {

void NativeDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CheckDevicesCompatible(a, indices, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> a_iarray{a};
        IndexableArray<T> out_iarray{out};
        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> a_indexer{a.shape()};
        Indexer<> out_indexer{out.shape()};
        Indexer<> indices_indexer{indices.shape()};

        int64_t axis_dim = a.shape()[axis];

        // left: set of input dimensions lower than the axis
        // right: set of input dimensions higher than the axis
        Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
        Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
        Shape axis_shape{axis_dim};  // always ndim==1
        Indexer<> left_indexer{left_shape};
        Indexer<> right_indexer{right_shape};
        Indexer<> axis_indexer{axis_shape};

        for (auto it = indices_indexer.It(0); it; ++it) {
            int64_t index = indices_iarray[it];
            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            XCHAINER_ASSERT(0 <= index);
            XCHAINER_ASSERT(index < axis_dim);
            auto it_axis = axis_indexer.It(index);

            for (auto it_left = left_indexer.It(0); it_left; ++it_left) {
                for (auto it_right = right_indexer.It(0); it_right; ++it_right) {
                    auto it_out = out_indexer.At(it_left, it, it_right);
                    auto it_a = a_indexer.At(it_left, it_axis, it_right);
                    out_iarray[it_out] = a_iarray[it_a];
                }
            }
        }
    });
}

void NativeDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, indices, b);
    XCHAINER_ASSERT(a.shape() == out.shape());
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        IndexableArray<const int64_t> indices_iarray{indices};
        IndexableArray<T> out_iarray{out};
        Indexer<> b_indexer{b.shape()};
        Indexer<> indices_indexer{indices.shape()};
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

        // Add
        for (auto it = indices_indexer.It(0); it; ++it) {
            int64_t index = indices_iarray[it];
            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            XCHAINER_ASSERT(0 <= index);
            XCHAINER_ASSERT(index < axis_dim);
            auto it_axis = axis_indexer.It(index);

            for (auto it_left = left_indexer.It(0); it_left; ++it_left) {
                for (auto it_right = right_indexer.It(0); it_right; ++it_right) {
                    auto it_out = out_indexer.At(it_left, it_axis, it_right);
                    auto it_b = b_indexer.At(it_left, it, it_right);
                    out_iarray[it_out] += b_iarray[it_b];
                }
            }
        }
    });
}

}  // namespace native
}  // namespace xchainer
