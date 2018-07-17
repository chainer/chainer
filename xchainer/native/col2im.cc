#include "xchainer/native/col2im.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace native_internal {

Array Col2Im(
        const Array& col,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    int8_t batch_size = col.shape()[0];
    int8_t channels = col.shape()[1];
    auto ndim = static_cast<int8_t>(stride.size());

    Shape padded_shape{batch_size, channels};
    for (int8_t i = 0; i < ndim; ++i) {
        padded_shape.emplace_back(out_size[i] + 2 * pad[i] + stride[i] - 1);
    }
    Array padded_out = Zeros(padded_shape, col.dtype(), col.device());

    // Write to the output array
    VisitDtype(col.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};
        Indexer<> kernel_indexer{Shape{col.shape().begin() + 2, col.shape().begin() + 2 + ndim}};
        Indexer<> in_image_dims_indexer{Shape{col.shape().begin() + 2 + ndim, col.shape().end()}};
        Indexer<> col_indexer{col.shape()};
        Indexer<> padded_out_indexer{padded_shape};
        IndexableArray<const T> col_iarray{col};
        IndexableArray<T> padded_out_iarray{padded_out};

        // Indices over the output image.
        NdimIndex out_image_index{ndim};

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_in_image_dims = in_image_dims_indexer.It(0); it_in_image_dims; ++it_in_image_dims) {
                for (int8_t i = 0; i < ndim; ++i) {
                    out_image_index.index()[i] = it_in_image_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
                    auto it_col = col_indexer.At(it_batch_channel, it_kernel, it_in_image_dims);
                    auto it_padded_out = padded_out_indexer.At(it_batch_channel, out_image_index);
                    padded_out_iarray[it_padded_out] += col_iarray[it_col];
                }
            }
        }
    });

    std::vector<ArrayIndex> slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int8_t i = 0; i < ndim; ++i) {
        slice.emplace_back(Slice{pad[i], pad[i] + out_size[i]});
    }
    return padded_out.At(slice);
}

}  // namespace native_internal
}  // namespace native
}  // namespace xchainer
