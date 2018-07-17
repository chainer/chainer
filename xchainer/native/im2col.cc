#include "xchainer/native/im2col.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace native_internal {

Array Im2Col(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        Scalar pad_value) {
    auto ndim = static_cast<int8_t>(kernel_size.size());  // Number of input image dimensions.
    assert(ndim == static_cast<int8_t>(stride.size()));
    assert(ndim == static_cast<int8_t>(pad.size()));
    assert(ndim + 2 == x.ndim());  // Batch and channel dimensions.

    Device& device = x.device();

    // Create a padded copy of the input image.
    // TODO(hvy): Use the Pad function when implemented.
    Shape padded_shape = x.shape();
    std::vector<ArrayIndex> unpadded_slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int64_t i = 0; i < ndim; ++i) {
        padded_shape[i + 2] += pad[i] * 2 + (cover_all ? stride[i] - 1 : 0);  // Pad on both sides.
        unpadded_slice.emplace_back(Slice{pad[i], pad[i] + x.shape()[i]});
    }
    Array padded_x = static_cast<int64_t>(pad_value) == int64_t{0} ? Zeros(padded_shape, x.dtype(), device)
                                                                   : Full(padded_shape, pad_value, x.dtype(), device);
    device.Copy(x, padded_x.At(unpadded_slice));

    // Create the output array.
    StackVector<int64_t, kMaxNdim> out_dims;  // Number of patches along each axis
    for (int8_t i = 0; i < ndim; ++i) {
        out_dims.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], kernel_size[i], stride[i], pad[i], cover_all));
        assert(out_dims.back() > 0);
    }

    int64_t batch_size = x.shape()[0];
    int64_t channels = x.shape()[1];

    Shape out_shape{batch_size, channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(out_shape));
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
    Array out = Empty(out_shape, x.dtype(), device);

    // Write to the output array.
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};
        Indexer<> kernel_indexer{Shape{kernel_size.begin(), kernel_size.end()}};
        Indexer<> out_dims_indexer{Shape{out_dims.begin(), out_dims.end()}};
        Indexer<> x_indexer{padded_x.shape()};
        Indexer<> out_indexer{out.shape()};
        IndexableArray<const T> x_iarray{padded_x};
        IndexableArray<T> out_iarray{out};

        // Indices over input image.
        NdimIndex img_index{ndim};

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_out_dims = out_dims_indexer.It(0); it_out_dims; ++it_out_dims) {
                for (int i = 0; i < ndim; ++i) {
                    img_index.index()[i] = it_out_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
                    auto it_x = x_indexer.At(it_batch_channel, img_index);
                    auto it_out = out_indexer.At(it_batch_channel, it_kernel, it_out_dims);

                    // Write the output column value.
                    out_iarray[it_out] = x_iarray[it_x];
                }
            }
        }
    });

    return out;
}

}  // namespace native_internal
}  // namespace native
}  // namespace xchainer
