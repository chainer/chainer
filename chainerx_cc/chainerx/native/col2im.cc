#include "chainerx/native/col2im.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dims.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/routines/creation.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/slice.h"

namespace chainerx {
namespace native {
namespace native_internal {

namespace {

template <typename T, int8_t kKernelNdim>
void Col2ImImpl(const Array& col, const Array& out, const Dims& stride, const Indexer<2>& batch_channel_indexer) {
    static constexpr int8_t kColNdim = 2 + 2 * kKernelNdim;
    static constexpr int8_t kOutNdim = 2 + kKernelNdim;

    CHAINERX_ASSERT(kKernelNdim == static_cast<int8_t>(stride.size()));
    CHAINERX_ASSERT(kColNdim == col.ndim());
    CHAINERX_ASSERT(kOutNdim == out.ndim());

    Indexer<kKernelNdim> kernel_indexer{Shape{col.shape().begin() + 2, col.shape().begin() + 2 + kKernelNdim}};
    Indexer<kKernelNdim> in_image_dims_indexer{Shape{col.shape().begin() + 2 + kKernelNdim, col.shape().end()}};
    Indexer<kColNdim> col_indexer{col.shape()};
    Indexer<kOutNdim> out_indexer{out.shape()};
    IndexableArray<const T, kColNdim> col_iarray{col};
    IndexableArray<T, kOutNdim> out_iarray{out};

    // Indices over the output image.
    NdimIndex out_image_index{kKernelNdim};

    auto it_kernel = kernel_indexer.It(0);
    auto it_in_image_dims = in_image_dims_indexer.It(0);
    auto it_col = col_indexer.It(0);
    auto it_out = out_indexer.It(0);

    for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
        it_col.CopyIndex(it_batch_channel);
        it_out.CopyIndex(it_batch_channel);

        for (it_kernel.Restart(); it_kernel; ++it_kernel) {
            it_col.CopyIndex(it_kernel, 2);

            for (it_in_image_dims.Restart(); it_in_image_dims; ++it_in_image_dims) {
                for (int8_t i = 0; i < kKernelNdim; ++i) {
                    out_image_index.index()[i] = it_in_image_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }
                it_col.CopyIndex(it_in_image_dims, 2 + kKernelNdim);
                it_out.CopyIndex(out_image_index, 2);

                native_internal::StorageToDataType<T>(out_iarray[it_out]) +=
                        native_internal::StorageToDataType<const T>(col_iarray[it_col]);
            }
        }
    }
}

}  // namespace

Array Col2Im(const Array& col, const Dims& stride, const Dims& pad, const Dims& out_size) {
    int64_t batch_size = col.shape()[0];
    int64_t channels = col.shape()[1];
    auto ndim = static_cast<int8_t>(stride.size());
    CHAINERX_ASSERT(ndim * 2 + 2 == col.ndim());

    Shape padded_shape{batch_size, channels};
    for (int8_t i = 0; i < ndim; ++i) {
        padded_shape.emplace_back(out_size[i] + 2 * pad[i] + stride[i] - 1);
    }
    Array padded_out = Zeros(padded_shape, col.dtype(), col.device());
    CHAINERX_ASSERT(ndim + 2 == padded_out.ndim());

    // Write to the output array
    VisitDtype(col.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};

        static_assert(4 * 2 + 2 == kMaxNdim, "4 is the maximum kernel ndim whose col ndim does not exceed kMaxNdim");
        switch (ndim) {
            case 0:
                Col2ImImpl<T, 0>(col, padded_out, stride, batch_channel_indexer);
                break;
            case 1:
                Col2ImImpl<T, 1>(col, padded_out, stride, batch_channel_indexer);
                break;
            case 2:
                Col2ImImpl<T, 2>(col, padded_out, stride, batch_channel_indexer);
                break;
            case 3:
                Col2ImImpl<T, 3>(col, padded_out, stride, batch_channel_indexer);
                break;
            case 4:
                Col2ImImpl<T, 4>(col, padded_out, stride, batch_channel_indexer);
                break;
            default:
                CHAINERX_NEVER_REACH();  // Never col.ndim() > kMaxNdim
                break;
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
}  // namespace chainerx
