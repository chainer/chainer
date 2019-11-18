#include "chainerx/native/im2col.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/backend.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dims.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/slice.h"

namespace chainerx {
namespace native {
namespace native_internal {

namespace {

template <typename T, int8_t kKernelNdim>
void Im2ColImpl(
        const Array& x,
        const Array& out,
        const Dims& kernel_size,
        const Dims& stride,
        const Dims& out_dims,
        const Indexer<2>& batch_channel_indexer) {
    static constexpr int8_t kInNdim = 2 + kKernelNdim;
    static constexpr int8_t kOutNdim = 2 + 2 * kKernelNdim;

    CHAINERX_ASSERT(kKernelNdim == static_cast<int8_t>(kernel_size.size()));
    CHAINERX_ASSERT(kKernelNdim == static_cast<int8_t>(stride.size()));
    CHAINERX_ASSERT(kKernelNdim == static_cast<int8_t>(out_dims.size()));
    CHAINERX_ASSERT(kInNdim == x.ndim());
    CHAINERX_ASSERT(kOutNdim == out.ndim());

    Indexer<kKernelNdim> kernel_indexer{Shape{kernel_size.begin(), kernel_size.end()}};
    Indexer<kKernelNdim> out_dims_indexer{Shape{out_dims.begin(), out_dims.end()}};
    Indexer<kInNdim> x_indexer{x.shape()};
    Indexer<kOutNdim> out_indexer{out.shape()};
    IndexableArray<const T, kInNdim> x_iarray{x};
    IndexableArray<T, kOutNdim> out_iarray{out};

    NdimIndex img_index{kKernelNdim};

    auto it_kernel = kernel_indexer.It(0);
    auto it_out_dims = out_dims_indexer.It(0);
    auto it_x = x_indexer.It(0);
    auto it_out = out_indexer.It(0);

    for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
        it_x.CopyIndex(it_batch_channel);
        it_out.CopyIndex(it_batch_channel);

        for (it_kernel.Restart(); it_kernel; ++it_kernel) {
            it_out.CopyIndex(it_kernel, 2);

            for (it_out_dims.Restart(); it_out_dims; ++it_out_dims) {
                for (int i = 0; i < kKernelNdim; ++i) {
                    img_index.index()[i] = it_out_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }
                it_x.CopyIndex(img_index, 2);
                it_out.CopyIndex(it_out_dims, 2 + kKernelNdim);

                out_iarray[it_out] = x_iarray[it_x];
            }
        }
    }
}

}  // namespace

Array Im2Col(const Array& x, const Dims& kernel_size, const Dims& stride, const Dims& pad, bool cover_all, Scalar pad_value) {
    auto ndim = static_cast<int8_t>(kernel_size.size());  // Number of input image dimensions.
    CHAINERX_ASSERT(ndim == static_cast<int8_t>(stride.size()));
    CHAINERX_ASSERT(ndim == static_cast<int8_t>(pad.size()));
    CHAINERX_ASSERT(ndim + 2 == x.ndim());  // Batch and channel dimensions.

    Device& device = x.device();

    // Create a padded copy of the input image.
    // TODO(hvy): Use the Pad function when implemented.
    Shape padded_shape = x.shape();
    std::vector<ArrayIndex> unpadded_slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int64_t i = 0; i < ndim; ++i) {
        padded_shape[i + 2] += pad[i] * 2 + (cover_all ? stride[i] - 1 : 0);  // Pad on both sides.
        unpadded_slice.emplace_back(Slice{pad[i], pad[i] + x.shape()[i + 2]});
    }
    Array padded_x = static_cast<int64_t>(pad_value) == int64_t{0} ? Zeros(padded_shape, x.dtype(), device)
                                                                   : Full(padded_shape, pad_value, x.dtype(), device);
    device.backend().CallKernel<CopyKernel>(x, padded_x.At(unpadded_slice));
    CHAINERX_ASSERT(ndim + 2 == padded_x.ndim());

    // Create the output array.
    Dims out_dims;  // Number of patches along each axis
    for (int8_t i = 0; i < ndim; ++i) {
        out_dims.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], kernel_size[i], stride[i], pad[i], cover_all));
        CHAINERX_ASSERT(out_dims.back() > 0);
    }
    CHAINERX_ASSERT(ndim == static_cast<int8_t>(out_dims.size()));

    int64_t batch_size = x.shape()[0];
    int64_t channels = x.shape()[1];

    Shape out_shape{batch_size, channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(out_shape));
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
    Array out = Empty(out_shape, x.dtype(), device);
    CHAINERX_ASSERT(ndim * 2 + 2 == out.ndim());

    // Write to the output array.
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};

        static_assert(4 * 2 + 2 == kMaxNdim, "4 is the maximum kernel ndim whose output ndim does not exceed kMaxNdim");
        switch (ndim) {
            case 0:
                Im2ColImpl<T, 0>(padded_x, out, kernel_size, stride, out_dims, batch_channel_indexer);
                break;
            case 1:
                Im2ColImpl<T, 1>(padded_x, out, kernel_size, stride, out_dims, batch_channel_indexer);
                break;
            case 2:
                Im2ColImpl<T, 2>(padded_x, out, kernel_size, stride, out_dims, batch_channel_indexer);
                break;
            case 3:
                Im2ColImpl<T, 3>(padded_x, out, kernel_size, stride, out_dims, batch_channel_indexer);
                break;
            case 4:
                Im2ColImpl<T, 4>(padded_x, out, kernel_size, stride, out_dims, batch_channel_indexer);
                break;
            default:
                CHAINERX_NEVER_REACH();  // Never out.ndim() > kMaxNdim
                break;
        }
    });

    return out;
}

}  // namespace native_internal
}  // namespace native
}  // namespace chainerx
