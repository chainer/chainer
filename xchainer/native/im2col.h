#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/index_iterator.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/scalar.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace native_internal {

template <typename T, int8_t kKernelNdim>
void Im2ColImpl(
        const Array& x,
        const Array& out,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& out_dims,
        const Indexer<2>& batch_channel_indexer) {
    static constexpr int8_t kInNdim = 2 + kKernelNdim;
    static constexpr int8_t kOutNdim = 2 + 2 * kKernelNdim;

    assert(kKernelNdim == static_cast<int8_t>(kernel_size.size()));
    assert(kKernelNdim == static_cast<int8_t>(stride.size()));
    assert(kKernelNdim == static_cast<int8_t>(out_dims.size()));
    assert(kInNdim == x.ndim());
    assert(kOutNdim == out.ndim());

    Indexer<kKernelNdim> kernel_indexer{Shape{kernel_size.begin(), kernel_size.end()}};
    Indexer<kKernelNdim> out_dims_indexer{Shape{out_dims.begin(), out_dims.end()}};
    Indexer<kInNdim> x_indexer{x.shape()};
    Indexer<kOutNdim> out_indexer{out.shape()};
    IndexableArray<const T, kInNdim> x_iarray{x};
    IndexableArray<T, kOutNdim> out_iarray{out};

    NdimIndex img_index{kKernelNdim};

    for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_out_dims = out_dims_indexer.It(0); it_out_dims; ++it_out_dims) {
                for (int i = 0; i < kKernelNdim; ++i) {
                    img_index.index()[i] = it_out_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                auto it_x = x_indexer.At(it_batch_channel, img_index);
                auto it_out = out_indexer.At(it_batch_channel, it_kernel, it_out_dims);

                // Write the output column value.
                out_iarray[it_out] = x_iarray[it_x];
            }
        }
    }
}

Array Im2Col(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        Scalar pad_value = 0);

}  // namespace native_internal
}  // namespace native
}  // namespace xchainer
