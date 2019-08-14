#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/op.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace internal {

// Calculates output size of convolution.
//
// DimensionError is thrown if the output size is 0 or negative.
int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all);

int64_t GetConvTransposeOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all);

}  // namespace internal

// Computes the n-dimensional convolution.
//
// x: (batch_size, in_channels, in_1, in_2, ..., in_n)
// w: (out_channels, in_channels, k_1, k_2, ..., k_n)
// b: (out_channels)
//
// Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
class ConvOp : public Op {
public:
    static const char* name() { return "Conv"; }

    virtual Array Call(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            Dtype out_dtype,
            const nonstd::optional<Array>& out) = 0;
};

// Computes the n-dimensional transposed convolution.
//
// x: (batch_size, in_channels, in_1, in_2, ..., in_n)
// w: (in_channels, out_channels, k_1, k_2, ..., k_n)
// b: (out_channels)
//
// Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
class ConvTransposeOp : public Op {
public:
    static const char* name() { return "ConvTranspose"; }
    virtual Array Call(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& out_size,
            Dtype out_dtype,
            const nonstd::optional<Array>& out) = 0;
};

class ConvGradWeightOp : public Op {
public:
    static const char* name() { return "ConvGradWeight"; }
    virtual Array Call(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            const nonstd::optional<Array>& out) = 0;
};

// Computes the n-dimensional convolution.
//
// x: (batch_size, in_channels, in_1, in_2, ..., in_n)
// w: (out_channels, in_channels, k_1, k_2, ..., k_n)
// b: (out_channels)
//
// Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
Array Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all = false,
        nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

Array ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size = nonstd::nullopt,
        nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

Array Linear(const Array& x, const Array& w, const nonstd::optional<Array>& b = nonstd::nullopt, uint8_t n_batch_axes = 1);

}  // namespace chainerx
