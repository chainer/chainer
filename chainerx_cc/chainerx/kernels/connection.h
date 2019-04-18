#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/kernel.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

// Computes the n-dimensional convolution.
//
// x: (batch_size, in_channels, in_1, in_2, ..., in_n)
// w: (out_channels, in_channels, k_1, k_2, ..., k_n)
// b: (out_channels)
//
// Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
class ConvKernel : public Kernel {
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
class ConvTransposeKernel : public Kernel {
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

class ConvGradWeightKernel : public Kernel {
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

}  // namespace chainerx
