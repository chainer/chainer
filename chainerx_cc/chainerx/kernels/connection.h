#pragma once

#include <cstdint>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dims.h"
#include "chainerx/kernel.h"

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
    virtual Array Call(
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            Dtype out_dtype,
            const absl::optional<Array>& out) = 0;
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
    virtual Array Call(
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            const Dims& out_size,
            Dtype out_dtype,
            const absl::optional<Array>& out) = 0;
};

class ConvGradWeightKernel : public Kernel {
public:
    virtual Array Call(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            const absl::optional<Array>& out) = 0;
};

}  // namespace chainerx
