#pragma once

#include <cstdint>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dims.h"

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
Array Conv(
        const Array& x,
        const Array& w,
        const absl::optional<Array>& b,
        const Dims& stride,
        const Dims& pad,
        bool cover_all = false,
        absl::optional<Dtype> out_dtype = absl::nullopt);

Array ConvTranspose(
        const Array& x,
        const Array& w,
        const absl::optional<Array>& b,
        const Dims& stride,
        const Dims& pad,
        const absl::optional<Dims>& out_size = absl::nullopt,
        absl::optional<Dtype> out_dtype = absl::nullopt);

Array Linear(const Array& x, const Array& w, const absl::optional<Array>& b = absl::nullopt, uint8_t n_batch_axes = 1);

std::vector<Array> Lstm(const Array& c, const Array& x);

}  // namespace chainerx
