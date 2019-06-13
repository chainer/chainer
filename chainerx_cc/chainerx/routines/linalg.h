#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

enum class QRMode {
    // if K = min(M, N), where `a` of shape (M, N)
    reduced,  // returns q, r with dimensions (M, K), (K, N) (default)
    complete,  // returns q, r with dimensions (M, M), (M, N)
    r,  // returns empty q and r with dimensions (0, 0), (K, N)
    raw  // returns h, tau with dimensions (N, M), (K, 1)
};

std::tuple<Array, Array> QR(const Array& a, QRMode mode);

}  // namespace chainerx
