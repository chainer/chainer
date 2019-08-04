#pragma once

#include <tuple>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, absl::optional<Dtype> out_dtype = absl::nullopt);

Array Solve(const Array& a, const Array& b);

Array Inverse(const Array& a);

enum class QrMode {
    // if K = min(M, N), where `a` of shape (M, N)
    kReduced,  // returns q, r with dimensions (M, K), (K, N) (default)
    kComplete,  // returns q, r with dimensions (M, M), (M, N)
    kR,  // returns empty q and r with dimensions (0, 0), (K, N)
    kRaw  // returns h, tau with dimensions (N, M), (K, 1)
};

std::tuple<Array, Array> Qr(const Array& a, QrMode mode);

}  // namespace chainerx
