#pragma once

#include <nonstd/optional.hpp>
#include <string>
#include <tuple>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

std::tuple<Array, Array> Eigh(const Array& a, const std::string& uplo);

Array Eigvalsh(const Array& a, const std::string& uplo);

}  // namespace chainerx
