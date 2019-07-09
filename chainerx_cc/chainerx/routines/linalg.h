#pragma once

#include <absl/types/optional.h>
#include <string>
#include <tuple>

#include "chainerx/array.h"
#include "chainerx/dtype.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, absl::optional<Dtype> out_dtype = absl::nullopt);

std::tuple<Array, Array> Eigh(const Array& a, const std::string& uplo);

Array Eigvalsh(const Array& a, const std::string& uplo);

}  // namespace chainerx
