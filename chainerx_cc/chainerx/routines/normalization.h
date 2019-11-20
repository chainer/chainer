#pragma once

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/scalar.h"

namespace chainerx {

// Computes the batch normalization along the given axis.
// If axis is omitted, the first axis is treated as the batch axis and will be reduced during normalization.
// Running mean and running variance that are passed as arguments will be updated in-place.
Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps = 2e-5,
        Scalar decay = 0.9,
        const OptionalAxes& axis = absl::nullopt);

// Computes the fixed batch normalization.
// axis argument is treated in the same way as BatchNorm.
// Backward computation is not implemented.
Array FixedBatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& mean,
        const Array& var,
        Scalar eps,
        const OptionalAxes& axis = absl::nullopt);

}  // namespace chainerx
