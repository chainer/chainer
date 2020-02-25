#pragma once

#include <functional>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"

namespace chainerx {

// Tests differentiation of a given procedure.
//
// This function automatically checks if the backward procedure of `func` is
// correctly implemented, starting from the initial gradient given by `grad_outputs`.
//
// It throws GradientCheckError when the test fails.
void CheckBackward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& eps,
        size_t concurrent_check_thread_count = 2U,
        double atol = 1e-5,
        double rtol = 1e-4,
        const absl::optional<BackpropId>& backprop_id = absl::nullopt);

// Tests twice differentiation of a given procedure.
//
// This function automatically checks if the backward procedure of `func` is
// correctly implemented for further differentiation. It first computes the
// gradient of `func` w.r.t. its inputs in the same way as `CheckBackwardComputation`.
// This function then further invokes the backward procedure against the
// gradient variables, starting from the initial gradient given by `grad_grad_inputs`.
// It also computes the second gradient using `CalculateNumericalGradient`.
// The resulting gradients are compared to confirm if the second-order gradients
// are approximately correct.
//
// It throws GradientCheckError when the test fails.
//
// Note that this function **DOES NOT** check if the first-order differentiation
// is correct; the numerical gradient assumes that the first-order gradient given
// by the usual `Backward` is correct. The implementation of each differentiable
// function should be tested by `CheckBackwardComputation` first, and then should be
// tested by this function if necessary.
void CheckDoubleBackwardComputation(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& grad_grad_inputs,
        const std::vector<Array>& eps,
        size_t concurrent_check_thread_count = 2U,
        double atol = 1e-5,
        double rtol = 1e-4,
        const absl::optional<BackpropId>& backprop_id = absl::nullopt);

}  // namespace chainerx
