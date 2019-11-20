#pragma once

#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array_fwd.h"
#include "chainerx/backward_fwd.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/shape.h"

namespace chainerx {

class BackwardContext;

namespace internal {

class ArrayBody;
class ArrayNode;

// Throws GradientError in case of mismatch in gradient array props.
void AccumulateGrad(absl::optional<Array>& target_grad, Array partial_grad, const Shape& shape, Dtype dtype, Device& device);

// Throws GradientError in case of mismatch in gradient array props.
void SetGrad(absl::optional<Array>& target_grad, Array grad, const Shape& shape, Dtype dtype, Device& device);

}  // namespace internal

// Updates the gradients held by the input arrays using backpropagation.
//
// This functions is not thread safe.
void Backward(
        const Array& output,
        const absl::optional<BackpropId>& backprop_id = absl::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable,
        absl::optional<float> loss_scale = absl::nullopt);

// Updates the gradients held by the input arrays using backpropagation.
//
// This functions is not thread safe.
void Backward(
        const std::vector<ConstArrayRef>& outputs,
        const absl::optional<BackpropId>& backprop_id = absl::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable,
        absl::optional<float> loss_scale = absl::nullopt);

// Returns gradient arrays for all inputs.
std::vector<absl::optional<Array>> Grad(
        const std::vector<ConstArrayRef>& outputs,
        const std::vector<ConstArrayRef>& inputs,
        const absl::optional<BackpropId>& backprop_id = absl::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable,
        bool set_grad = false,
        bool retain_grad = false,
        const std::vector<ConstArrayRef>& grad_outputs = {},
        absl::optional<float> loss_scale = absl::nullopt);

}  // namespace chainerx
