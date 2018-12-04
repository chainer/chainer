#pragma once

#include <vector>

#include <nonstd/optional.hpp>

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
void AccumulateGrad(nonstd::optional<Array>& target_grad, Array partial_grad, const Shape& shape, Dtype dtype, Device& device);

// Throws GradientError in case of mismatch in gradient array props.
void SetGrad(nonstd::optional<Array>& target_grad, Array grad, const Shape& shape, Dtype dtype, Device& device);

}  // namespace internal

// Computes the gradients by back propagation.
//
// This functions is not thread safe.
void Backward(
        const Array& output,
        const nonstd::optional<BackpropId>& backprop_id = nonstd::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

// Computes the gradients by back propagation.
//
// This functions is not thread safe.
void Backward(
        const std::vector<ConstArrayRef>& outputs,
        const nonstd::optional<BackpropId>& backprop_id = nonstd::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace chainerx
