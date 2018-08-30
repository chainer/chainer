#pragma once

#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array_fwd.h"
#include "xchainer/backward_fwd.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"

namespace xchainer {

class BackwardContext;

namespace internal {

class ArrayBody;
class ArrayNode;

void AccumulateGrad(nonstd::optional<Array>& target_grad, Array partial_grad, const Shape& shape, Dtype dtype, Device& device);

void SetGrad(nonstd::optional<Array>& target_grad, Array grad, const Shape& shape, Dtype dtype, Device& device);

}  // namespace internal

void Backward(
        const Array& output,
        const nonstd::optional<BackpropId>& backprop_id = nonstd::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

void Backward(
        const std::vector<ConstArrayRef>& outputs,
        const nonstd::optional<BackpropId>& backprop_id = nonstd::nullopt,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace xchainer
