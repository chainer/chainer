#pragma once

#include <memory>
#include <vector>

#include "xchainer/array_index.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {

class Array;

namespace routines {

Array FromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device = GetDefaultDevice());

Array Empty(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Full(const Shape& shape, Scalar scalar, Dtype dtype, Device& device = GetDefaultDevice());
Array Full(const Shape& shape, Scalar scalar, Device& device = GetDefaultDevice());
Array Zeros(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Ones(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());

// Creates an array which has the same shape and dtype as the other array.
// The new array is allocated in the default device. The device of the other array
// is ignored.
Array EmptyLike(const Array& array, Device& device = GetDefaultDevice());
Array FullLike(const Array& array, Scalar scalar, Device& device = GetDefaultDevice());
Array ZerosLike(const Array& array, Device& device = GetDefaultDevice());
Array OnesLike(const Array& array, Device& device = GetDefaultDevice());

// Creates a copy.
// It will be connected to all the graphs.
// It will be always C-contiguous.
Array Copy(const Array& array);

}  // namespace routines
}  // namespace xchainer
