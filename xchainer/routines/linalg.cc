#include "xchainer/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace {}  // namespace

Array Dot(const Array& a, const Array& b) {
    if (a.ndim() == 0 || b.ndim() == 0) {
        return a * b;
    }

    // TODO(beam2d): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    Array x = a.AsConstant();
    Array y = b.AsConstant();

    // TODO(beam2d): Support it. Need to transpose y so that the inner-product axis is moved to the top.
    if (y.ndim() > 2) {
        throw NotImplementedError("dot does not support rhs operand with ndim > 2");
    }

    std::vector<int64_t> out_shape_v;
    out_shape_v.reserve(x.ndim() + y.ndim() - 2);
    std::copy(x.shape().begin(), x.shape().end() - 1, std::back_inserter(out_shape_v));
    std::copy(y.shape().begin() + 1, y.shape().end(), std::back_inserter(out_shape_v));
    Shape out_shape{out_shape_v.begin(), out_shape_v.end()};

    int64_t k = x.shape()[x.ndim() - 1];
    if (y.shape()[0] != k) {
        throw DimensionError("Axis dimension mismatch");
    }
    if (k == 0) {
        return Array::Zeros(out_shape, a.dtype(), a.device());
    }

    // Make each operand a matrix
    int64_t m = x.GetTotalSize() / k;
    int64_t n = y.GetTotalSize() / k;
    Array x_matrix = x.Reshape({m, k});
    Array y_matrix = y.Reshape({k, n});

    Array out = Array::Empty(out_shape, a.dtype(), a.device());
    Array out_matrix = out.Reshape({m, n});
    x.device().Dot(x_matrix, y_matrix, out_matrix);

    // TODO(beam2d): Implement backward.
    return out;
}

}  // namespace xchainer
