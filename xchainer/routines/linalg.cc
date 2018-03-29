#include "xchainer/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/shape.h"

namespace xchainer {

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

    auto a_backward_fn = [other = b](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return Dot(gout, other.AsConstant(graph_ids_to_stop_gradient).Transpose());
    };
    auto b_backward_fn = [ other = a, m, n, k ](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        Array a_matrix = other.AsConstant(graph_ids_to_stop_gradient).Reshape({m, k});
        Array gout_matrix = gout.Reshape({m, n});
        return Dot(a_matrix.Transpose(), gout_matrix);
    };
    internal::SetUpOpNodes("dot", {a, b}, out, {a_backward_fn, b_backward_fn});

    return out;
}

}  // namespace xchainer
