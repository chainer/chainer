#include "xchainer/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/backward.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"

namespace xchainer {

Array Dot(const Array& a, const Array& b) {
    if (a.ndim() == 0 || b.ndim() == 0) {
        return a * b;
    }

    // TODO(beam2d): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    // TODO(beam2d): Support it. Need to transpose b so that the inner-product axis is moved to the top.
    if (b.ndim() > 2) {
        throw NotImplementedError{"dot does not support rhs operand with ndim > 2"};
    }

    Shape out_shape{};
    std::copy(a.shape().begin(), a.shape().end() - 1, std::back_inserter(out_shape));
    std::copy(b.shape().begin() + 1, b.shape().end(), std::back_inserter(out_shape));

    int64_t k = a.shape()[a.ndim() - 1];
    if (b.shape()[0] != k) {
        throw DimensionError{"Axis dimension mismatch"};
    }
    if (k == 0) {
        return Zeros(out_shape, a.dtype(), a.device());
    }

    // Make each operand a matrix
    int64_t m = a.GetTotalSize() / k;
    int64_t n = b.GetTotalSize() / k;
    Array a_matrix = a.Reshape({m, k});
    Array b_matrix = b.Reshape({k, n});

    // Matrix-matrix product
    Array out_matrix = Empty({m, n}, a.dtype(), a.device());
    {
        NoBackpropModeScope scope{};
        a.device().Dot(a_matrix, b_matrix, out_matrix);
    }

    {
        BackwardBuilder bb{"dot", out_matrix};
        if (a_matrix.IsGradRequired(AnyGraph{})) {
            bb.Define({a_matrix}, [b_matrix](BackwardContext& bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = Dot(gout, b_matrix.Transpose());
            });
        }
        if (b_matrix.IsGradRequired(AnyGraph{})) {
            bb.Define({b_matrix}, [a_matrix](BackwardContext& bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = Dot(a_matrix.Transpose(), gout);
            });
        }
    }

    return out_matrix.Reshape(out_shape);
}

}  // namespace xchainer
