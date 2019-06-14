#include "chainerx/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, nonstd::optional<Dtype> out_dtype) {
    Dtype real_out_dtype = out_dtype.has_value() ? *out_dtype : ResultType(a, b);

    if (a.ndim() == 0 || b.ndim() == 0) {
        return a * b;
    }

    Array modified_b{};

    Shape out_shape{};
    std::copy(a.shape().begin(), a.shape().end() - 1, std::back_inserter(out_shape));

    if (b.ndim() > 2) {
        std::vector<int> b_transpose_axes{};
        b_transpose_axes.reserve(b.ndim());
        std::vector<int> axes_index(b.ndim());
        std::iota(axes_index.begin(), axes_index.end(), 0);
        std::copy(axes_index.begin(), axes_index.end() - 2, std::back_inserter(b_transpose_axes));
        std::reverse_copy(axes_index.end() - 2, axes_index.end(), std::back_inserter(b_transpose_axes));

        Axes axes(b_transpose_axes.begin(), b_transpose_axes.end());
        modified_b = b.Transpose(axes);
        std::copy(modified_b.shape().begin(), modified_b.shape().end() - 1, std::back_inserter(out_shape));

        modified_b = modified_b.Reshape({-1, modified_b.shape().back()});
        modified_b = modified_b.Transpose();
    } else {
        std::copy(b.shape().begin() + 1, b.shape().end(), std::back_inserter(out_shape));
        modified_b = b;
    }

    int64_t k = a.shape()[a.ndim() - 1];
    if (modified_b.shape()[0] != k) {
        throw DimensionError{"Axis dimension mismatch"};
    }
    if (k == 0) {
        return Zeros(out_shape, real_out_dtype, a.device());
    }

    // Make each operand a matrix
    int64_t m = a.GetTotalSize() / k;
    int64_t n = b.GetTotalSize() / k;
    Array a_matrix = a.Reshape({m, k});
    Array b_matrix = modified_b.Reshape({k, n});

    // Matrix-matrix product
    Array out_matrix = Empty({m, n}, real_out_dtype, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<DotKernel>(a_matrix, b_matrix, out_matrix);
    }

    {
        BackwardBuilder bb{"dot", {a_matrix, b_matrix}, out_matrix};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([b_matrix_tok = bb.RetainInput(1), a_dtype = a.dtype()](BackwardContext& bctx) {
                const Array& b_matrix = bctx.GetRetainedInput(b_matrix_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Dot(gout, b_matrix.Transpose(), a_dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([a_matrix_tok = bb.RetainInput(0), b_dtype = b.dtype()](BackwardContext& bctx) {
                const Array& a_matrix = bctx.GetRetainedInput(a_matrix_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Dot(a_matrix.Transpose(), gout, b_dtype);
            });
        }
        bb.Finalize();
    }

    return out_matrix.Reshape(out_shape);
}

}  // namespace chainerx
