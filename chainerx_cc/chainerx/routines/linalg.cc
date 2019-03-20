#include "chainerx/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b) {
    Dtype out_dtype = ResultType(a, b);

    if (a.ndim() == 0 || b.ndim() == 0) {
        // TODO(hvy): Avoid unnecessary cast here when multiplication supports mixed dtypes.
        const Array& a_cast = a.dtype() == out_dtype ? a : a.AsType(out_dtype);
        const Array& b_cast = b.dtype() == out_dtype ? b : b.AsType(out_dtype);
        return a_cast * b_cast;
    }

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
        return Zeros(out_shape, out_dtype, a.device());
    }

    // Make each operand a matrix
    int64_t m = a.GetTotalSize() / k;
    int64_t n = b.GetTotalSize() / k;
    Array a_matrix = a.Reshape({m, k});
    Array b_matrix = b.Reshape({k, n});

    // Matrix-matrix product
    Array out_matrix = Empty({m, n}, out_dtype, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().Dot(a_matrix, b_matrix, out_matrix);
    }

    {
        BackwardBuilder bb{"dot", {a_matrix, b_matrix}, out_matrix};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([b_matrix_tok = bb.RetainInput(1), a_dtype = a.dtype()](BackwardContext& bctx) {
                const Array& b_matrix = bctx.GetRetainedInput(b_matrix_tok);
                const Array& gout = *bctx.output_grad();
                // TODO(hvy): Compute gradients by specifying accumulation/output dtype to Dot when it is supported.
                Array ga = Dot(gout, b_matrix.Transpose());
                if (ga.dtype() == a_dtype) {
                    bctx.input_grad() = std::move(ga);
                } else {
                    bctx.input_grad() = ga.AsType(a_dtype);
                }
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([a_matrix_tok = bb.RetainInput(0), b_dtype = b.dtype()](BackwardContext& bctx) {
                const Array& a_matrix = bctx.GetRetainedInput(a_matrix_tok);
                const Array& gout = *bctx.output_grad();
                // TODO(hvy): Compute gradients by specifying accumulation/output dtype to Dot when it is supported.
                Array gb = Dot(a_matrix.Transpose(), gout);
                if (gb.dtype() == b_dtype) {
                    bctx.input_grad() = std::move(gb);
                } else {
                    bctx.input_grad() = gb.AsType(b_dtype);
                }
            });
        }
        bb.Finalize();
    }

    return out_matrix.Reshape(out_shape);
}

}  // namespace chainerx
