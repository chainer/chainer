#include "chainerx/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"

namespace chainerx {

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
        BackwardBuilder bb{"dot", {a_matrix, b_matrix}, out_matrix};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([b_matrix_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& b_matrix = bctx.GetRetainedInput(b_matrix_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Dot(gout, b_matrix.Transpose());
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([a_matrix_tok = bb.RetainInput(0)](BackwardContext& bctx) {
                const Array& a_matrix = bctx.GetRetainedInput(a_matrix_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Dot(a_matrix.Transpose(), gout);
            });
        }
        bb.Finalize();
    }

    return out_matrix.Reshape(out_shape);
}

Array Linear(const Array& x, const Array& w, const nonstd::optional<Array>& b, uint8_t n_batch_axes) {
    n_batch_axes = internal::NormalizeAxis(n_batch_axes, x.ndim());

    // TODO(imanishi): dtype conversion
    CheckEqual(x.dtype(), w.dtype());
    if (b.has_value()) {
        CheckEqual(x.dtype(), b->dtype());
    }

    if (w.ndim() != 2) {
        throw DimensionError{"w.ndim should be 2"};
    }
    if (b.has_value() && b->ndim() != 1) {
        throw DimensionError{"b.ndim should be 1"};
    }

    int64_t out_dim = std::accumulate(x.shape().begin(), x.shape().begin() + n_batch_axes, int64_t{1}, std::multiplies<>());
    int64_t m_dim = w.shape()[0];
    int64_t n_dim = w.shape()[1];

    Shape out_shape{};
    std::copy(x.shape().begin(), x.shape().begin() + n_batch_axes, std::back_inserter(out_shape));
    out_shape.emplace_back(m_dim);

    if (m_dim == 0 || n_dim == 0) {
        if (b.has_value()) {
            return b->BroadcastTo(out_shape);
        }
        return Zeros(out_shape, x.dtype(), x.device());
    }

    Array x_matrix = x.Reshape({out_dim, n_dim});
    Array out_matrix = Empty({out_dim, m_dim}, x.dtype(), x.device());

    if (b.has_value()) {
        Array b_matrix = b->BroadcastTo({out_dim, m_dim});
        {
            NoBackpropModeScope scope{};
            x.device().Dot(x_matrix, w.Transpose(), out_matrix);
            x.device().Add(out_matrix, b_matrix, out_matrix);
        }
        BackwardBuilder bb{"linear", {x_matrix, w, b_matrix}, out_matrix};
        {
            if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
                bt.Define([w_matrix_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                    const Array& w_matrix = bctx.GetRetainedInput(w_matrix_tok);
                    const Array& gout = *bctx.output_grad();
                    bctx.input_grad() = Dot(gout, w_matrix);
                });
            }
            if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
                bt.Define([x_matrix_tok = bb.RetainInput(0)](BackwardContext& bctx) {
                    const Array& x_matrix = bctx.GetRetainedInput(x_matrix_tok);
                    const Array& gout = *bctx.output_grad();
                    bctx.input_grad() = Dot(gout.Transpose(), x_matrix);
                });
            }
            if (BackwardBuilder::Target bt = bb.CreateTarget(2)) {
                bt.Define([](BackwardContext& bctx) {
                    const Array& gout = *bctx.output_grad();
                    bctx.input_grad() = gout;
                });
            }
            bb.Finalize();
        }
    } else {
        {
            NoBackpropModeScope scope{};
            x.device().Dot(x_matrix, w.Transpose(), out_matrix);
        }
        BackwardBuilder bb{"linear_nobias", {x_matrix, w}, out_matrix};
        {
            if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
                bt.Define([w_matrix_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                    const Array& w_matrix = bctx.GetRetainedInput(w_matrix_tok);
                    const Array& gout = *bctx.output_grad();
                    bctx.input_grad() = Dot(gout, w_matrix);
                });
            }
            if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
                bt.Define([x_matrix_tok = bb.RetainInput(0)](BackwardContext& bctx) {
                    const Array& x_matrix = bctx.GetRetainedInput(x_matrix_tok);
                    const Array& gout = *bctx.output_grad();
                    bctx.input_grad() = Dot(gout.Transpose(), x_matrix);
                });
            }
            // LinearGradWeight(bb);
            bb.Finalize();
        }
    }
    return out_matrix.Reshape(out_shape);
}

}  // namespace chainerx
