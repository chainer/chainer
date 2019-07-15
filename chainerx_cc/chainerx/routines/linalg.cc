#include "chainerx/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

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
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

Array Dot(const Array& a, const Array& b, absl::optional<Dtype> out_dtype) {
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

std::tuple<Array, Array, Array> SVD(const Array& a, bool full_matrices, bool compute_uv) {
    if (a.ndim() != 2) {
        throw DimensionError{"ChainerX SVD supports only 2-dimensional arrays."};
    }
    Array u{};
    Array s{};
    Array v{};

    {
        NoBackpropModeScope scope{};
        std::tie(u, s, v) = a.device().backend().CallKernel<SVDKernel>(a, full_matrices, compute_uv);
    }

    // Reference:
    // https://j-towns.github.io/papers/svd-derivative.pdf
    {
        BackwardBuilder bb{"svd", a, {u, s, v}};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([a_tok = bb.RetainInput(0), u_tok = bb.RetainOutput(0), s_tok = bb.RetainOutput(1), v_tok = bb.RetainOutput(2), full_matrices, compute_uv](
                              BackwardContext& bctx) {
                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& u = bctx.GetRetainedOutput(u_tok);
                const Array& s = bctx.GetRetainedOutput(s_tok);
                const Array& v = bctx.GetRetainedOutput(v_tok);

                auto m = a.shape()[0];
                auto n = a.shape()[1];
                // auto k = s.shape()[0];

                const Array& gu = bctx.output_grad(0).has_value() ? *bctx.output_grad(0) : Zeros(u.shape(), a.dtype(), a.device());
                const Array& gsigma = bctx.output_grad(1).has_value() ? *bctx.output_grad(1) : Zeros(s.shape(), a.dtype(), a.device());
                const Array& gv = bctx.output_grad(2).has_value() ? *bctx.output_grad(2) : Zeros(v.shape(), a.dtype(), a.device());

                const Array& vt = v.Transpose();

                Array sigma_term{};
                if (bctx.output_grad(1).has_value()) {
                    sigma_term = Dot(Dot(u, Diag(gsigma)), vt);
                } else {
                    sigma_term = Zeros(s.shape(), a.dtype(), a.device());
                }

                auto ut = u.Transpose();
                auto im = Eye(m, m, 0, a.dtype(), a.device());
                auto in = Eye(n, n, 0, a.dtype(), a.device());
                auto sigma_mat = Diag(s);
                auto sigma_mat_inv = Diag(Power(s, -1));
                auto sigma_sq = Power(s, 2);
                auto F = ExpandDims(sigma_sq, 0) - ExpandDims(sigma_sq, 1);
                // Invert values of F, and fill the diagonal with 0s.
                // F has 0s on the diagonal, therefore fill it first with infinity.
                // F.FillDiagonal(INFINITY);
                F = Power(F, -1);

            });
        }
        bb.Finalize();
    }

    return std::make_tuple(std::move(u), std::move(s), std::move(v));
}

Array PseudoInverse(const Array& a, float rcond) {
    if (a.ndim() != 2) {
        throw DimensionError{"ChainerX pseudo-inverse supports only 2-dimensional arrays."};
    }
    Dtype dtype = internal::GetMathResultDtype(a.dtype());
    Array out = Empty(Shape({a.shape()[1], a.shape()[0]}), dtype, a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<PseudoInverseKernel>(a, out, rcond);
    }

    // Reference:
    // https://mathoverflow.net/questions/25778/analytical-formula-for-numerical-derivative-of-the-matrix-pseudo-inverse
    {
        BackwardBuilder bb{"pinv", a, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([a_tok = bb.RetainInput(0), out_tok = bb.RetainOutput(0), a_dtype = a.dtype(), &a_device = a.device()](
                              BackwardContext& bctx) {
                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& out = bctx.GetRetainedOutput(out_tok);
                const Array& gout = *bctx.output_grad();
                Array I_a = Identity(a.shape()[0], a_dtype, a_device);
                Array I_out = Identity(out.shape()[0], a_dtype, a_device);
                bctx.input_grad() = (-Dot(Dot(out, gout.Transpose()), out) + Dot(Dot(Dot(out, out.Transpose()), gout), I_a - Dot(a, out)) +
                                     Dot(Dot(Dot(I_out - Dot(out, a), gout), out.Transpose()), out))
                                            .Transpose();
            });
        }
        bb.Finalize();
    }

    return out;
}

}  // namespace chainerx
