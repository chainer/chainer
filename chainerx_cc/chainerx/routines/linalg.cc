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
#include "chainerx/routines/indexing.h"
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

namespace {

void CheckRankTwoArray(const Array& a) {
    if (a.ndim() != 2) {
        throw DimensionError{"ChainerX linear algebra routines support only 2-dimensional arrays."};
    }
}

void CheckSquareMatrix(const Array& a) {
    if (a.shape()[0] != a.shape()[1]) {
        throw DimensionError{"Matrix is not square."};
    }
}

}  // namespace

Array Solve(const Array& a, const Array& b) {
    CheckRankTwoArray(a);
    CheckSquareMatrix(a);
    CheckEqual(a.device(), b.device());
    CheckEqual(a.dtype(), b.dtype());
    Dtype dtype = internal::GetMathResultDtype(b.dtype());
    Array out = Empty(b.shape(), dtype, b.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<SolveKernel>(a, b, out);
    }

    // Reference:
    // https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    // Sec. 2.3.1 Matrix inverse product
    {
        BackwardBuilder bb{"solve", {a, b}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([out_tok = bb.RetainOutput(0), a_tok = bb.RetainInput(0), a_dtype = a.dtype()](BackwardContext& bctx) {
                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& out = bctx.GetRetainedOutput(out_tok);
                const Array& gout = *bctx.output_grad();
                auto updim = [&](const Array& x) {
                    if (x.ndim() == a.ndim()) {
                        return x;
                    }
                    return ExpandDims(x, 1);
                };
                bctx.input_grad() = -Dot(updim(Solve(a.Transpose(), gout)), updim(out).Transpose(), a_dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([a_tok = bb.RetainInput(0)](BackwardContext& bctx) {
                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Solve(a.Transpose(), gout);
            });
        }
        bb.Finalize();
    }

    return out;
}

Array Inverse(const Array& a) {
    CheckRankTwoArray(a);
    CheckSquareMatrix(a);
    Dtype dtype = internal::GetMathResultDtype(a.dtype());
    Array out = Empty(a.shape(), dtype, a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<InverseKernel>(a, out);
    }

    // Reference:
    // https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    // Sec. 2.2.3 Inverse
    {
        BackwardBuilder bb{"inv", a, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([out_tok = bb.RetainOutput(0), a_dtype = a.dtype()](BackwardContext& bctx) {
                const Array& out = bctx.GetRetainedOutput(out_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = -Dot(Dot(out.Transpose(), gout, a_dtype), out.Transpose(), a_dtype);
            });
        }
        bb.Finalize();
    }

    return out;
}

std::tuple<Array, Array, Array> SVD(const Array& a, bool full_matrices, bool compute_uv) {
    CheckRankTwoArray(a);
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
            bt.Define([a_tok = bb.RetainInput(0),
                       u_tok = bb.RetainOutput(0),
                       s_tok = bb.RetainOutput(1),
                       v_tok = bb.RetainOutput(2),
                       full_matrices,
                       compute_uv](BackwardContext& bctx) {
                if (full_matrices) {
                    throw ChainerxError{"ChainerX SVD differentiation is not implemented for full_matrices=true."};
                }
                if (!compute_uv) {
                    throw ChainerxError{"ChainerX SVD differentiation cannot be computed with compute_uv=false."};
                }

                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& u = bctx.GetRetainedOutput(u_tok);
                const Array& s = bctx.GetRetainedOutput(s_tok);
                const Array& vt = bctx.GetRetainedOutput(v_tok);

                auto m = a.shape()[0];
                auto n = a.shape()[1];
                auto k = s.shape()[0];

                const Array& gu = bctx.output_grad(0).has_value() ? *bctx.output_grad(0) : Zeros(u.shape(), a.dtype(), a.device());
                const Array& gsigma = bctx.output_grad(1).has_value() ? *bctx.output_grad(1) : Zeros(s.shape(), a.dtype(), a.device());
                const Array& gvt = bctx.output_grad(2).has_value() ? *bctx.output_grad(2) : Zeros(vt.shape(), a.dtype(), a.device());

                auto v = vt.Transpose();
                auto gv = gvt.Transpose();

                auto sigma_term = Dot(Dot(u, Diag(gsigma)), vt);

                auto ut = u.Transpose();
                auto im = Eye(m, m, 0, a.dtype(), a.device());
                auto in = Eye(n, n, 0, a.dtype(), a.device());
                auto sigma_mat = Diag(s);
                auto sigma_mat_inv = Diag(Power(s, -1));
                auto sigma_sq = Power(s, 2);
                auto F = ExpandDims(sigma_sq, 0) - ExpandDims(sigma_sq, 1);
                // Invert values of F, and fill the diagonal with 0s.
                // F has 0s on the diagonal, therefore fill it first with infinity.
                auto mask = Eye(F.shape()[0], F.shape()[1], 0, Dtype::kBool, a.device());
                F = Where(mask, INFINITY, F);
                F = Power(F, -1);

                Array u_term{};
                Array utgu = Dot(u.Transpose(), gu);
                u_term = Dot(Dot(u, F * (utgu - utgu.Transpose())), sigma_mat);
                if (m > k) {
                    u_term = u_term + Dot(Dot(im - Dot(u, ut), gu), sigma_mat_inv);
                }
                u_term = Dot(u_term, vt);

                Array v_term{};
                Array vtgv = Dot(vt, gv);
                v_term = Dot(Dot(sigma_mat, F * (vtgv - vtgv.Transpose())), vt);
                if (n > k) {
                    v_term = v_term + Dot(sigma_mat_inv, Dot(gvt, in - Dot(v, vt)));
                }
                v_term = Dot(u, v_term);

                bctx.input_grad() = u_term + sigma_term + v_term;
            });
        }
        bb.Finalize();
    }

    return std::make_tuple(std::move(u), std::move(s), std::move(v));
}

Array PseudoInverse(const Array& a, float rcond) {
    CheckRankTwoArray(a);
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
