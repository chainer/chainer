#include "chainerx/routines/linalg.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
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
        modified_b = Swapaxes(b, -1, -2);
        std::copy(modified_b.shape().begin(), modified_b.shape().end() - 1, std::back_inserter(out_shape));

        modified_b = modified_b.Reshape({b.GetTotalSize() > 0 ? -1 : 0, modified_b.shape().back()});
        modified_b = modified_b.Transpose();
    } else {
        std::copy(b.shape().begin() + 1, b.shape().end(), std::back_inserter(out_shape));
        modified_b = b;
    }

    int64_t k = a.shape()[a.ndim() - 1];
    if (modified_b.shape()[0] != k) {
        throw DimensionError{"Axis dimension mismatch between ", a.shape(), " and ", b.shape()};
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
        throw DimensionError{"ChainerX linear algebra routines only support 2-dimensional arrays."};
    }
}

void CheckSquareMatrix(const Array& a) {
    if (a.shape()[0] != a.shape()[1]) {
        throw DimensionError{"Matrix is not square."};
    }
}

void CheckUplo(char uplo) {
    char uplo_uppercase = toupper(uplo);
    if (uplo_uppercase != 'U' && uplo_uppercase != 'L') {
        throw ChainerxError{"UPLO argument must be 'L' or 'U'."};
    }
}

}  // namespace

Array Solve(const Array& a, const Array& b) {
    CheckRankTwoArray(a);
    CheckSquareMatrix(a);
    CheckEqual(a.device(), b.device());
    Dtype out_dtype = ResultType(a, b);
    Array out = Empty(b.shape(), out_dtype, b.device());

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
                bctx.input_grad() = -Dot(updim(Solve(a.Transpose(), gout).AsType(a_dtype)), updim(out).Transpose(), a_dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([a_tok = bb.RetainInput(0), b_dtype = b.dtype()](BackwardContext& bctx) {
                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Solve(a.Transpose(), gout).AsType(b_dtype);
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

std::tuple<Array, Array, Array> Svd(const Array& a, bool full_matrices, bool compute_uv) {
    CheckRankTwoArray(a);

    Array u{};
    Array s{};
    Array vt{};

    Shape u_shape;
    Shape vt_shape;
    int64_t m = a.shape()[0];
    int64_t n = a.shape()[1];
    int64_t k = std::min(m, n);

    if (compute_uv) {
        if (full_matrices) {
            u_shape = Shape{m, m};
            vt_shape = Shape{n, n};
        } else {
            u_shape = Shape{m, k};
            vt_shape = Shape{k, n};
        }
    } else {
        u_shape = Shape{0};
        vt_shape = Shape{0};
    }

    u = Empty(u_shape, a.dtype(), a.device());
    vt = Empty(vt_shape, a.dtype(), a.device());
    s = Empty(Shape{k}, a.dtype(), a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<SvdKernel>(a, u, s, vt, full_matrices, compute_uv);
    }

    // Reference:
    // https://j-towns.github.io/papers/svd-derivative.pdf
    {
        BackwardBuilder bb{"svd", a, {u, s, vt}};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([a_tok = bb.RetainInput(0),
                       u_tok = bb.RetainOutput(0),
                       s_tok = bb.RetainOutput(1),
                       vt_tok = bb.RetainOutput(2),
                       full_matrices,
                       compute_uv](BackwardContext& bctx) {
                if (full_matrices) {
                    throw ChainerxError{"ChainerX SVD differentiation is not implemented for full_matrices mode."};
                }
                if (!compute_uv) {
                    throw ChainerxError{"ChainerX SVD differentiation cannot be computed without u, vt matrices."};
                }

                const Array& a = bctx.GetRetainedInput(a_tok);
                const Array& u = bctx.GetRetainedOutput(u_tok);
                const Array& s = bctx.GetRetainedOutput(s_tok);
                const Array& vt = bctx.GetRetainedOutput(vt_tok);

                auto m = a.shape()[0];
                auto n = a.shape()[1];
                auto k = s.shape()[0];

                const Array& gu = bctx.output_grad(0).has_value() ? *bctx.output_grad(0) : Zeros(u.shape(), a.dtype(), a.device());
                const Array& gsigma = bctx.output_grad(1).has_value() ? *bctx.output_grad(1) : Zeros(s.shape(), a.dtype(), a.device());
                const Array& gvt = bctx.output_grad(2).has_value() ? *bctx.output_grad(2) : Zeros(vt.shape(), a.dtype(), a.device());

                Array v = vt.Transpose();
                Array gv = gvt.Transpose();

                Array sigma_term = Dot(Dot(u, Diag(gsigma)), vt);

                Array ut = u.Transpose();
                Array im = Eye(m, m, 0, a.dtype(), a.device());
                Array in = Eye(n, n, 0, a.dtype(), a.device());
                Array sigma_mat = Diag(s);
                Array sigma_mat_inv = Diag(Reciprocal(s));
                Array sigma_sq = Power(s, 2);
                Array f = ExpandDims(sigma_sq, 0) - ExpandDims(sigma_sq, 1);
                // Invert values of f, and fill the diagonal with 0s.
                // f has 0s on the diagonal, therefore fill it first with infinity.
                Array mask = Eye(f.shape()[0], f.shape()[1], 0, Dtype::kBool, a.device());
                f = Where(mask, std::numeric_limits<float>::infinity(), f);
                f = Reciprocal(f);

                Array u_term{};
                Array utgu = Dot(u.Transpose(), gu);
                u_term = Dot(Dot(u, f * (utgu - utgu.Transpose())), sigma_mat);
                if (m > k) {
                    u_term = u_term + Dot(Dot(im - Dot(u, ut), gu), sigma_mat_inv);
                }
                u_term = Dot(u_term, vt);

                Array v_term{};
                Array vtgv = Dot(vt, gv);
                v_term = Dot(Dot(sigma_mat, f * (vtgv - vtgv.Transpose())), vt);
                if (n > k) {
                    v_term = v_term + Dot(sigma_mat_inv, Dot(gvt, in - Dot(v, vt)));
                }
                v_term = Dot(u, v_term);

                bctx.input_grad() = u_term + sigma_term + v_term;
            });
        }
        bb.Finalize();
    }

    return std::make_tuple(std::move(u), std::move(s), std::move(vt));
}

Array PseudoInverse(const Array& a, float rcond) {
    CheckRankTwoArray(a);

    Array u{};
    Array s{};
    Array vt{};

    std::tie(u, s, vt) = Svd(a, /*full_matrices=*/false, /*compute_uv=*/true);

    // Computing the maximum along zero-sized axis is not supported
    // therefore return earlier
    if (a.shape().GetTotalSize() == 0) {
        // Copy instead of new empty array is used so that backward does not raise errors
        return Copy(a.Transpose());
    }

    Array cutoff = rcond * s.Max();
    Array cutoff_indices = s <= cutoff;

    Array sinv = Reciprocal(s);
    sinv = Where(cutoff_indices, 0, sinv);

    Array out = vt.Transpose().Dot(Diag(sinv)).Dot(u.Transpose());

    // Note that there is an analytical formula for the derivative:
    // https://mathoverflow.net/questions/25778/analytical-formula-for-numerical-derivative-of-the-matrix-pseudo-inverse
    // However, it does not hold if singular values are truncated based on rcond.

    return out;
}

std::tuple<Array, Array> Qr(const Array& a, QrMode mode) {
    CheckRankTwoArray(a);
    Device& device = a.device();
    Dtype dtype = internal::GetMathResultDtype(a.dtype());

    int64_t m = a.shape()[0];
    int64_t n = a.shape()[1];
    int64_t k = std::min(m, n);

    Array tau = Empty(Shape{k}, dtype, device);
    Shape q_shape{0};
    Shape r_shape{};
    switch (mode) {
        case QrMode::kReduced:
            q_shape = Shape{m, k};
            r_shape = Shape{k, n};
            break;
        case QrMode::kComplete:
            q_shape = Shape{m, m};
            r_shape = Shape{m, n};
            break;
        case QrMode::kR:
            r_shape = Shape{k, n};
            break;
        case QrMode::kRaw:
            r_shape = Shape{n, m};  // for "raw" mode r in the code corrensponds to h in docs
            break;
        default:
            throw ChainerxError{"Invalid QR mode"};
            break;
    }

    Array q = Empty(q_shape, dtype, device);
    Array r = Empty(r_shape, dtype, device);

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<QrKernel>(a, q, r, tau, mode);
    }

    // Backward of (Q, R) = QR(A):
    // dA = (dQ + Q * symmetrize(M)) * R^(-T), M = R * dR^T - dQ^T * Q
    {
        BackwardBuilder bb{"qr", a, {q, r}};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([q_tok = bb.RetainOutput(0), r_tok = bb.RetainOutput(1), mode](BackwardContext& bctx) {
                if (mode == QrMode::kR || mode == QrMode::kRaw) {
                    throw ChainerxError{"ChainerX QR differentiation is not implemented for 'r' or 'raw' modes."};
                }

                const Array& q = bctx.GetRetainedOutput(q_tok);
                const Array& r = bctx.GetRetainedOutput(r_tok);

                if (r.shape()[0] != r.shape()[1]) {
                    throw DimensionError{"ChainerX QR differentiation is not implemented for non-square R."};
                }

                const Array& gq = bctx.output_grad(0).has_value() ? *bctx.output_grad(0) : Zeros(q.shape(), q.dtype(), q.device());
                const Array& gr = bctx.output_grad(1).has_value() ? *bctx.output_grad(1) : Zeros(r.shape(), r.dtype(), r.device());

                Array m = Dot(r, gr.Transpose()) - Dot(gq.Transpose(), q);

                // Here a symmetric matrix is created from the square matrix M
                // by setting the upper triangle to be equal to the lower triangle, leaving
                // lower triangle and diagonal unchanged.
                Array m_sym = Tril(m, 0) + Tril(m, -1).Transpose();
                Array rhs = gq + Dot(q, m_sym);

                // Note that rhs * R^(-T) = (R^(-1) * rhs^T)^T
                bctx.input_grad() = Solve(r, rhs.Transpose()).Transpose();
            });
        }
        bb.Finalize();
    }

    if (mode == QrMode::kRaw) {
        return std::make_tuple(std::move(r), std::move(tau));
    }
    return std::make_tuple(std::move(q), std::move(r));
}

Array Cholesky(const Array& a) {
    CheckRankTwoArray(a);
    CheckSquareMatrix(a);
    Dtype dtype = internal::GetMathResultDtype(a.dtype());
    Array out = Empty(a.shape(), dtype, a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<CholeskyKernel>(a, out);
    }

    // Reference:
    // Differentiation of the Cholesky decomposition, Iain Murray https://arxiv.org/abs/1602.07527
    {
        BackwardBuilder bb{"cholesky", a, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([out_tok = bb.RetainOutput(0), a_dtype = a.dtype(), &a_device = a.device()](BackwardContext& bctx) {
                const Array& L = bctx.GetRetainedOutput(out_tok);
                const Array& gL = *bctx.output_grad();

                Array L_inv = Inverse(L);
                Array phi = Dot(L.Transpose(), gL);
                phi = Tril(phi, 0);
                Array mask = Eye(phi.shape()[0], phi.shape()[1], 0, Dtype::kBool, a_device);
                phi = Where(mask, 0.5 * Diag(phi), phi);

                Array gin = Dot(Dot(L_inv.Transpose(), phi), L_inv);

                bctx.input_grad() = (gin + gin.Transpose()) * 0.5;
            });
        }
        bb.Finalize();
    }

    return out;
}

std::tuple<Array, Array> Eigh(const Array& a, char uplo) {
    CheckRankTwoArray(a);
    CheckSquareMatrix(a);
    CheckUplo(uplo);

    Array w = Empty(Shape{a.shape()[0]}, a.dtype(), a.device());
    Array v = Empty(a.shape(), a.dtype(), a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<SyevdKernel>(a, w, v, uplo, /*compute_v=*/true);
    }

    // Reference:
    // Section 3.1 https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    {
        BackwardBuilder bb{"eigh", a, {w, v}};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([w_tok = bb.RetainOutput(0), v_tok = bb.RetainOutput(1), a_dtype = a.dtype(), &a_device = a.device()](
                              BackwardContext& bctx) {
                const Array& w = bctx.GetRetainedOutput(w_tok);
                const Array& v = bctx.GetRetainedOutput(v_tok);

                const Array& gw = bctx.output_grad(0).has_value() ? *bctx.output_grad(0) : Zeros(w.shape(), a_dtype, a_device);
                const Array& gv = bctx.output_grad(1).has_value() ? *bctx.output_grad(1) : Zeros(v.shape(), a_dtype, a_device);

                Array vt = v.Transpose();

                Array f = ExpandDims(w, 0) - ExpandDims(w, 1);
                // Invert values of `f`, and fill the diagonal with 0s.
                // `f` has 0s on the diagonal, therefore fill it first with infinity.
                Array mask = Eye(f.shape()[0], f.shape()[1], 0, Dtype::kBool, a_device);
                f = Where(mask, std::numeric_limits<float>::infinity(), f);
                f = Reciprocal(f);

                Array gin = Dot(Dot(v, f * Dot(vt, gv) + Diag(gw)), vt);
                bctx.input_grad() = 0.5 * (gin + gin.Transpose());
            });
        }
        bb.Finalize();
    }

    return std::make_tuple(std::move(w), std::move(v));
}

Array Eigvalsh(const Array& a, char uplo) {
    CheckRankTwoArray(a);
    CheckSquareMatrix(a);
    CheckUplo(uplo);

    Array w = Empty(Shape{a.shape()[0]}, a.dtype(), a.device());
    Array v = Empty(a.shape(), a.dtype(), a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<SyevdKernel>(a, w, v, uplo, /*compute_v=*/false);
    }

    return w;
}

}  // namespace chainerx
