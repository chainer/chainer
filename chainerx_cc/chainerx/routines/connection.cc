#include "chainerx/routines/connection.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dims.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernel_registry.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/math.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace internal {

int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    CHAINERX_ASSERT(stride > 0);
    int64_t numerator{0};
    if (cover_all) {
        numerator = in_dim + pad * 2 - kernel_size + stride - 1;
    } else {
        numerator = in_dim + pad * 2 - kernel_size;
    }
    if (numerator < 0) {
        throw DimensionError{"Output size should be positive."};
    }
    return numerator / stride + 1;
}

int64_t GetConvTransposeOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return stride * (in_dim - 1) + kernel_size - stride + 1 - 2 * pad;
    }
    return stride * (in_dim - 1) + kernel_size - 2 * pad;
}

}  // namespace internal

namespace {

Array ConvGradWeight(
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    CHAINERX_ASSERT(x.ndim() == w_shape.ndim());
    CHAINERX_ASSERT(gy.ndim() == w_shape.ndim());
    CHAINERX_ASSERT(stride.size() == static_cast<size_t>(w_shape.ndim() - 2));
    CHAINERX_ASSERT(pad.size() == static_cast<size_t>(w_shape.ndim() - 2));

    Array out{};
    {
        NoBackpropModeScope scope{};
        out = x.device().backend().CallKernel<ConvGradWeightKernel>(w_dtype, w_shape, x, gy, stride, pad, cover_all, nonstd::nullopt);
        CHAINERX_ASSERT(out.dtype() == w_dtype);
    }

    {
        BackwardBuilder bb{"conv-grad-weight", {x, gy}, out};

        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x_shape = x.shape(), x_dtype = x.dtype(), gy_tok = bb.RetainInput(1), stride, pad](BackwardContext& bctx) {
                const Array& gy = bctx.GetRetainedInput(gy_tok);
                const Array& gout = *bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                CHAINERX_ASSERT(out_size.size() == stride.size());
                bctx.input_grad() = ConvTranspose(gy, gout, nonstd::nullopt, stride, pad, out_size, x_dtype);
            });
        }

        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([gy_dtype = gy.dtype(), x_tok = bb.RetainInput(0), stride, pad, cover_all](BackwardContext& bctx) {
                const Array& x = bctx.GetRetainedInput(x_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Conv(x, gout, nonstd::nullopt, stride, pad, cover_all, gy_dtype);
            });
        }
        bb.Finalize();
    }

    return out;
}

void ConvCheckNdim(
        const Array& x, const Array& w, const StackVector<int64_t, kMaxNdim>& stride, const StackVector<int64_t, kMaxNdim>& pad) {
    if (w.ndim() != x.ndim()) {
        throw DimensionError{"Mismatched number of dimensions between input ", x.ndim(), " and weights ", w.ndim(), "."};
    }
    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 0) {
        throw DimensionError{"Number of spatial dimensions must be greater than or equal to 0"};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (std::any_of(stride.begin(), stride.end(), [](int64_t s) { return s <= 0; })) {
        throw DimensionError{"Stride elements must be greater than 0: ", DimsFormatter{stride}, "."};
    }
}

}  // namespace

Array Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        nonstd::optional<Dtype> out_dtype) {
    ConvCheckNdim(x, w, stride, pad);
    if (w.shape()[1] != x.shape()[1]) {
        throw DimensionError{"Mismatched number of input channels in input ", x.shape(), " and weights ", w.shape(), "."};
    }
    if (b.has_value() && (b->ndim() != 1 || b->shape()[0] != w.shape()[0])) {
        throw DimensionError{"Mismatched bias shape ", b->shape(), " for weights ", w.shape(), "."};
    }

    Dtype real_out_dtype = out_dtype.has_value() ? *out_dtype : b.has_value() ? ResultType(x, w, *b) : ResultType(x, w);

    Array out{};
    {
        NoBackpropModeScope scope{};
        out = x.device().backend().CallKernel<ConvKernel>(x, w, b, stride, pad, cover_all, real_out_dtype, nonstd::nullopt);
    }

    {
        // TODO(niboshi): Improve interface of BackwardBuilder for accepting optional input arrays.
        std::vector<ConstArrayRef> inputs{};
        if (b.has_value()) {
            inputs = {x, w, *b};
        } else {
            inputs = {x, w};
        }
        BackwardBuilder bb{"conv", std::move(inputs), out};

        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x_shape = x.shape(), x_dtype = x.dtype(), w_tok = bb.RetainInput(1), stride, pad](BackwardContext& bctx) {
                const Array& w = bctx.GetRetainedInput(w_tok);
                const Array& gout = *bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                bctx.input_grad() = ConvTranspose(gout, w, nonstd::nullopt, stride, pad, out_size, x_dtype);
            });
        }

        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([w_dtype = w.dtype(), w_shape = w.shape(), x_tok = bb.RetainInput(0), stride, pad, cover_all](BackwardContext& bctx) {
                const Array& x = bctx.GetRetainedInput(x_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = ConvGradWeight(w_dtype, w_shape, x, gout, stride, pad, cover_all);
            });
        }

        if (b.has_value()) {
            if (BackwardBuilder::Target bt = bb.CreateTarget(2)) {
                bt.Define([b_dtype = b->dtype()](BackwardContext& bctx) {
                    const Array& gout = *bctx.output_grad();
                    Axes axis{0};
                    for (int8_t i = 2; i < gout.ndim(); ++i) {
                        axis.emplace_back(int64_t{i});
                    }
                    // TODO(niboshi): Avoid AsType
                    bctx.input_grad() = Sum(gout, axis, false).AsType(b_dtype, false);
                });
            }
        }
        bb.Finalize();
    }

    return out;
}

Array ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size,
        nonstd::optional<Dtype> out_dtype) {
    ConvCheckNdim(x, w, stride, pad);
    if (x.shape()[1] != w.shape()[0]) {
        throw DimensionError{"Mismatched number of input channels in input ", x.shape(), " and weights ", w.shape(), "."};
    }
    if (b.has_value() && (b->ndim() != 1 || b->shape()[0] != w.shape()[1])) {
        throw DimensionError{"Mismatched bias shape ", b->shape(), " for weights ", w.shape(), "."};
    }
    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    Shape in_dims{x.shape().begin() + 2, x.shape().end()};
    Shape kernel_size{w.shape().begin() + 2, w.shape().end()};

    bool cover_all = false;

    // Compute out_size if not specified
    StackVector<int64_t, kMaxNdim> real_out_size;
    if (out_size.has_value()) {
        real_out_size = *out_size;

        for (int64_t size : real_out_size) {
            if (size < 0) {
                throw DimensionError{"All output sizes must be non-negative."};
            }
        }

        // Detect cover_all from out_size
        for (int8_t i = 0; i < ndim; ++i) {
            if (in_dims[i] != internal::GetConvOutDim(real_out_size[i], kernel_size[i], stride[i], pad[i], false)) {
                cover_all = true;
                break;
            }
        }
    } else {
        // cover_all is assumed to be false.
        for (int8_t i = 0; i < ndim; ++i) {
            int64_t out_dim = internal::GetConvTransposeOutDim(in_dims[i], kernel_size[i], stride[i], pad[i], cover_all);
            if (out_dim < 0) {
                throw DimensionError{"Inconsistent dimensions. Output dimension at axis ", i, " would be negative."};
            }
            real_out_size.emplace_back(out_dim);
        }
    }

    // Check out_size and cover_all are consistent
    for (int8_t i = 0; i < ndim; ++i) {
        if (in_dims[i] != internal::GetConvOutDim(real_out_size[i], kernel_size[i], stride[i], pad[i], cover_all)) {
            throw DimensionError{"Output dims ", Shape{real_out_size.begin(), real_out_size.end()}, " are incosistent."};
        }
    }

    Dtype real_out_dtype = out_dtype.has_value() ? *out_dtype : b.has_value() ? ResultType(x, w, *b) : ResultType(x, w);

    // Compute transposed convolution
    Array out{};
    {
        NoBackpropModeScope scope{};
        out = x.device().backend().CallKernel<ConvTransposeKernel>(x, w, b, stride, pad, real_out_size, real_out_dtype, nonstd::nullopt);
    }

    {
        // TODO(niboshi): Improve interface of BackwardBuilder for accepting optional input arrays.
        std::vector<ConstArrayRef> inputs{};
        if (b.has_value()) {
            inputs = {x, w, *b};
        } else {
            inputs = {x, w};
        }
        BackwardBuilder bb{"conv_transpose", std::move(inputs), out};

        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x_shape = x.shape(), x_dtype = x.dtype(), w_tok = bb.RetainInput(1), stride, pad, cover_all](BackwardContext& bctx) {
                const Array& w = bctx.GetRetainedInput(w_tok);
                const Array& gout = *bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                bctx.input_grad() = Conv(gout, w, nonstd::nullopt, stride, pad, cover_all, x_dtype);
            });
        }

        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([w_dtype = w.dtype(), w_shape = w.shape(), x_tok = bb.RetainInput(0), stride, pad, cover_all](BackwardContext& bctx) {
                const Array& x = bctx.GetRetainedInput(x_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = ConvGradWeight(w_dtype, w_shape, gout, x, stride, pad, cover_all);
            });
        }

        if (b.has_value()) {
            if (BackwardBuilder::Target bt = bb.CreateTarget(2)) {
                bt.Define([b_dtype = b->dtype()](BackwardContext& bctx) {
                    const Array& gout = *bctx.output_grad();
                    Axes axis{0};
                    for (int8_t i = 2; i < gout.ndim(); ++i) {
                        axis.emplace_back(int64_t{i});
                    }
                    // TODO(niboshi): Avoid AsType
                    bctx.input_grad() = Sum(gout, axis, false).AsType(b_dtype, false);
                });
            }
        }
        bb.Finalize();
    }

    return out;
}

Array Linear(const Array& x, const Array& w, const nonstd::optional<Array>& b, uint8_t n_batch_axes) {
    n_batch_axes = internal::NormalizeAxis(n_batch_axes, x.ndim());

    if (x.ndim() < 1) {
        throw DimensionError{"x.ndim should be greater than or equal to 1"};
    }
    if (w.ndim() != 2) {
        throw DimensionError{"w.ndim should be 2"};
    }
    if (b.has_value() && b->ndim() != 1) {
        throw DimensionError{"b.ndim should be 1"};
    }

    Dtype out_dtype = b.has_value() ? ResultType(x, w, *b) : ResultType(x, w);

    bool has_bias = b.has_value();
    int64_t out_dim = std::accumulate(x.shape().begin(), x.shape().begin() + n_batch_axes, int64_t{1}, std::multiplies<>());
    int64_t m_dim = w.shape()[0];
    int64_t n_dim = w.shape()[1];

    Shape out_shape{x.shape().begin(), x.shape().begin() + n_batch_axes};
    out_shape.emplace_back(m_dim);

    if (m_dim == 0 || n_dim == 0) {
        if (has_bias) {
            return b->AsType(out_dtype, false).BroadcastTo(out_shape);
        }
        return Zeros(out_shape, out_dtype, x.device());
    }

    Array x_matrix = x.Reshape({out_dim, n_dim});
    Array out_matrix = Empty({out_dim, m_dim}, out_dtype, x.device());
    Array b_matrix = has_bias ? b->BroadcastTo({out_dim, m_dim}) : Array{};

    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<DotKernel>(x_matrix, w.Transpose(), out_matrix);

        if (has_bias) {
            x.device().backend().CallKernel<AddKernel>(out_matrix, b_matrix.AsType(out_dtype, false), out_matrix);
        }
    }

    BackwardBuilder bb = has_bias ? BackwardBuilder{"linear", {x_matrix, w, b_matrix}, out_matrix}
                                  : BackwardBuilder{"linear_nobias", {x_matrix, w}, out_matrix};
    {
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([x_dtype = x.dtype(), w_matrix_tok = bb.RetainInput(1)](BackwardContext& bctx) {
                const Array& w_matrix = bctx.GetRetainedInput(w_matrix_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Dot(gout, w_matrix, x_dtype);
            });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([w_dtype = w.dtype(), x_matrix_tok = bb.RetainInput(0)](BackwardContext& bctx) {
                const Array& x_matrix = bctx.GetRetainedInput(x_matrix_tok);
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = Dot(gout.Transpose(), x_matrix, w_dtype);
            });
        }
        if (has_bias) {
            if (BackwardBuilder::Target bt = bb.CreateTarget(2)) {
                bt.Define([b_dtype = b->dtype()](BackwardContext& bctx) {
                    const Array& gout = *bctx.output_grad();
                    bctx.input_grad() = gout.AsType(b_dtype, false);
                });
            }
        }
        bb.Finalize();
    }
    CHAINERX_ASSERT(out_matrix.dtype() == out_dtype);
    return out_matrix.Reshape(out_shape);
}

}  // namespace chainerx
