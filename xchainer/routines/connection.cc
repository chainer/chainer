#include "xchainer/routines/connection.h"

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/routines/math.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace internal {

int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return (in_dim + pad * 2 - kernel_size + stride - 1) / stride + 1;
    }
    return (in_dim + pad * 2 - kernel_size) / stride + 1;
}

int64_t GetConvTransposeOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return stride * (in_dim - 1) + kernel_size - stride + 1 - 2 * pad;
    }
    return stride * (in_dim - 1) + kernel_size - 2 * pad;
}

}  // namespace internal

namespace {

Array ConvGradW(
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    assert(w_shape.ndim() > 2);
    assert(x.ndim() == w_shape.ndim());
    assert(gy.ndim() == w_shape.ndim());
    assert(stride.size() == static_cast<size_t>(w_shape.ndim() - 2));
    assert(pad.size() == static_cast<size_t>(w_shape.ndim() - 2));
    Array out = x.device().ConvGradWeight(w_dtype, w_shape, x, gy, stride, pad, cover_all);

    auto x_backward_function =
            [ x_shape = x.shape(), gy, stride, pad ](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)->Array {
        StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
        assert(out_size.size() == stride.size());
        return ConvTranspose(gy.AsConstant(graph_ids_to_stop_gradient), gout, nonstd::nullopt, stride, pad, out_size);
    };
    auto gy_backward_function = [x, stride, pad, cover_all](
                                        const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) -> Array {
        return Conv(x.AsConstant(graph_ids_to_stop_gradient), gout, nonstd::nullopt, stride, pad, cover_all);
    };
    internal::SetUpOpNodes("conv-grad-weight", {x, gy}, out, {x_backward_function, gy_backward_function});

    return out;
}

void ConvCheckNdim(
        const Array& x, const Array& w, const StackVector<int64_t, kMaxNdim>& stride, const StackVector<int64_t, kMaxNdim>& pad) {
    if (w.ndim() != x.ndim()) {
        throw DimensionError{"Mismatched number of dimensions between input ", x.ndim(), " and weights ", w.ndim(), "."};
    }
    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
    if (ndim < 0) {
        throw DimensionError{"Number of spacial dimensions must be greater than or equal to 0"};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }
}

}  // namespace

Array Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    ConvCheckNdim(x, w, stride, pad);
    if (w.shape()[1] != x.shape()[1]) {
        throw DimensionError{"Mismatched number of input channels in input ", x.shape(), " and weights ", w.shape(), "."};
    }
    if (b.has_value() && (b->ndim() != 1 || b->shape()[0] != w.shape()[0])) {
        throw DimensionError{"Mismatched bias shape ", b->shape(), " for weights ", w.shape(), "."};
    }

    Array out = x.device().Conv(x, w, b, stride, pad, cover_all);
    auto x_backward_function =
            [ x_shape = x.shape(), w, stride, pad ](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)->Array {
        StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
        return ConvTranspose(gout, w.AsConstant(graph_ids_to_stop_gradient), nonstd::nullopt, stride, pad, out_size);
    };
    auto w_backward_function = [ w_dtype = w.dtype(), w_shape = w.shape(), x, stride, pad, cover_all ](
                                       const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)
                                       ->Array {
        return ConvGradW(w_dtype, w_shape, x.AsConstant(graph_ids_to_stop_gradient), gout, stride, pad, cover_all);
    };
    if (b.has_value()) {
        auto b_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array {
            Axes axis{0};
            for (int8_t i = 2; i < gout.ndim(); ++i) {
                axis.emplace_back(int64_t{i});
            }
            return Sum(gout, axis, false);
        };
        internal::SetUpOpNodes("conv", {x, w, *b}, out, {x_backward_function, w_backward_function, b_backward_function});
    } else {
        internal::SetUpOpNodes("conv", {x, w}, out, {x_backward_function, w_backward_function});
    }
    return out;
}

Array ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size) {
    ConvCheckNdim(x, w, stride, pad);
    if (x.shape()[1] != w.shape()[0]) {
        throw DimensionError{"Mismatched number of input channels in input ", x.shape(), " and weights ", w.shape(), "."};
    }
    if (b.has_value() && (b->ndim() != 1 || b->shape()[0] != w.shape()[1])) {
        throw DimensionError{"Mismatched bias shape ", b->shape(), " for weights ", w.shape(), "."};
    }
    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
    Shape in_dims{x.shape().begin() + 2, x.shape().end()};
    Shape kernel_size{w.shape().begin() + 2, w.shape().end()};

    bool cover_all = false;
    bool cover_all_determined = false;

    // Compute out_size if not specified
    StackVector<int64_t, kMaxNdim> real_out_size;
    if (out_size.has_value()) {
        real_out_size = *out_size;
    } else {
        cover_all_determined = true;
        for (int8_t i = 0; i < ndim; ++i) {
            real_out_size.emplace_back(internal::GetConvTransposeOutDim(in_dims[i], kernel_size[i], stride[i], pad[i], cover_all));
        }
    }
    for (int64_t size : real_out_size) {
        if (size < 0) {
            throw DimensionError{"All output sizes must be positive"};
        }
    }

    // Compute transposed convolution
    Array out = x.device().ConvTranspose(x, w, b, stride, pad, real_out_size);

    // Detect cover_all
    // TODO(niboshi): This logic is only required if x belongs to some graph.
    if (!cover_all_determined) {
        for (int8_t i = 0; i < ndim; ++i) {
            if (in_dims[i] != internal::GetConvOutDim(real_out_size[i], kernel_size[i], stride[i], pad[i], false)) {
                cover_all = true;
                break;
            }
        }
        cover_all_determined = true;

        // Check detected cover_all is consistent
        for (int8_t i = 0; i < ndim; ++i) {
            if (in_dims[i] != internal::GetConvOutDim(real_out_size[i], kernel_size[i], stride[i], pad[i], cover_all)) {
                throw DimensionError{"Output dims ", Shape{real_out_size.begin(), real_out_size.end()}, " are inconsistent."};
            }
        }
    }

    auto x_backward_function =
            [ x_shape = x.shape(), w, stride, pad, cover_all ](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)
                    ->Array {
        return Conv(gout, w.AsConstant(graph_ids_to_stop_gradient), nonstd::nullopt, stride, pad, cover_all);
    };
    auto w_backward_function = [ w_dtype = w.dtype(), w_shape = w.shape(), x, stride, pad, cover_all ](
                                       const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)
                                       ->Array {
        return ConvGradW(w_dtype, w_shape, gout, x.AsConstant(graph_ids_to_stop_gradient), stride, pad, cover_all);
    };
    if (b.has_value()) {
        auto b_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array {
            Axes axis{0};
            for (int8_t i = 2; i < gout.ndim(); ++i) {
                axis.emplace_back(int64_t{i});
            }
            return Sum(gout, axis, false);
        };
        internal::SetUpOpNodes("conv_transpose", {x, w, *b}, out, {x_backward_function, w_backward_function, b_backward_function});
    } else {
        internal::SetUpOpNodes("conv_transpose", {x, w}, out, {x_backward_function, w_backward_function});
    }
    return out;
}

}  // namespace xchainer
