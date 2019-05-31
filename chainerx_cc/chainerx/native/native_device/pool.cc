#include "chainerx/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/indexing.h"
#include "chainerx/kernels/math.h"
#include "chainerx/kernels/pooling.h"
#include "chainerx/kernels/reduction.h"
#include "chainerx/macro.h"
#include "chainerx/native/col2im.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/im2col.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/native/tensor_dot.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace native {
namespace {

Scalar GetLowestOrInf(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) {
        using T = typename decltype(pt)::type;
        return Scalar{NumericLimits<T>::LowestOrInf()};
    });
}

// Returns axes that does the following transpose.
// (batch_size, channel, a_1, a_2, ...., a_n, b_1, b_2, ..., b_n) -> (batch_size, channel, b_1, b_2, ...., b_n, a_1, a_2, ..., a_n).
Axes GetSwapSpatialDimensionsAxes(size_t n) {
    Axes axes;
    axes.resize(2 + 2 * n);  // E.g. (batch_size, channel, out_1, out_2, ..., out_n, k_1, k_2, ..., k_n).
    axes[0] = 0;  // Batch dimension kept as is.
    axes[1] = 1;  // Channel dimension kept as is.
    for (size_t i = 2; i < n + 2; ++i) {  // Output and kernel spatial dimensions to be swapped.
        axes[i] = n + i;
        axes[n + i] = i;
    }
    return axes;
}

class NativeMaxPoolKernel : public MaxPoolKernel {
public:
    std::tuple<Array, std::unique_ptr<MaxPoolGradState>> Call(
            const Array& x,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all,
            bool return_state,
            const nonstd::optional<Array>& out) override {
        CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());

        // TODO(hvy): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        // Convert to column representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        Array col = native_internal::Im2Col(x, kernel_size, stride, pad, cover_all, GetLowestOrInf(x.dtype()));
        Axes axes{};
        axes.resize(kernel_size.size());
        std::iota(axes.begin(), axes.end(), 2);

        Array actual_out = col.Max(axes);

        std::unique_ptr<MaxPoolGradState> state =
                return_state ? std::make_unique<NativeMaxPoolGradState>(x, std::move(col), std::move(axes)) : nullptr;

        return std::make_tuple(std::move(actual_out), std::move(state));
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(MaxPoolKernel, NativeMaxPoolKernel);

class NativeMaxPoolGradKernel : public MaxPoolGradKernel {
public:
    std::tuple<Array, std::unique_ptr<MaxPoolGradGradState>> Call(
            const Array& gout,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            const std::shared_ptr<MaxPoolGradState>& state,
            bool return_state,
            const nonstd::optional<Array>& gx) override {
        CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

        // TODO(hvy): Implement and test the `gx` argument.
        if (gx.has_value()) {
            throw NotImplementedError{"Passing gx as an argument is not yet supported."};
        }

        // TODO(hvy): Implement recomputation of state members.
        CHAINERX_ASSERT(state != nullptr);
        NativeMaxPoolGradState& native_state = dynamic_cast<NativeMaxPoolGradState&>(*state);
        const Array& x = native_state.x();
        const Array& col = native_state.col();
        const Axes& axes = native_state.axes();

        Array indices = col.ArgMax(axes);
        CHAINERX_ASSERT(indices.shape() == gout.shape());

        // Compute flattened col gradients.
        int64_t kernel_total_size = std::accumulate(kernel_size.begin(), kernel_size.end(), int64_t{1}, std::multiplies<>());
        int64_t out_total_size = indices.GetTotalSize();
        Shape out_flat{out_total_size};
        Device& device = x.device();
        Array gcol = Zeros({out_total_size * kernel_total_size}, x.dtype(), device);
        Array offset = Arange(0, out_total_size * kernel_total_size, kernel_total_size, indices.dtype(), device);
        device.backend().CallKernel<AddAtKernel>(gcol, indices.Reshape(out_flat) + offset, 0, gout.Reshape(out_flat), gcol);

        // Reshape col gradients to (batch_size, channel, out_1, out_2, ..., out_n, k_1, k_2, ..., k_n).
        Shape out_shape_with_kernel = gout.shape();
        std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(out_shape_with_kernel));

        // Transform col gradients to input shape.
        Array actual_gx = native_internal::Col2Im(
                gcol.Reshape(out_shape_with_kernel).Transpose(GetSwapSpatialDimensionsAxes(kernel_size.size())),
                stride,
                pad,
                {x.shape().begin() + 2, x.shape().end()});

        std::unique_ptr<MaxPoolGradGradState> grad_grad_state =
                return_state ? std::make_unique<NativeMaxPoolGradGradState>(std::move(indices), std::move(offset), x.dtype()) : nullptr;

        return std::make_tuple(std::move(actual_gx), std::move(grad_grad_state));
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(MaxPoolGradKernel, NativeMaxPoolGradKernel);

class NativeMaxPoolGradGradKernel : public MaxPoolGradGradKernel {
public:
    Array Call(
            const Array& ggx,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all,
            const std::shared_ptr<MaxPoolGradGradState>& state,
            const nonstd::optional<Array>& ggout) override {
        CHAINERX_ASSERT(internal::GetArrayBody(ggx)->nodes().empty());

        // TODO(hvy): Implement and test the `ggout` argument.
        if (ggout.has_value()) {
            throw NotImplementedError{"Passing ggout as an argument is not yet supported."};
        }

        // TODO(hvy): Implement recomputation of state members.
        CHAINERX_ASSERT(state != nullptr);
        NativeMaxPoolGradGradState& native_state = dynamic_cast<NativeMaxPoolGradGradState&>(*state);
        const Array& indices = native_state.indices();
        const Array& offset = native_state.offset();
        Dtype x_dtype = native_state.x_dtype();

        Array col = native_internal::Im2Col(ggx, kernel_size, stride, pad, cover_all, GetLowestOrInf(x_dtype));
        return Take(
                col.Transpose(GetSwapSpatialDimensionsAxes(kernel_size.size())).Reshape({col.GetTotalSize()}),
                indices + offset.Reshape(indices.shape()),
                0);
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(MaxPoolGradGradKernel, NativeMaxPoolGradGradKernel);

// TODO(hvy): Use Device::Mean when implemented.
void Mean(const Array& a, const Axes& axis, const Array& out) {
    Device& device = a.device();
    device.backend().CallKernel<SumKernel>(a, axis, out);
    device.backend().CallKernel<DivideASKernel>(out, internal::CountItemsAlongAxes(a.shape(), axis), out);
}

Array GetPadModeIgnorePoolingWidths(
        const Shape& shape,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        Dtype dtype) {
    int8_t n = shape.ndim() - 2;
    CHAINERX_ASSERT(n == static_cast<int8_t>(kernel_size.size()));
    CHAINERX_ASSERT(n == static_cast<int8_t>(stride.size()));
    CHAINERX_ASSERT(n == static_cast<int8_t>(pad.size()));
    CHAINERX_ASSERT(GetKind(dtype) == DtypeKind::kFloat);

    Array widths;
    for (int64_t i = 0; i < n; ++i) {
        int64_t dim_i = shape[2 + i];
        int64_t kernel_size_i = kernel_size[i];
        int64_t stride_i = stride[i];
        int64_t pad_i = pad[i];

        Array width = Empty({internal::GetConvOutDim(dim_i, kernel_size_i, stride_i, pad_i, false)}, dtype);
        VisitFloatingPointDtype(dtype, [dim_i, kernel_size_i, stride_i, pad_i, &width](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t i, T& w) {
                    T start = static_cast<T>(i) * s - p;
                    T end = start + k;
                    if (start < T{0}) {
                        start = T{0};
                    }
                    if (end > d) {
                        end = d;
                    }
                    w = end - start;
                }

                T d;
                T k;
                T s;
                T p;
            };
            Elementwise<T>(
                    Impl{static_cast<T>(dim_i), static_cast<T>(kernel_size_i), static_cast<T>(stride_i), static_cast<T>(pad_i)}, width);
        });

        if (i == 0) {
            widths = std::move(width);
        } else {
            Shape widths_expanded = widths.shape();
            widths_expanded.emplace_back(1);

            Shape width_expanded{1};
            std::copy(width.shape().begin(), width.shape().end(), std::back_inserter(width_expanded));

            widths = TensorDot(
                    widths.Reshape(widths_expanded), width.Reshape(width_expanded), {static_cast<int8_t>(widths.ndim())}, {0}, dtype);
        }
    }
    return widths;
}

class NativeAveragePoolKernel : public AveragePoolKernel {
public:
    std::tuple<Array, std::unique_ptr<AveragePoolGradState>> Call(
            const Array& x,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            AveragePoolPadMode pad_mode,
            bool return_state,
            const nonstd::optional<Array>& out) override {
        CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());

        // TODO(hvy): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        Array col = native_internal::Im2Col(x, kernel_size, stride, pad, false, 0);

        // Average along the kernel dimensions of col with shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        Axes kernel_axes{};
        kernel_axes.resize(kernel_size.size());
        std::iota(kernel_axes.begin(), kernel_axes.end(), 2);  // From k_1, up to k_n.

        Device& device = col.device();
        Array actual_out = internal::EmptyReduced(col.shape(), col.dtype(), kernel_axes, false, device);

        nonstd::optional<Array> width_ignore{nonstd::nullopt};

        switch (pad_mode) {
            case AveragePoolPadMode::kZero:
                Mean(col, kernel_axes, actual_out);
                break;
            case AveragePoolPadMode::kIgnore: {
                Device& device = x.device();
                device.backend().CallKernel<SumKernel>(col, kernel_axes, actual_out);
                width_ignore =
                        GetPadModeIgnorePoolingWidths(x.shape(), kernel_size, stride, pad, x.dtype()).BroadcastTo(actual_out.shape());
                device.backend().CallKernel<DivideKernel>(actual_out, *width_ignore, actual_out);
                break;
            }
            default:
                CHAINERX_NEVER_REACH();
        }

        std::unique_ptr<AveragePoolGradState> state =
                return_state ? std::make_unique<NativeAveragePoolGradState>(x, col.shape(), width_ignore) : nullptr;

        return std::make_tuple(std::move(actual_out), std::move(state));
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AveragePoolKernel, NativeAveragePoolKernel);

class NativeAveragePoolGradKernel : public AveragePoolGradKernel {
public:
    Array Call(
            const Array& gout,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            AveragePoolPadMode pad_mode,
            const std::shared_ptr<AveragePoolGradState>& state,
            const nonstd::optional<Array>& gx) override {
        CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

        // TODO(hvy): Implement and test the `gx` argument.
        if (gx.has_value()) {
            throw NotImplementedError{"Passing gx as an argument is not yet supported."};
        }

        CHAINERX_ASSERT(state != nullptr);
        NativeAveragePoolGradState& native_state = dynamic_cast<NativeAveragePoolGradState&>(*state);
        const Array& x = native_state.x();
        const Shape& gcol_shape = native_state.gcol_shape();

        Shape reshape_to = gcol_shape;
        std::fill(reshape_to.begin() + 2, reshape_to.begin() + x.ndim(), int64_t{1});
        Array actual_gx{};

        switch (pad_mode) {
            case AveragePoolPadMode::kZero: {
                Array gcol = gout.Reshape(reshape_to).BroadcastTo(gcol_shape);
                actual_gx = native_internal::Col2Im(gcol, stride, pad, {x.shape().begin() + 2, x.shape().end()});
                int64_t width_zero = std::accumulate(kernel_size.begin(), kernel_size.end(), int64_t{1}, std::multiplies<>());
                actual_gx /= width_zero;
                break;
            }
            case AveragePoolPadMode::kIgnore: {
                const Array& width_ignore = native_state.width_ignore().value();
                Array gcol = (gout / width_ignore).Reshape(reshape_to).BroadcastTo(gcol_shape);
                actual_gx = native_internal::Col2Im(gcol, stride, pad, {x.shape().begin() + 2, x.shape().end()});
                break;
            }
            default:
                CHAINERX_NEVER_REACH();
        }

        return actual_gx;
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AveragePoolGradKernel, NativeAveragePoolGradKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
