#include "chainerx/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/native/col2im.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/im2col.h"
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

class NativeMaxPoolForwardBackward : public chainerx::MaxPoolForwardBackward {
public:
    explicit NativeMaxPoolForwardBackward(
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all)
        : kernel_size_{std::move(kernel_size)}, stride_{std::move(stride)}, pad_{std::move(pad)}, cover_all_{cover_all} {}

    Array Forward(const Array& x) override {
        CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());

        // Convert to column representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        col_ = native_internal::Im2Col(x, kernel_size_, stride_, pad_, cover_all_, GetLowestOrInf(x.dtype()));
        axes_.resize(kernel_size_.size());
        std::iota(axes_.begin(), axes_.end(), 2);
        x_ = x;
        return col_.Max(axes_);
    }

    Array Backward(const Array& gout) override {
        CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

        indices_ = col_.ArgMax(axes_);
        CHAINERX_ASSERT(indices_.shape() == gout.shape());

        // Compute flattened col gradients.
        int64_t kernel_total_size = std::accumulate(kernel_size_.begin(), kernel_size_.end(), int64_t{1}, std::multiplies<>());
        int64_t out_total_size = indices_.GetTotalSize();
        Shape out_flat{out_total_size};
        Device& device = x_.device();
        Array gcol = Zeros({out_total_size * kernel_total_size}, x_.dtype(), device);
        offset_ = Arange(0, out_total_size * kernel_total_size, kernel_total_size, indices_.dtype(), device);
        device.AddAt(gcol, indices_.Reshape(out_flat) + offset_, 0, gout.Reshape(out_flat), gcol);

        // Reshape col gradients to (batch_size, channel, out_1, out_2, ..., out_n, k_1, k_2, ..., k_n).
        Shape out_shape_with_kernel = gout.shape();
        std::copy(kernel_size_.begin(), kernel_size_.end(), std::back_inserter(out_shape_with_kernel));

        // Transform col gradients to input shape.
        return native_internal::Col2Im(
                gcol.Reshape(out_shape_with_kernel).Transpose(GetSwapSpatialDimensionsAxes(kernel_size_.size())),
                stride_,
                pad_,
                {x_.shape().begin() + 2, x_.shape().end()});
    }

    Array DoubleBackward(const Array& ggx) override {
        CHAINERX_ASSERT(internal::GetArrayBody(ggx)->nodes().empty());

        Array col = native_internal::Im2Col(ggx, kernel_size_, stride_, pad_, cover_all_, GetLowestOrInf(x_.dtype()));
        return Take(
                col.Transpose(GetSwapSpatialDimensionsAxes(kernel_size_.size())).Reshape({col.GetTotalSize()}),
                indices_ + offset_.Reshape(indices_.shape()),
                0);
    }

private:
    const StackVector<int64_t, kMaxNdim> kernel_size_;
    const StackVector<int64_t, kMaxNdim> stride_;
    const StackVector<int64_t, kMaxNdim> pad_;
    Array x_;
    bool cover_all_;
    Array col_{};
    Axes axes_{};
    Array indices_{};
    Array offset_{};
};

}  // namespace

std::unique_ptr<MaxPoolForwardBackward> NativeDevice::GetMaxPoolForwardBackward(
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    return std::make_unique<NativeMaxPoolForwardBackward>(kernel_size, stride, pad, cover_all);
}

namespace {

// TODO(hvy): Use Device::Mean when implemented.
void Mean(const Array& a, const Axes& axis, const Array& out) {
    Device& device = a.device();
    device.Sum(a, axis, out);
    device.DivideAS(out, internal::CountItemsAlongAxes(a.shape(), axis), out);
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

            widths = TensorDot(widths.Reshape(widths_expanded), width.Reshape(width_expanded), {static_cast<int8_t>(widths.ndim())}, {0});
        }
    }
    return widths;
}

class NativeAveragePoolForwardBackward : public chainerx::AveragePoolForwardBackward {
public:
    explicit NativeAveragePoolForwardBackward(
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            AveragePoolPadMode pad_mode)
        : kernel_size_{std::move(kernel_size)}, stride_{std::move(stride)}, pad_{std::move(pad)}, pad_mode_{pad_mode} {}

    Array Forward(const Array& x) override {
        CHAINERX_ASSERT(internal::GetArrayBody(x)->nodes().empty());

        Array col = native_internal::Im2Col(x, kernel_size_, stride_, pad_, false, 0);

        // Average along the kernel dimensions of col with shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        Axes kernel_axes;
        kernel_axes.resize(kernel_size_.size());
        std::iota(kernel_axes.begin(), kernel_axes.end(), 2);  // From k_1, up to k_n.

        Array out = internal::EmptyReduced(col.shape(), col.dtype(), kernel_axes, false, col.device());

        switch (pad_mode_) {
            case AveragePoolPadMode::kZero:
                Mean(col, kernel_axes, out);
                break;
            case AveragePoolPadMode::kIgnore: {
                Device& device = x.device();
                device.Sum(col, kernel_axes, out);
                width_ignore_ = GetPadModeIgnorePoolingWidths(x.shape(), kernel_size_, stride_, pad_, x.dtype()).BroadcastTo(out.shape());
                device.Divide(out, width_ignore_, out);
                break;
            }
            default:
                CHAINERX_NEVER_REACH();
        }
        x_ = x;
        gcol_shape_ = col.shape();
        return out;
    }

    Array Backward(const Array& gout) override {
        CHAINERX_ASSERT(internal::GetArrayBody(gout)->nodes().empty());

        Shape reshape_to = gcol_shape_;
        std::fill(reshape_to.begin() + 2, reshape_to.begin() + x_.ndim(), int64_t{1});
        Array gx{};
        switch (pad_mode_) {
            case AveragePoolPadMode::kZero: {
                Array gcol = gout.Reshape(reshape_to).BroadcastTo(gcol_shape_);
                gx = native_internal::Col2Im(gcol, stride_, pad_, {x_.shape().begin() + 2, x_.shape().end()});
                int64_t width_zero = std::accumulate(kernel_size_.begin(), kernel_size_.end(), int64_t{1}, std::multiplies<>());
                gx /= width_zero;
                break;
            }
            case AveragePoolPadMode::kIgnore: {
                Array gcol = (gout / width_ignore_).Reshape(reshape_to).BroadcastTo(gcol_shape_);
                gx = native_internal::Col2Im(gcol, stride_, pad_, {x_.shape().begin() + 2, x_.shape().end()});
                break;
            }
            default:
                CHAINERX_NEVER_REACH();
        }
        return gx;
    }

private:
    const StackVector<int64_t, kMaxNdim> kernel_size_;
    const StackVector<int64_t, kMaxNdim> stride_;
    const StackVector<int64_t, kMaxNdim> pad_;
    const AveragePoolPadMode pad_mode_;
    Array x_;
    Shape gcol_shape_;
    Array width_ignore_;
};

}  // namespace

std::unique_ptr<AveragePoolForwardBackward> NativeDevice::GetAveragePoolForwardBackward(
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        AveragePoolPadMode pad_mode) {
    return std::make_unique<NativeAveragePoolForwardBackward>(kernel_size, stride, pad, pad_mode);
}

}  // namespace native
}  // namespace chainerx
