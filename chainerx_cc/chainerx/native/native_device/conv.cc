#include "chainerx/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

#include <absl/types/optional.h>
#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/dims.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/macro.h"
#include "chainerx/native/col2im.h"
#include "chainerx/native/im2col.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/native/tensor_dot.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Conv)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(ConvTranspose)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(ConvGradWeight)
}  // namespace internal

namespace native {
namespace {

class NativeConvKernel : public ConvKernel {
public:
    Array Call(
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            Dtype out_dtype,
            const absl::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        int8_t ndim = w.ndim() - 2;  // Number of spatial dimensions

        // Compute the kernel size from the weight array.
        Dims kernel_size;
        std::copy_n(w.shape().begin() + 2, ndim, std::back_inserter(kernel_size));

        // Convert to colum representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        Array col = native_internal::Im2Col(x, kernel_size, stride, pad, cover_all, 0);

        // Compute the tensor dot product of col and w, reducing (channel, k_1, k_2, ..., k_n).
        Axes axes;
        axes.resize(ndim + 1);
        std::iota(axes.begin(), axes.end(), 1);
        Array y = TensorDot(col, w, axes, axes, out_dtype);  // (batch_size, out_1, out_2, ..., out_n, out_channel)

        // Add bias, if given.
        if (b.has_value()) {
            // TODO(niboshi): Remove AsType when += supports dtype promotion.
            y += b->AsType(y.dtype(), false);
        }

        // Move the out channel axis to the second
        Axes roll_axes;
        roll_axes.resize(y.ndim());
        roll_axes[0] = 0;
        roll_axes[1] = ndim + 1;
        std::iota(roll_axes.begin() + 2, roll_axes.end(), 1);
        Array actual_out = y.Transpose(roll_axes);

        CHAINERX_ASSERT(actual_out.dtype() == out_dtype);
        return actual_out;
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ConvKernel, NativeConvKernel);

class NativeConvGradWeightKernel : public ConvGradWeightKernel {
public:
    Array Call(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            const absl::optional<Array>& out) override {
        CHAINERX_ASSERT(x.ndim() == w_shape.ndim());

        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions

        // Compute the kernel size
        Dims kernel_size{w_shape.begin() + 2, w_shape.end()};

        // Im2Col
        Array col = native_internal::Im2Col(x, kernel_size, stride, pad, cover_all, 0);

        // TensorDot
        Axes out_axes{0};
        Axes col_axes{0};
        for (int8_t i = 0; i < ndim; ++i) {
            out_axes.emplace_back(int64_t{2 + i});
            col_axes.emplace_back(int64_t{2 + ndim + i});
        }
        return TensorDot(gy, col, out_axes, col_axes, w_dtype);
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ConvGradWeightKernel, NativeConvGradWeightKernel);

class NativeConvTransposeKernel : public ConvTransposeKernel {
public:
    Array Call(
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            const Dims& out_size,
            Dtype out_dtype,
            const absl::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        Array col = TensorDot(w, x, {0}, {1}, out_dtype);  // shape: out_channel, k_1, ..., k_n, batch_size, out_1, ..., out_n
        col = RollAxis(col, x.ndim() - 1);  // batch axis is rolled to the top

        Array actual_out = native_internal::Col2Im(col, stride, pad, out_size);  // shape: batch_size, out_channel, out_size...

        // Add bias, if given.
        if (b.has_value()) {
            std::vector<ArrayIndex> slice{NewAxis{}, Slice{}};
            for (size_t i = 0; i < out_size.size(); ++i) {
                slice.emplace_back(NewAxis{});
            }
            // TODO(niboshi): Remove AsType when += supports dtype promotion.
            actual_out += b->At(slice).AsType(out_dtype);
        }

        CHAINERX_ASSERT(actual_out.dtype() == out_dtype);
        return actual_out;
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ConvTransposeKernel, NativeConvTransposeKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
