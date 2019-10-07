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
            int groups,
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

        if (groups > 1) {
            // Run grouped convolution.
            const int64_t G = groups;
            const int64_t N = x.shape()[0], iC = x.shape()[1];
            const int64_t oC = w.shape()[0];
            const int64_t iCg = iC / G;
            const int64_t oCg = oC / G;
            if (iC % G != 0) {
                throw ChainerxError("The number of groups(", G, ") must be a divisor of that of input channels(", iC, ")");
            }
            if (oC % G != 0) {
                throw ChainerxError("The number of groups(", G, ") must be a divisor of that of output channels(", oC, ")");
            }

            // Convert to colum representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
            Array nx = native_internal::Im2Col(x, kernel_size, stride, pad, cover_all, 0);
            const Dims o_size(nx.shape().end() - ndim, nx.shape().end());

            nx = RollAxis(nx, 0, ndim + 2);
            const int64_t mul_len = iCg * std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<int64_t>());
            nx = nx.Reshape({G, mul_len, N * std::accumulate(o_size.begin(), o_size.end(), 1, std::multiplies<int64_t>())});
            Array W = w.Reshape({G, oCg, mul_len});

            // (G, oCg, N*o_size) = (G, oCg, iCg*k_size) @ (G, iCg*k_size, N*o_size)
            std::vector<Array> y_data;
            for (int64_t i = 0; i < G; ++i) {
                y_data.push_back(W.At({i}).Dot(nx.At({i})));
            }
            Array y = Concatenate(y_data);
            Shape new_shape{oC, N};
            new_shape.insert(new_shape.end(), o_size.begin(), o_size.end());
            y = y.Reshape(new_shape);
            y = RollAxis(y, 1);

            if (b.has_value()) {
                Shape bias_shape;
                bias_shape.push_back(1);
                bias_shape.push_back(b->GetTotalSize());
                for (int8_t i = 0; i < ndim; ++i) {
                    bias_shape.push_back(1);
                }
                y += b->Reshape(bias_shape);
            }

            return y;
        } else {
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
            int groups,
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

        if (groups > 1) {
            int64_t G = groups;
            int64_t N = x.shape()[0], iC = x.shape()[1];
            int64_t oC = gy.shape()[1];
            const Shape o_size(gy.shape().begin() + 2, gy.shape().end());
            int64_t o_size_prod = std::accumulate(o_size.begin(), o_size.end(), 1, std::multiplies<int64_t>());
            int64_t iCg = iC / G;
            int64_t oCg = oC / G;

            // Im2Col
            // (N, iC, k_size..., o_size...)
            Array nx = native_internal::Im2Col(x, kernel_size, stride, pad, cover_all, 0);

            // Do not check iCg and oCg because this class is rarely used alone

            nx = RollAxis(nx, 0, ndim + 2);  // (iC, k_size..., N, o_size...)
            int64_t mul_len = iCg * std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<int64_t>());
            nx = nx.Reshape({G, mul_len, N * o_size_prod});
            nx = nx.Transpose({0, 2, 1});  // (G, N*o_size, iCg*k_size)

            chainerx::Array ngy = RollAxis(gy, 1, 0);  // (oC, N, o_size...)
            ngy = ngy.Reshape({G, oCg, N * o_size_prod});

            // (G, oCg, iCg*k_size) = (G, oCg, N*o_size) @ (G, N*o_size, iCg*k_size)
            Axes out_axes{0};
            Axes col_axes{0};
            for (int8_t i = 0; i < ndim; ++i) {
                out_axes.emplace_back(int64_t{2 + i});
                col_axes.emplace_back(int64_t{2 + ndim + i});
            }
            std::vector<Array> gW_data;
            for (int64_t i = 0; i < G; ++i) {
                gW_data.push_back(ngy.At({i}).Dot(nx.At({i})));
            }
            chainerx::Array gW = Concatenate(gW_data);
            Shape gw_nsize;
            gw_nsize.push_back(oC);
            gw_nsize.push_back(iCg);
            gw_nsize.insert(gw_nsize.end(), kernel_size.begin(), kernel_size.end());
            gW = gW.Reshape(gw_nsize);

            return gW;
        } else {
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
            int groups,
            const Dims& out_size,
            Dtype out_dtype,
            const absl::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        if (groups > 1) {
            // G: group count
            // N: batch size
            // xC: input channels
            // yC: output channels
            int const G = groups;
            int64_t const N = x.shape()[0], xC = x.shape()[1];
            std::vector<int64_t> const x_size(x.shape().begin() + 2, x.shape().end());
            int64_t const yCg = w.shape()[1];
            int64_t const yC = yCg * G;
            int64_t const xCg = xC / G;
            std::vector<int64_t> k_size(w.shape().begin() + 2, w.shape().end());
            int64_t const dims = k_size.size();
            if (xC % G != 0) {
                throw ChainerxError("The number of groups must be a divisor of that of input channels");
            }

            Array nx = RollAxis(x, 1);  // (xC, N, x_size...);
            nx = nx.Reshape({G, xCg, N * std::accumulate(x_size.begin(), x_size.end(), 1, std::multiplies<int64_t>())});

            Array W = w.Reshape({G, xCg, yCg * std::accumulate(k_size.begin(), k_size.end(), 1, std::multiplies<int64_t>())});
            W = W.Transpose({0, 2, 1});  // (G, yCg*k_size, xCg);

            // (G, yCg*k_size, N*x_size) = (G, yCg*k_size, xCg) @ (G, xCg, N*x_size);
            std::vector<Array> col_data;
            for (int64_t i = 0; i < G; ++i) {
                col_data.push_back(W.At({i}).Dot(nx.At({i})));
            }
            Array col = Concatenate(col_data);

            Shape col_shape;
            col_shape.push_back(yC);
            col_shape.insert(col_shape.end(), k_size.begin(), k_size.end());
            col_shape.push_back(N);
            col_shape.insert(col_shape.end(), x_size.begin(), x_size.end());
            col = col.Reshape(col_shape);
            col = RollAxis(col, dims + 1);  // (N, yC, k_size..., x_size...);

            Array y = native_internal::Col2Im(col, stride, pad, out_size);

            if (b.has_value()) {
                Shape s;
                s.push_back(1);
                s.push_back(yC);
                for (auto i = 0; i < dims; ++i) {
                    s.push_back(1);
                }
                y += b->Reshape(s);
            }
            return y;
        } else {
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
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(ConvTransposeKernel, NativeConvTransposeKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
