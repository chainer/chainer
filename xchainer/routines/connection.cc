#include "xchainer/routines/connection.h"

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backward.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
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
    int8_t ndim = w_shape.ndim() - 2;  // Number of spacial dimensions
    assert(ndim > 0);
    assert(x.ndim() == ndim + 2);
    assert(gy.ndim() == ndim + 2);
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));
    Array out = x.device().ConvGradWeight(w_dtype, w_shape, x, gy, stride, pad, cover_all);

    {
        DefineBackwardScope bwd{"conv-grad-weight", {out}};

        if (!x.IsConstant()) {
            bwd.Define({x}, [ x_shape = x.shape(), gy, stride, pad ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                assert(out_size.size() == stride.size());
                bctx.input_grad() = ConvTranspose(bctx.Cut(gy), gout, nonstd::nullopt, stride, pad, out_size);
            });
        }

        if (!gy.IsConstant()) {
            bwd.Define({gy}, [x, stride, pad, cover_all](BackwardContext& bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = Conv(bctx.Cut(x), gout, nonstd::nullopt, stride, pad, cover_all);
            });
        }
    }

    return out;
}

}  // namespace

Array Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    Array out = x.device().Conv(x, w, b, stride, pad, cover_all);

    {
        DefineBackwardScope bwd{"conv", {out}};

        if (!x.IsConstant()) {
            bwd.Define({x}, [ x_shape = x.shape(), w, stride, pad ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                bctx.input_grad() = ConvTranspose(gout, bctx.Cut(w), nonstd::nullopt, stride, pad, out_size);
            });
        }

        if (!w.IsConstant()) {
            bwd.Define({w}, [ w_dtype = w.dtype(), w_shape = w.shape(), x, stride, pad, cover_all ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = ConvGradW(w_dtype, w_shape, bctx.Cut(x), gout, stride, pad, cover_all);
            });
        }

        if (b.has_value() && !b->IsConstant()) {
            bwd.Define({*b}, [](BackwardContext& bctx) {
                const Array& gout = bctx.output_grad();
                Axes axis{0};
                for (int8_t i = 2; i < gout.ndim(); ++i) {
                    axis.emplace_back(int64_t{i});
                }
                bctx.input_grad() = Sum(gout, axis, false);
            });
        }
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

    // Compute transposed convolution
    Array out = x.device().ConvTranspose(x, w, b, stride, pad, real_out_size);

    {
        DefineBackwardScope bwd{"conv_transpose", {out}};

        if (!x.IsConstant() || !w.IsConstant()) {
            // Detect cover_all
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
                        throw XchainerError{"Output dims ", Shape{real_out_size.begin(), real_out_size.end()}, " is incosistent."};
                    }
                }
            }

            if (!x.IsConstant()) {
                bwd.Define({x}, [ x_shape = x.shape(), w, stride, pad, cover_all ](BackwardContext & bctx) {
                    const Array& gout = bctx.output_grad();
                    StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                    bctx.input_grad() = Conv(gout, bctx.Cut(w), nonstd::nullopt, stride, pad, cover_all);
                });
            }

            if (!w.IsConstant()) {
                bwd.Define({w}, [ w_dtype = w.dtype(), w_shape = w.shape(), x, stride, pad, cover_all ](BackwardContext & bctx) {
                    const Array& gout = bctx.output_grad();
                    bctx.input_grad() = ConvGradW(w_dtype, w_shape, gout, bctx.Cut(x), stride, pad, cover_all);
                });
            }
        }

        if (b.has_value() && !b->IsConstant()) {
            bwd.Define({*b}, [](BackwardContext& bctx) {
                const Array& gout = bctx.output_grad();
                Axes axis{0};
                for (int8_t i = 2; i < gout.ndim(); ++i) {
                    axis.emplace_back(int64_t{i});
                }
                bctx.input_grad() = Sum(gout, axis, false);
            });
        }
    }

    return out;
}

}  // namespace xchainer
