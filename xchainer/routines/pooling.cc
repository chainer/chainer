#include "xchainer/routines/pooling.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/routines/math.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace {

void CheckPoolInputs(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad) {
    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions.
    if (static_cast<int8_t>(kernel_size.size()) != ndim) {
        throw DimensionError{"Wrong numbers of kernel size dimensions ", kernel_size.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }
}

}  // namespace

Array MaxPool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    CheckPoolInputs(x, kernel_size, stride, pad);
    std::unique_ptr<MaxPoolForwardBackward> fb = x.device().GetMaxPoolForwardBackward();
    Array out = fb->Forward(x, kernel_size, stride, pad, cover_all);

    // Supporting arbitrary number of backwards using a recursive definition.
    // TODO(hvy): Test backward of double backward.
    struct MaxPoolBwd {
        Array operator()(const Array& gout, const std::vector<GraphId>&) {
            Array gx = fb->Backward(x, kernel_size, stride, pad, cover_all, gout);
            auto double_backward_function = [this, gout](const Array& ggx, const std::vector<GraphId>&) {
                Array ggout = fb->DoubleBackward(x, kernel_size, stride, pad, cover_all, gout, ggx);
                // Make ggout further backpropable.
                internal::SetUpOpNodes("max_pooling_double_backward", {ggx}, ggout, {*this});
                return ggout;
            };
            internal::SetUpOpNodes("max_pooling_backward", {gout}, gx, {double_backward_function});
            return gx;
        }

        Array x;
        StackVector<int64_t, kMaxNdim> kernel_size;
        StackVector<int64_t, kMaxNdim> stride;
        StackVector<int64_t, kMaxNdim> pad;
        bool cover_all;
        std::shared_ptr<MaxPoolForwardBackward> fb;
    };

    internal::SetUpOpNodes("max_pooling", {x}, out, {MaxPoolBwd{x, kernel_size, stride, pad, cover_all, std::move(fb)}});
    return out;
}

Array AveragePool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        bool count_include_pad) {
    CheckPoolInputs(x, kernel_size, stride, pad);
    // TODO(hvy): Implement backward.
    return x.device().AveragePool(x, kernel_size, stride, pad, cover_all, count_include_pad);
}

}  // namespace xchainer
