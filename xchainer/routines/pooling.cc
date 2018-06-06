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

Array MaxPool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
    if (static_cast<int8_t>(kernel_size.size()) != ndim) {
        throw DimensionError{"Wrong numbers of kernel size dimensions ", kernel_size.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }

    std::shared_ptr<MaxPoolForwardBackward> fb = x.device().GetMaxPoolForwardBackward();

    Array out = fb->Forward(x, kernel_size, stride, pad, cover_all);

    auto backward_function = [=](const Array& gout, const std::vector<GraphId>&) -> Array {
        Array gx = fb->Backward(x, kernel_size, stride, pad, cover_all, gout);
        auto double_backward_function = [ =, fb = std::move(fb) ](const Array& ggx, const std::vector<GraphId>&)->Array {
            return fb->DoubleBackward(x, kernel_size, stride, pad, cover_all, gout, ggx);  // ggout.
        };
        internal::SetUpOpNodes("max_pooling_backward", {gout}, gx, {double_backward_function});
        return gx;
    };
    internal::SetUpOpNodes("max_pooling", {x}, out, {backward_function});
    return out;
}

}  // namespace xchainer
