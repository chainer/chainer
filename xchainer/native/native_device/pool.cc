#include "xchainer/native/native_device.h"

#include <cstdint>
#include <memory>
#include <numeric>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/native/im2col.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/scalar.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace {

class NativeMaxPoolForwardBackward : public xchainer::MaxPoolForwardBackward {
public:
    Array Forward(
            const Array& x,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override {
        Scalar min = VisitDtype(x.dtype(), [](auto pt) {
            using T = typename decltype(pt)::type;
            return Scalar{NumericLimits<T>::LowestOrInf()};
        });

        // Convert to colum representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        col_ = internal::Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all, min);
        axes_.resize(kernel_size.size());
        std::iota(axes_.begin(), axes_.end(), 2);
        return col_.Max(axes_);
    }

    // TODO(hvy): Implement me.
    Array Backward(
            const Array& /*x*/,
            const StackVector<int64_t, kMaxNdim>& /*kernel_size*/,
            const StackVector<int64_t, kMaxNdim>& /*stride*/,
            const StackVector<int64_t, kMaxNdim>& /*pad*/,
            bool /*cover_all*/,
            const Array& /*gout*/) override {
        return Array{};
    }

private:
    // Cached in Forward and reused in Backward to compute indices.
    Array col_{};
    Axes axes_{};
};

}  // namespace

std::unique_ptr<MaxPoolForwardBackward> NativeDevice::GetMaxPoolForwardBackward() {
    return std::make_unique<NativeMaxPoolForwardBackward>();
}

}  // namespace native
}  // namespace xchainer
