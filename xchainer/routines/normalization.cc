#include "xchainer/routines/normalization.h"

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/constant.h"
#include "xchainer/routines/creation.h"
#include "xchainer/stack_vector.h"

namespace xchainer {

Array BatchNormalization(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        float eps,
        float decay,
        const OptionalAxes& axis) {
    Array out = EmptyLike(x, x.device());

    // TODO(hvy): Check input dtypes.

    if (!axis.has_value()) {
        // TODO(hvy): Infer axes.
        throw NotImplementedError{};
    }
    Axes sorted_axis = internal::GetSortedAxes(*axis, x.ndim());

    // Running mean and running var are updated inside the device call.
    x.device().BatchNormalization(x, gamma, beta, running_mean, running_var, eps, decay, sorted_axis, out);
    return out;
}

}  // namespace xchainer
