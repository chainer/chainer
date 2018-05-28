#include "xchainer/routines/normalization.h"

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"
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
    Dtype dtype = x.dtype();
    CheckEqual(dtype, gamma.dtype());
    CheckEqual(dtype, beta.dtype());
    CheckEqual(dtype, running_mean.dtype());
    CheckEqual(dtype, running_var.dtype());

    Shape reduced = gamma.shape();
    CheckEqual(reduced, beta.shape());
    CheckEqual(reduced, running_mean.shape());
    CheckEqual(reduced, running_var.shape());

    Array out = EmptyLike(x, x.device());
    x.device().BatchNormalization(
            x,
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
            decay,
            axis.has_value() ? internal::GetSortedAxes(*axis, x.ndim()) : Axes{0},
            out);
    // TODO(hvy): Implement backward.
    return out;
}

}  // namespace xchainer
