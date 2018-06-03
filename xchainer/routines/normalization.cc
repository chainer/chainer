#include "xchainer/routines/normalization.h"

#include <memory>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {

Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const OptionalAxes& axis) {
    // TODO(hvy): Check that running_mean, running_var is contiguous.
    if (GetKind(eps.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"BatchNormalization expects eps of floating point kind but found ", eps.dtype(), "."};
    }
    if (GetKind(decay.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"BatchNormalization expects decay of floating point kind but found ", decay.dtype(), "."};
    }

    Dtype dtype = x.dtype();
    CheckEqual(dtype, gamma.dtype());
    CheckEqual(dtype, beta.dtype());
    CheckEqual(dtype, running_mean.dtype());
    CheckEqual(dtype, running_var.dtype());
    CheckEqual(dtype, eps.dtype());
    CheckEqual(dtype, decay.dtype());

    Shape reduced = gamma.shape();
    CheckEqual(reduced, beta.shape());
    CheckEqual(reduced, running_mean.shape());
    CheckEqual(reduced, running_var.shape());

    Axes sorted_axis = axis.has_value() ? internal::GetSortedAxes(*axis, x.ndim()) : Axes{0};
    CheckEqual(reduced, internal::ReduceShape(x.shape(), sorted_axis, true));

    // TODO(hvy): Implement backward.
    std::unique_ptr<BatchNormForwardBackward> fb = x.device().GetBatchNormForwardBackward();
    return fb->Forward(
            x, gamma, beta, running_mean, running_var, eps, decay, axis.has_value() ? internal::GetSortedAxes(*axis, x.ndim()) : Axes{0});
}

}  // namespace xchainer
