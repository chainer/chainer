#pragma once

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/constant.h"
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
        const OptionalAxes& axes);

}  // namespace xchainer
