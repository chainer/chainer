#include "chainerx/routines/math.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/math.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {

Array Sigmoid(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Reciprocal(1 + Exp(-x_cast));
}

Array Relu(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Maximum(0, x_cast);
}

Array Ceil(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<CeilKernel>(x, out);
    }
    return out;
}

Array Floor(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    Array out = Empty(x.shape(), dtype, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<FloorKernel>(x, out);
    }
    return out;
}

Array IsNan(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsNanKernel>(x, out);
    }
    return out;
}

Array IsInf(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsInfKernel>(x, out);
    }
    return out;
}

Array IsFinite(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsFiniteKernel>(x, out);
    }
    return out;
}

}  // namespace chainerx
