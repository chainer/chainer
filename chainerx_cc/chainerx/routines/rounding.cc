#include "chainerx/routines/rounding.h"

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/rounding.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

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

}  // namespace chainerx
