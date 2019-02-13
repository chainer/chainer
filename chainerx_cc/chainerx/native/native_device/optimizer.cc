#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/numeric.h"

namespace chainerx {
namespace native {

void NativeDevice::Adam(
        const Array& grad,
        Scalar alpha,
        Scalar beta1,
        Scalar beta2,
        Scalar eps,
        Scalar eta,
        Scalar weight_decay_rate,
        const Array& param,
        const Array& m,
        const Array& t) {
    std::cout << grad << alpha << beta1 << beta2 << eps << eta << weight_decay_rate << param << m << t << std::endl;
}

}  // namespace native
}  // namespace chainerx
