#pragma once

#include "xchainer/array.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace testing {

bool AllClose(const Array& a, const Array& b, const Scalar& atol, const Scalar& rtol);

}  // namespace testing
}  // namespace xchainer
