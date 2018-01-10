#pragma once

#include "xchainer/array.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace testing {

bool AllClose(const Array& a, const Array& b, double rtol, double atol);

}  // namespace testing
}  // namespace xchainer
