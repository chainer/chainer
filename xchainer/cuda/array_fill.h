#pragma once

#include "xchainer/array.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace cuda {

void Fill(Array& out, Scalar value);

}  // namespace cuda
}  // namespace xchainer
