#pragma once

#include "xchainer/array.h"

namespace xchainer {
namespace routines {

Array& IAdd(Array& lhs, const Array& rhs);
Array& IMul(Array& lhs, const Array& rhs);

Array Add(const Array& lhs, const Array& rhs);
Array Mul(const Array& lhs, const Array& rhs);

}  // namespace routines
}  // namespace xchainer
