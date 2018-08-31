#pragma once

#include <functional>

namespace chainerx {

class Array;
using ArrayRef = std::reference_wrapper<Array>;
using ConstArrayRef = std::reference_wrapper<const Array>;

}  // namespace chainerx
