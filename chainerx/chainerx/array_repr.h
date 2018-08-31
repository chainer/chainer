#pragma once

#include <iostream>
#include <string>

#include "chainerx/array.h"

namespace chainerx {

class Array;

std::string ArrayRepr(const Array& array);

std::ostream& operator<<(std::ostream& os, const Array& array);

}  // namespace chainerx
