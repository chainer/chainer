#pragma once

#include <iostream>
#include <string>

#include "xchainer/array.h"

namespace xchainer {

class Array;

std::string ArrayRepr(const Array& array);

std::ostream& operator<<(std::ostream& os, const Array& array);

}  // namespace xchainer
