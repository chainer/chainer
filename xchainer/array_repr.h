#pragma once

#include "xchainer/array_repr.h"

#include <iostream>
#include <string>

#include "xchainer/array.h"

namespace xchainer {

class Array;

std::string ArrayRepr(const Array& array);
void ArrayRepr(std::ostream& os, const Array& array);

}  // namespace xchainer
