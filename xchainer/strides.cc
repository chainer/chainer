#include "xchainer/strides.h"

#include <algorithm>
#include <iterator>
#include <ostream>
#include <sstream>

#include <gsl/gsl>

#include "xchainer/error.h"
#include "xchainer/shape.h"

namespace xchainer {

Strides::Strides(const Shape& shape, int64_t element_size) : ndim_(shape.ndim()) {
    int64_t stride = element_size;
    for (int i = ndim_ - 1; i >= 0; --i) {
        gsl::at(dims_, i) = stride;
        stride *= shape[i];
    }
}

std::string Strides::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Strides& strides) {
    os << "(";
    for (auto iter = strides.begin(); iter != strides.end(); ++iter) {
        if (iter != strides.begin()) {
            os << ", ";
        }
        os << *iter;
    }
    // same as Python tuples with trailing comma in case of length 1
    return os << (strides.ndim() == 1 ? ",)" : ")");
}

void CheckEqual(const Strides& lhs, const Strides& rhs) {
    if (lhs != rhs) {
        throw DimensionError("strides mismatched");
    }
}

}  // namespace xchainer
