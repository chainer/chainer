#include "xchainer/scalar.h"

#include <cassert>
#include <sstream>

namespace xchainer {

std::string Scalar::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, Scalar value) {
    switch (value.dtype()) {
        case Dtype::kBool:
            os << (static_cast<bool>(value) ? "True" : "False");
            break;
        case Dtype::kInt8:
        case Dtype::kInt16:
        case Dtype::kInt32:
        case Dtype::kInt64:
        case Dtype::kUInt8:
            os << static_cast<int64_t>(value);
            break;
        case Dtype::kFloat32:
        case Dtype::kFloat64:
            os << static_cast<double>(value);
            break;
        default:
            assert(false);  // should never be reached
    }
    return os;
}

}  // namespace xchainer
