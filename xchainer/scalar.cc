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
            os << (value.UnwrapAndCast<bool>() ? "True" : "False");
            break;
        case Dtype::kInt8:
        case Dtype::kInt16:
        case Dtype::kInt32:
        case Dtype::kInt64:
        case Dtype::kUInt8:
            os << value.UnwrapAndCast<int64_t>();
            break;
        case Dtype::kFloat32:
        case Dtype::kFloat64:
            os << value.UnwrapAndCast<double>();
            break;
        default:  // never reach
            assert(0);
    }
    return os;
}

}  // namespace xchainer
