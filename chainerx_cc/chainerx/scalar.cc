#include "chainerx/scalar.h"

#include <sstream>
#include <string>

#include "chainerx/macro.h"

namespace chainerx {

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
        case Dtype::kFloat16:
        case Dtype::kFloat32:
        case Dtype::kFloat64:
            os << static_cast<double>(value);
            break;
        default:
            CHAINERX_NEVER_REACH();
    }
    return os;
}

}  // namespace chainerx
