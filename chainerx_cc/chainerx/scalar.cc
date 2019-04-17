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
    switch (value.kind()) {
        case DtypeKind::kBool:
            os << (static_cast<bool>(value) ? "True" : "False");
            break;
        case DtypeKind::kInt:
            os << static_cast<int64_t>(value);
            break;
        case DtypeKind::kFloat:
            os << static_cast<double>(value);
            break;
        default:
            CHAINERX_NEVER_REACH();
    }
    return os;
}

}  // namespace chainerx
