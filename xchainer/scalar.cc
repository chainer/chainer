#include "xchainer/scalar.h"

#include <cassert>
#include <sstream>

#include "xchainer/error.h"

namespace xchainer {

std::string Scalar::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

Scalar operator-(Scalar value) {
    switch (value.dtype()) {
        case Dtype::kBool:
            throw DtypeError("bool scalar cannot be negated");

        case Dtype::kInt8:
            return -value.Cast<int8_t>();
        case Dtype::kInt16:
            return -value.Cast<int16_t>();
        case Dtype::kInt32:
            return -value.Cast<int32_t>();
        case Dtype::kInt64:
            return -value.Cast<int64_t>();
        case Dtype::kUInt8:
            return -value.Cast<uint8_t>();
        case Dtype::kFloat32:
            return -value.Cast<float>();
        case Dtype::kFloat64:
            return -value.Cast<double>();
        default:  // never reach
            assert(0);
    }
    return 0;
}

std::ostream& operator<<(std::ostream& os, Scalar value) {
    switch (value.dtype()) {
        case Dtype::kBool:
            os << (value.Cast<bool>() ? "True" : "False");
            break;
        case Dtype::kInt8:
        case Dtype::kInt16:
        case Dtype::kInt32:
        case Dtype::kInt64:
            os << value.Cast<int64_t>();
            break;
        case Dtype::kUInt8:
            os << value.Cast<uint8_t>();
            break;
        case Dtype::kFloat32:
        case Dtype::kFloat64:
            os << value.Cast<double>();
            break;
        default:  // never reach
            assert(0);
    }
    return os;
}

}  // namespace xchainer
