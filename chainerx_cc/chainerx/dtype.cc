#include "chainerx/dtype.h"

#include <cstring>
#include <ostream>
#include <sstream>

namespace chainerx {

std::ostream& operator<<(std::ostream& os, Dtype dtype) { return os << GetDtypeName(dtype); }

Dtype GetDtype(const std::string& name) {
    // We define an ad-hoc POD struct to comply with the coding guideline.
    // Note that std::tuple is not a POD type.
    struct Pair {
        const char* name;
        Dtype dtype;
    };

    static const Pair kMapping[] = {
            // full name
            {"bool", Dtype::kBool},
            {"int8", Dtype::kInt8},
            {"int16", Dtype::kInt16},
            {"int32", Dtype::kInt32},
            {"int64", Dtype::kInt64},
            {"uint8", Dtype::kUInt8},
            {"float16", Dtype::kFloat16},
            {"float32", Dtype::kFloat32},
            {"float64", Dtype::kFloat64},
            // character code
            {"?", Dtype::kBool},
            {"b", Dtype::kInt8},
            {"h", Dtype::kInt16},
            {"i", Dtype::kInt32},
            {"l", Dtype::kInt64},
            {"B", Dtype::kUInt8},
            {"e", Dtype::kFloat16},
            {"f", Dtype::kFloat32},
            {"d", Dtype::kFloat64},
    };
    static_assert(std::is_pod<decltype(kMapping)>::value, "static variable must be POD to comply with the coding guideline");

    const char* cname = name.c_str();
    for (const Pair& pair : kMapping) {
        if (0 == std::strcmp(pair.name, cname)) {
            return pair.dtype;
        }
    }
    throw DtypeError{"unknown dtype name: \"", name, '"'};
}

std::vector<Dtype> GetAllDtypes() {
    return {
            Dtype::kBool,
            Dtype::kInt8,
            Dtype::kInt16,
            Dtype::kInt32,
            Dtype::kInt64,
            Dtype::kUInt8,
            Dtype::kFloat16,
            Dtype::kFloat32,
            Dtype::kFloat64,
    };
}

void CheckEqual(Dtype lhs, Dtype rhs) {
    if (lhs != rhs) {
        throw DtypeError{"dtype mismatched: ", lhs, " != ", rhs};
    }
}

}  // namespace chainerx
