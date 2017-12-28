#include "xchainer/dtype.h"

#include <ostream>
#include <sstream>
#include <unordered_map>

namespace xchainer {

std::ostream& operator<<(std::ostream& os, Dtype dtype) { return os << GetDtypeName(dtype); }

Dtype GetDtype(const std::string& name) {
    static const std::unordered_map<std::string, Dtype> kMapping = {
        // full name
        {"bool", Dtype::kBool},
        {"int8", Dtype::kInt8},
        {"int16", Dtype::kInt16},
        {"int32", Dtype::kInt32},
        {"int64", Dtype::kInt64},
        {"uint8", Dtype::kUInt8},
        {"float32", Dtype::kFloat32},
        {"float64", Dtype::kFloat64},
        // character code
        {"?", Dtype::kBool},
        {"b", Dtype::kInt8},
        {"h", Dtype::kInt16},
        {"i", Dtype::kInt32},
        {"q", Dtype::kInt64},
        {"B", Dtype::kUInt8},
        {"f", Dtype::kFloat32},
        {"d", Dtype::kFloat64},
    };

    auto it = kMapping.find(name);
    if (it == kMapping.end()) {
        throw DtypeError("unknown dtype name: \"" + name + '"');
    }
    return it->second;
}

std::vector<Dtype> GetAllDtypes() {
    return {
        Dtype::kBool, Dtype::kInt8, Dtype::kInt16, Dtype::kInt32, Dtype::kInt64, Dtype::kUInt8, Dtype::kFloat32, Dtype::kFloat64,
    };
}

void CheckEqual(Dtype lhs, Dtype rhs) {
    if (lhs != rhs) {
        std::ostringstream ss;
        ss << "dtype mismatched: " << lhs << " != " << rhs;
        throw DtypeError(ss.str());
    }
}

}  // namespace xchainer
