#include "chainerx/dtype.h"

#include <cstring>
#include <ostream>
#include <sstream>

namespace chainerx {

std::ostream& operator<<(std::ostream& os, Dtype dtype) { return os << GetDtypeName(dtype); }

// Returns the minimal dtype which can be safely casted from both dtypes.
Dtype PromoteTypes(Dtype dt1, Dtype dt2) {
    DtypeKind kind1 = GetKind(dt1);
    DtypeKind kind2 = GetKind(dt2);
    // Bools always have least priority
    if (kind1 == DtypeKind::kBool) {
        return dt2;
    }
    if (kind2 == DtypeKind::kBool) {
        return dt1;
    }
    // Same kinds -> return the wider one
    if (kind1 == kind2) {
        if (GetItemSize(dt1) >= GetItemSize(dt2)) {
            return dt1;
        }
        return dt2;
    }
    // Float takes priority over the other
    if (kind1 == DtypeKind::kFloat) {
        return dt1;
    }
    if (kind2 == DtypeKind::kFloat) {
        return dt2;
    }
    // Kinds are kInt and kUInt
    if (kind1 == DtypeKind::kUInt) {
        std::swap(dt1, dt2);
        std::swap(kind1, kind2);
    }
    CHAINERX_ASSERT(kind1 == DtypeKind::kInt && kind2 == DtypeKind::kUInt);
    if (GetItemSize(dt1) > GetItemSize(dt2)) {
        // Unsigned one has narrower width.
        // Return the signed dtype.
        return dt1;
    }
    // Otherwise return the signed dtype with one-level wider than the unsigned one.
    switch (dt2) {
        case Dtype::kUInt8:
            return Dtype::kInt16;
            // If there will be more unsigned int types, add here.
        default:
            CHAINERX_NEVER_REACH();
    }
}

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
