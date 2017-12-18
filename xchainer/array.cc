#include "xchainer/array.h"

#include <cassert>

namespace xchainer {

template <typename T>
Array& Array::iadd(const Array& other) {
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(data_.get());
    T* rdata = (T*)(other.data().get());
    for (i = 0; i < total_size; i++) {
        ldata[i] += rdata[i];
    }
    return *this;
}

Array& Array::iadd(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            throw DtypeError("bool cannot be added");
        case Dtype::kInt8:
            return iadd<int8_t>(other);
        case Dtype::kInt16:
            return iadd<int16_t>(other);
        case Dtype::kInt32:
            return iadd<int32_t>(other);
        case Dtype::kInt64:
            return iadd<int64_t>(other);
        case Dtype::kUInt8:
            return iadd<uint8_t>(other);
        case Dtype::kFloat32:
            return iadd<float>(other);
        case Dtype::kFloat64:
            return iadd<double>(other);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
void CheckEqual(const Array& lhs, const Array& rhs) {
    auto total_size = lhs.shape().total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(lhs.data().get());
    T* rdata = (T*)(rhs.data().get());
    for (i = 0; i < total_size; i++) {
        if (ldata[i] != rdata[i]) {
          // TODO: repl
          throw DataError("data mismatch");
        }
    }
}

void CheckEqual(const Array& lhs, const Array& rhs) {
    CheckEqual(lhs.dtype(), rhs.dtype());
    CheckEqual(lhs.shape(), rhs.shape());
    switch (lhs.dtype()) {
        case Dtype::kBool:
            return CheckEqual<bool>(lhs, rhs);
        case Dtype::kInt8:
            return CheckEqual<int8_t>(lhs, rhs);
        case Dtype::kInt16:
            return CheckEqual<int16_t>(lhs, rhs);
        case Dtype::kInt32:
            return CheckEqual<int32_t>(lhs, rhs);
        case Dtype::kInt64:
            return CheckEqual<int64_t>(lhs, rhs);
        case Dtype::kUInt8:
            return CheckEqual<uint8_t>(lhs, rhs);
        case Dtype::kFloat32:
            return CheckEqual<float>(lhs, rhs);
        case Dtype::kFloat64:
            return CheckEqual<double>(lhs, rhs);
        default:
            assert(0);  // should never be reached
    }
}


}  // namespace xchainer
