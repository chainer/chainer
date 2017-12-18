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
Array& Array::imul(const Array& other) {
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(data_.get());
    T* rdata = (T*)(other.data().get());
    for (i = 0; i < total_size; i++) {
        ldata[i] *= rdata[i];
    }
    return *this;
}

Array& Array::imul(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            throw DtypeError("bool cannot be multiplied");
        case Dtype::kInt8:
            return imul<int8_t>(other);
        case Dtype::kInt16:
            return imul<int16_t>(other);
        case Dtype::kInt32:
            return imul<int32_t>(other);
        case Dtype::kInt64:
            return imul<int64_t>(other);
        case Dtype::kUInt8:
            return imul<uint8_t>(other);
        case Dtype::kFloat32:
            return imul<float>(other);
        case Dtype::kFloat64:
            return imul<double>(other);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
Array Array::add(const Array& other) {
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(data_.get());
    T* rdata = (T*)(other.data().get());
    T* odata = new T[total_size];
    for (i = 0; i < total_size; i++) {
        odata[i] = ldata[i] + rdata[i];
    }
    Array out = {shape_, dtype_, std::unique_ptr<T[]>(odata)};
    return out;
}

Array Array::add(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            throw DtypeError("bool cannot be added");
        case Dtype::kInt8:
            return add<int8_t>(other);
        case Dtype::kInt16:
            return add<int16_t>(other);
        case Dtype::kInt32:
            return add<int32_t>(other);
        case Dtype::kInt64:
            return add<int64_t>(other);
        case Dtype::kUInt8:
            return add<uint8_t>(other);
        case Dtype::kFloat32:
            return add<float>(other);
        case Dtype::kFloat64:
            return add<double>(other);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
Array Array::mul(const Array& other) {
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(data_.get());
    T* rdata = (T*)(other.data().get());
    T* odata = new T[total_size];
    for (i = 0; i < total_size; i++) {
        odata[i] = ldata[i] * rdata[i];
    }
    Array out = {shape_, dtype_, std::unique_ptr<T[]>(odata)};
    return out;
}

Array Array::mul(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            throw DtypeError("bool cannot be multiplied");
        case Dtype::kInt8:
            return mul<int8_t>(other);
        case Dtype::kInt16:
            return mul<int16_t>(other);
        case Dtype::kInt32:
            return mul<int32_t>(other);
        case Dtype::kInt64:
            return mul<int64_t>(other);
        case Dtype::kUInt8:
            return mul<uint8_t>(other);
        case Dtype::kFloat32:
            return mul<float>(other);
        case Dtype::kFloat64:
            return mul<double>(other);
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
