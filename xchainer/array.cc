#include "xchainer/array.h"

#include <cassert>

namespace xchainer {

template <typename T>
Array& Array::IAdd(const Array& other) {
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(data_.get());
    T* rdata = (T*)(other.data().get());
    for (i = 0; i < total_size; i++) {
        ldata[i] += rdata[i];
    }
    return *this;
}

Array& Array::IAdd(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            return IAdd<bool>(other);
        case Dtype::kInt8:
            return IAdd<int8_t>(other);
        case Dtype::kInt16:
            return IAdd<int16_t>(other);
        case Dtype::kInt32:
            return IAdd<int32_t>(other);
        case Dtype::kInt64:
            return IAdd<int64_t>(other);
        case Dtype::kUInt8:
            return IAdd<uint8_t>(other);
        case Dtype::kFloat32:
            return IAdd<float>(other);
        case Dtype::kFloat64:
            return IAdd<double>(other);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
Array& Array::IMul(const Array& other) {
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(data_.get());
    T* rdata = (T*)(other.data().get());
    for (i = 0; i < total_size; i++) {
        ldata[i] *= rdata[i];
    }
    return *this;
}

Array& Array::IMul(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            return IMul<bool>(other);
        case Dtype::kInt8:
            return IMul<int8_t>(other);
        case Dtype::kInt16:
            return IMul<int16_t>(other);
        case Dtype::kInt32:
            return IMul<int32_t>(other);
        case Dtype::kInt64:
            return IMul<int64_t>(other);
        case Dtype::kUInt8:
            return IMul<uint8_t>(other);
        case Dtype::kFloat32:
            return IMul<float>(other);
        case Dtype::kFloat64:
            return IMul<double>(other);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
Array Array::Add(const Array& other) {
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

Array Array::Add(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            return Add<bool>(other);
        case Dtype::kInt8:
            return Add<int8_t>(other);
        case Dtype::kInt16:
            return Add<int16_t>(other);
        case Dtype::kInt32:
            return Add<int32_t>(other);
        case Dtype::kInt64:
            return Add<int64_t>(other);
        case Dtype::kUInt8:
            return Add<uint8_t>(other);
        case Dtype::kFloat32:
            return Add<float>(other);
        case Dtype::kFloat64:
            return Add<double>(other);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
Array Array::Mul(const Array& other) {
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

Array Array::Mul(const Array& other) {
    // TODO: dtype conversion
    CheckEqual(dtype_, other.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, other.shape());
    switch (dtype_) {
        case Dtype::kBool:
            return Mul<bool>(other);
        case Dtype::kInt8:
            return Mul<int8_t>(other);
        case Dtype::kInt16:
            return Mul<int16_t>(other);
        case Dtype::kInt32:
            return Mul<int32_t>(other);
        case Dtype::kInt64:
            return Mul<int64_t>(other);
        case Dtype::kUInt8:
            return Mul<uint8_t>(other);
        case Dtype::kFloat32:
            return Mul<float>(other);
        case Dtype::kFloat64:
            return Mul<double>(other);
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
