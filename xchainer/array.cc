#include "xchainer/array.h"

#include <cassert>

namespace xchainer {

template <typename T>
Array& Array::Add(const Array& rhs, Array& out) {
    const Array& lhs = *this;
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(lhs.data().get());
    T* rdata = (T*)(rhs.data().get());
    T* odata = (T*)(out.data().get());
    for (i = 0; i < total_size; i++) {
        odata[i] = ldata[i] + rdata[i];
    }
    return out;
}

Array& Array::IAdd(const Array& rhs) {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    Array& out = *this;
    switch (dtype_) {
        case Dtype::kBool:
            return Add<bool>(rhs, out);
        case Dtype::kInt8:
            return Add<int8_t>(rhs, out);
        case Dtype::kInt16:
            return Add<int16_t>(rhs, out);
        case Dtype::kInt32:
            return Add<int32_t>(rhs, out);
        case Dtype::kInt64:
            return Add<int64_t>(rhs, out);
        case Dtype::kUInt8:
            return Add<uint8_t>(rhs, out);
        case Dtype::kFloat32:
            return Add<float>(rhs, out);
        case Dtype::kFloat64:
            return Add<double>(rhs, out);
        default:
            assert(0);  // should never be reached
    }
}

Array Array::Add(const Array& rhs) {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    Array out = {shape_, dtype_, std::shared_ptr<void>(new char[total_bytes()])};
    switch (dtype_) {
        case Dtype::kBool:
            return Add<bool>(rhs, out);
        case Dtype::kInt8:
            return Add<int8_t>(rhs, out);
        case Dtype::kInt16:
            return Add<int16_t>(rhs, out);
        case Dtype::kInt32:
            return Add<int32_t>(rhs, out);
        case Dtype::kInt64:
            return Add<int64_t>(rhs, out);
        case Dtype::kUInt8:
            return Add<uint8_t>(rhs, out);
        case Dtype::kFloat32:
            return Add<float>(rhs, out);
        case Dtype::kFloat64:
            return Add<double>(rhs, out);
        default:
            assert(0);  // should never be reached
    }
}

template <typename T>
Array& Array::Mul(const Array& rhs, Array& out) {
    const Array& lhs = *this;
    auto total_size = shape_.total_size();
    decltype(total_size) i = 0;
    T* ldata = (T*)(lhs.data().get());
    T* rdata = (T*)(rhs.data().get());
    T* odata = (T*)(out.data().get());
    for (i = 0; i < total_size; i++) {
        odata[i] = ldata[i] * rdata[i];
    }
    return out;
}

Array& Array::IMul(const Array& rhs) {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    Array& out = *this;
    switch (dtype_) {
        case Dtype::kBool:
            return Mul<bool>(rhs, out);
        case Dtype::kInt8:
            return Mul<int8_t>(rhs, out);
        case Dtype::kInt16:
            return Mul<int16_t>(rhs, out);
        case Dtype::kInt32:
            return Mul<int32_t>(rhs, out);
        case Dtype::kInt64:
            return Mul<int64_t>(rhs, out);
        case Dtype::kUInt8:
            return Mul<uint8_t>(rhs, out);
        case Dtype::kFloat32:
            return Mul<float>(rhs, out);
        case Dtype::kFloat64:
            return Mul<double>(rhs, out);
        default:
            assert(0);  // should never be reached
    }
}

Array Array::Mul(const Array& rhs) {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    Array out = {shape_, dtype_, std::shared_ptr<void>(new char[total_bytes()])};
    switch (dtype_) {
        case Dtype::kBool:
            return Mul<bool>(rhs, out);
        case Dtype::kInt8:
            return Mul<int8_t>(rhs, out);
        case Dtype::kInt16:
            return Mul<int16_t>(rhs, out);
        case Dtype::kInt32:
            return Mul<int32_t>(rhs, out);
        case Dtype::kInt64:
            return Mul<int64_t>(rhs, out);
        case Dtype::kUInt8:
            return Mul<uint8_t>(rhs, out);
        case Dtype::kFloat32:
            return Mul<float>(rhs, out);
        case Dtype::kFloat64:
            return Mul<double>(rhs, out);
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
