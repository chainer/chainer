#include "xchainer/array_math.h"

#include <cassert>

#include "xchainer/array.h"

namespace xchainer {

namespace {

template <typename T>
void Add(const Array& lhs, const Array& rhs, Array& out) {
    auto total_size = lhs.shape().total_size();
    const T* ldata = static_cast<const T*>(lhs.data().get());
    const T* rdata = static_cast<const T*>(rhs.data().get());
    T* odata = static_cast<T*>(out.data().get());
    for (decltype(total_size) i = 0; i < total_size; i++) {
        odata[i] = ldata[i] + rdata[i];
    }
}

template <typename T>
void Mul(const Array& lhs, const Array& rhs, Array& out) {
    auto total_size = lhs.shape().total_size();
    const T* ldata = static_cast<const T*>(lhs.data().get());
    const T* rdata = static_cast<const T*>(rhs.data().get());
    T* odata = static_cast<T*>(out.data().get());
    for (decltype(total_size) i = 0; i < total_size; i++) {
        odata[i] = ldata[i] * rdata[i];
    }
}

}  // namespace

void Add(const Array& lhs, const Array& rhs, Array& out) {
    switch (lhs.dtype()) {
        case Dtype::kBool:
            Add<bool>(lhs, rhs, out);
            break;
        case Dtype::kInt8:
            Add<int8_t>(lhs, rhs, out);
            break;
        case Dtype::kInt16:
            Add<int16_t>(lhs, rhs, out);
            break;
        case Dtype::kInt32:
            Add<int32_t>(lhs, rhs, out);
            break;
        case Dtype::kInt64:
            Add<int64_t>(lhs, rhs, out);
            break;
        case Dtype::kUInt8:
            Add<uint8_t>(lhs, rhs, out);
            break;
        case Dtype::kFloat32:
            Add<float>(lhs, rhs, out);
            break;
        case Dtype::kFloat64:
            Add<double>(lhs, rhs, out);
            break;
        default:
            assert(0);  // should never be reached
    }
}

void Mul(const Array& lhs, const Array& rhs, Array& out) {
    switch (lhs.dtype()) {
        case Dtype::kBool:
            Mul<bool>(lhs, rhs, out);
            break;
        case Dtype::kInt8:
            Mul<int8_t>(lhs, rhs, out);
            break;
        case Dtype::kInt16:
            Mul<int16_t>(lhs, rhs, out);
            break;
        case Dtype::kInt32:
            Mul<int32_t>(lhs, rhs, out);
            break;
        case Dtype::kInt64:
            Mul<int64_t>(lhs, rhs, out);
            break;
        case Dtype::kUInt8:
            Mul<uint8_t>(lhs, rhs, out);
            break;
        case Dtype::kFloat32:
            Mul<float>(lhs, rhs, out);
            break;
        case Dtype::kFloat64:
            Mul<double>(lhs, rhs, out);
            break;
        default:
            assert(0);  // should never be reached
    }
}

}  // namespace xchainer
