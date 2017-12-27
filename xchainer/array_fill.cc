#include "xchainer/array_fill.h"

#include <cassert>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

namespace xchainer {

namespace {

template <typename T>
void FillImpl(Array& array, T value) {
    int64_t size = array.total_size();
    T* ptr = static_cast<T*>(array.data().get());
    for (int64_t i = 0; i < size; ++i) {
        ptr[i] = value;
    }
}

}  // namespace

void Fill(Array& out, Scalar value) {
    switch (value.dtype()) {
        case Dtype::kBool:
            FillImpl(out, static_cast<bool>(value));
            break;
        case Dtype::kInt8:
            FillImpl(out, static_cast<int8_t>(value));
            break;
        case Dtype::kInt16:
            FillImpl(out, static_cast<int16_t>(value));
            break;
        case Dtype::kInt32:
            FillImpl(out, static_cast<int32_t>(value));
            break;
        case Dtype::kInt64:
            FillImpl(out, static_cast<int64_t>(value));
            break;
        case Dtype::kUInt8:
            FillImpl(out, static_cast<uint8_t>(value));
            break;
        case Dtype::kFloat32:
            FillImpl(out, static_cast<float>(value));
            break;
        case Dtype::kFloat64:
            FillImpl(out, static_cast<double>(value));
            break;
        default:
            assert(false);  // should never be reached
    }
}

}  // namespace xchainer
