#include "xchainer/native_device.h"

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

namespace xchainer {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) { return std::make_unique<uint8_t[]>(bytesize); }

void NativeDevice::MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) {
    (void)dst_ptr;   // unused
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
}

std::shared_ptr<void> NativeDevice::FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)bytesize;  // unused
    return src_ptr;
}

void NativeDevice::Fill(Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        T c_value{value};

        int64_t size = out.GetTotalSize();
        auto* ptr = static_cast<T*>(out.data().get());
        for (int64_t i = 0; i < size; ++i) {
            ptr[i] = c_value;
        }
    });
}

void NativeDevice::Add(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void NativeDevice::Mul(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void NativeDevice::Synchronize() {}

}  // namespace xchainer
