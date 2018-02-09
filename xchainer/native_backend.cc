#include "xchainer/native_backend.h"

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

namespace xchainer {

std::shared_ptr<void> NativeBackend::Allocate(const Device& device, size_t bytesize) {
    (void)device;    // unused
    (void)bytesize;  // unused
    return nullptr;
}

void NativeBackend::MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) {
    (void)dst_ptr;   // unused
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
}

std::shared_ptr<void> NativeBackend::FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)device;    // unused
    (void)src_ptr;   // unused
    (void)bytesize;  // unused
    return nullptr;
}

void NativeBackend::Fill(Array& out, Scalar value) {
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

void NativeBackend::Add(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void NativeBackend::Mul(const Array& lhs, const Array& rhs, Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
}

void NativeBackend::Synchronize() {}

}  // namespace xchainer
