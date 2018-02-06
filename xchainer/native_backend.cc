#include "xchainer/native_backend.h"

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
    (void)out;    // unused
    (void)value;  // unused
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
