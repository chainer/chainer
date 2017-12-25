#include "xchainer/array.h"

#include <cassert>
#include <cstring>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

#include "xchainer/array_math.h"
#include "xchainer/array_repr.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/array_math.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"

namespace xchainer {

namespace {

std::shared_ptr<void> AllocateCudaManaged(const void* src_ptr, size_t size) {
#ifdef XCHAINER_ENABLE_CUDA
    void* ptr = nullptr;
    cuda::CheckError(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    std::memcpy(ptr, src_ptr, size);
    return std::shared_ptr<void>(ptr, ::cudaFree);
#else
    (void)src_ptr;
    (void)size;
    assert(0);
    return std::shared_ptr<void>();
#endif  // XCHAINER_ENABLE_CUDA
}

}  // namespace

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, int64_t offset)
    : shape_(shape), is_contiguous_(true), dtype_(dtype), data_(nullptr), offset_(offset) {
    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        data_ = std::move(data);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        size_t size = static_cast<size_t>(shape_.total_size() * GetElementSize(dtype));
        data_ = AllocateCudaManaged(data.get(), size);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

Array& Array::IAdd(const Array& rhs) {
    Add(rhs, *this);
    return *this;
}

Array& Array::IMul(const Array& rhs) {
    Mul(rhs, *this);
    return *this;
}

Array Array::Add(const Array& rhs) const {
    Array out = {shape_, dtype_, std::make_unique<uint8_t[]>(total_bytes())};
    Add(rhs, out);
    return out;
}

Array Array::Mul(const Array& rhs) const {
    Array out = {shape_, dtype_, std::make_unique<uint8_t[]>(total_bytes())};
    Mul(rhs, out);
    return out;
}

void Array::Add(const Array& rhs, Array& out) const {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Add(*this, rhs, out);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Add(*this, rhs, out);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

void Array::Mul(const Array& rhs, Array& out) const {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Mul(*this, rhs, out);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Mul(*this, rhs, out);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

std::string Array::ToString() const { return ArrayRepr(*this); }

}  // namespace xchainer
