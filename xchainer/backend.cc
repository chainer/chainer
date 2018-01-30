#include "xchainer/backend.h"
#include "xchainer/error.h"

namespace xchainer {

thread_local Backend* thread_local_backend = nullptr;
static_assert(std::is_pod<decltype(thread_local_backend)>::value, "thread_local_backend must be POD");

Backend* GetCurrentBackend() {
    Backend* backend = thread_local_backend;
    if (backend != nullptr) {
        return backend;
    } else {
        throw XchainerError("No backend is available. Please set via SetCurrentBackend()");
    }
}

void SetCurrentBackend(Backend* backend) noexcept { thread_local_backend = backend; }

}  // namespace xchainer
