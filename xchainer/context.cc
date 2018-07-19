#include "xchainer/context.h"

#include <dlfcn.h>

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <stack>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native/native_backend.h"

namespace xchainer {
namespace {

std::atomic<Context*> g_global_default_context{nullptr};
thread_local Context* t_default_context{nullptr};

std::string GetXchainerPath() {
    char* xchainer_path = std::getenv("XCHAINER_PATH");
    if (xchainer_path != nullptr) {
        return xchainer_path;
    }

    char* home_path = std::getenv("HOME");
    if (home_path == nullptr) {
        throw XchainerError{"Xchainer path is not defined. Set either XCHAINER_PATH or HOME."};
    }
    return std::string(home_path) + "/.xchainer";
}

}  // namespace

Context::~Context() {
    // Need to call dtor of all backends before closing shared objects
    backends_.clear();
    for (void* handle : dlopen_handles_) {
        ::dlclose(handle);
    }
}

native::NativeBackend& Context::GetNativeBackend() {
    Backend& backend = GetBackend(native::NativeBackend::kDefaultName);
    return static_cast<native::NativeBackend&>(backend);  // NOLINT
}

Backend& Context::GetBackend(const std::string& backend_name) {
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto it = backends_.find(backend_name);
        if (it != backends_.end()) {
            return *it->second;
        }
    }

    // Ctor of each backend may call member functions of Context.
    // Lock is released here to avoid any deadlocks.
    std::unique_ptr<Backend> backend;
    if (backend_name == native::NativeBackend::kDefaultName) {
        backend = std::make_unique<native::NativeBackend>(*this);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (backend_name == cuda::CudaBackend::kDefaultName) {
        backend = std::make_unique<cuda::CudaBackend>(*this);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        // Load .so file
        std::string so_file_path = GetXchainerPath() + "/backends/" + backend_name + ".so";
        void* handle = ::dlopen(so_file_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            throw BackendError{"Backend not found: '", backend_name, "'"};
        }
        {
            std::lock_guard<std::mutex> lock{mutex_};
            dlopen_handles_.push_back(handle);
        }

        // Create backend
        void* ptr = ::dlsym(handle, "CreateBackend");
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto create_backend = reinterpret_cast<std::unique_ptr<Backend> (*)(Context&)>(ptr);
        if (create_backend == nullptr) {
            throw BackendError{"Invalid backend plugin: CreateBackend is not found in '", so_file_path, "'."};
        }
        backend = create_backend(*this);
    }

    {
        // In a multi-threaded case, backends_[backend_name] may already exist at this point.
        // In that case, the backend created above is thrown away.
        std::lock_guard<std::mutex> lock{mutex_};
        auto pair = backends_.emplace(backend_name, std::move(backend));
        return *pair.first->second;
    }
}

Device& Context::GetDevice(const DeviceId& device_id) {
    Backend& backend = GetBackend(device_id.backend_name());
    return backend.GetDevice(device_id.index());
}

// TODO(sonots): Create a map to get graph name from sub id
GraphId Context::MakeNextGraphId(std::string graph_name) {
    (void)graph_name;  // unused
    graph_sub_ids_.emplace(next_graph_sub_id_);
    return GraphId{*this, next_graph_sub_id_++};
}

void Context::ReleaseGraphId(const GraphId& graph_id) {
    // Pop all graphs from the stack that were created after the given graph and then the given graph itself.
    while (!graph_sub_ids_.empty() && graph_id.sub_id() >= graph_sub_ids_.top()) {
        graph_sub_ids_.pop();
    }
    if (graph_sub_ids_.empty()) {
        outermost_graph_id_.reset();
    }
}

std::vector<GraphId> Context::InnerGraphIds(GraphId graph_id) {
    // TODO(hvy): A stack is not appropriate if it is only popped during graph scope exit.
    std::vector<GraphId> inner_graph_ids;
    while (!graph_sub_ids_.empty() && graph_id.sub_id() < graph_sub_ids_.top()) {
        inner_graph_ids.emplace_back(*this, graph_sub_ids_.top());
        graph_sub_ids_.pop();
    }
    for (auto it = inner_graph_ids.rbegin(); it != inner_graph_ids.rend(); ++it) {
        graph_sub_ids_.emplace(it->sub_id());
    }
    return inner_graph_ids;
}

Context& GetGlobalDefaultContext() {
    Context* context = g_global_default_context;
    if (context == nullptr) {
        throw ContextError{"Global default context is not set."};
    }
    return *context;
}

void SetGlobalDefaultContext(Context* context) { g_global_default_context = context; }

namespace internal {

Context* GetDefaultContextNoExcept() noexcept { return t_default_context; }

}  // namespace internal

Context& GetDefaultContext() {
    if (t_default_context == nullptr) {
        return GetGlobalDefaultContext();
    }
    return *t_default_context;
}

void SetDefaultContext(Context* context) {
    if (t_default_context != context) {
        t_default_context = context;
        SetDefaultDevice(nullptr);
    }
}

}  // namespace xchainer
