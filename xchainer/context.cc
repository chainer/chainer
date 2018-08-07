#include "xchainer/context.h"

#include <dlfcn.h>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <mutex>
#include <string>
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

// TODO(sonots): Create a map to get backprop name from ordinal id
BackpropId Context::MakeNextBackpropId(std::string backprop_name) {
    backprop_stack_.emplace_back(next_backprop_ordinal_, std::move(backprop_name));
    return BackpropId{*this, next_backprop_ordinal_++};
}

void Context::ReleaseBackpropId(const BackpropId& backprop_id) {
    // Backprop IDs must be released in the reverse order of creation
    assert(&backprop_id.context() == this && backprop_id.ordinal() == backprop_stack_.back().ordinal);
    (void)backprop_id;  // unused

    backprop_stack_.pop_back();
}

std::string Context::GetBackpropName(const BackpropId& backprop_id) {
    // Note: backprop name cannot be returned by reference, as the reference may be invalidated when a new graph is pushed to the backprop
    // stack.

    static constexpr const char* kDefaultBackpropDisplayName = "<default>";

    if (backprop_id == default_backprop_id()) {
        return kDefaultBackpropDisplayName;
    }

    auto it = std::find_if(backprop_stack_.rbegin(), backprop_stack_.rend(), [&backprop_id](const BackpropStackItem& item) {
        return item.ordinal == backprop_id.ordinal();
    });
    if (it == backprop_stack_.rend()) {
        throw XchainerError{"Backprop not found in the context. Ordinal:", backprop_id.ordinal()};
    }
    return it->name;
}

void Context::CheckBackpropAllowed(const BackpropId& backprop_id) {
    // TODO(hvy): Check that backprop_id exists in the stack or that it is the default backprop id.
    for (auto it = backprop_stack_.rbegin(); it != backprop_stack_.rend(); ++it) {
        if (it->ordinal == backprop_id.ordinal()) {
            if (it->is_outer_graph_backpropped) {
                throw XchainerError{"Cannot backward for backprop ID ", backprop_id, " after outer backprop ID"};
            }
            break;
        }
    }
}

void Context::SetBackpropDone(const BackpropId& backprop_id) {
    for (auto it = backprop_stack_.rbegin(); it != backprop_stack_.rend(); ++it) {
        if (it->ordinal == backprop_id.ordinal()) {
            break;
        }
        it->is_outer_graph_backpropped = true;
    }
}

std::vector<BackpropId> Context::GetInnerBackpropIds(const BackpropId& backprop_id) {
    std::vector<BackpropId> inner_backprop_ids;
    inner_backprop_ids.reserve(backprop_stack_.size());
    for (auto it = backprop_stack_.rbegin(); it != backprop_stack_.rend() && it->ordinal > backprop_id.ordinal(); ++it) {
        inner_backprop_ids.emplace_back(BackpropId{*this, it->ordinal});
    }
    return inner_backprop_ids;
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
