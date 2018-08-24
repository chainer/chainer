#include "xchainer/context.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/macro.h"
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

Context::Context() {
    // Register the default backprop ID
    static constexpr const char* kDefaultBackpropName = "<default>";
    MakeBackpropId(kDefaultBackpropName);
}

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
    std::unique_ptr<Backend, context_detail::BackendDeleter> backend;
    if (backend_name == native::NativeBackend::kDefaultName) {
        backend = std::unique_ptr<Backend, context_detail::BackendDeleter>{
                new native::NativeBackend{*this}, context_detail::BackendDeleter{[](Backend* ptr) { delete ptr; }}};
#ifdef XCHAINER_ENABLE_CUDA
    } else if (backend_name == cuda::CudaBackend::kDefaultName) {
        backend = std::unique_ptr<Backend, context_detail::BackendDeleter>{
                new cuda::CudaBackend{*this}, context_detail::BackendDeleter{[](Backend* ptr) { delete ptr; }}};
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
        void* ptr_create_backend = ::dlsym(handle, "CreateBackend");
        void* ptr_destroy_backend = ::dlsym(handle, "DestroyBackend");
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto create_backend = reinterpret_cast<Backend* (*)(Context&)>(ptr_create_backend);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto destroy_backend = reinterpret_cast<void (*)(Backend*)>(ptr_destroy_backend);
        if (create_backend == nullptr) {
            throw BackendError{"Invalid backend plugin: CreateBackend is not found in '", so_file_path, "'."};
        }
        if (destroy_backend == nullptr) {
            throw BackendError{"Invalid backend plugin: DestroyBackend is not found in '", so_file_path, "'."};
        }
        backend = std::unique_ptr<Backend, context_detail::BackendDeleter>{create_backend(*this),
                                                                           context_detail::BackendDeleter{destroy_backend}};
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

BackpropId Context::MakeBackpropId(std::string backprop_name) {
    // Create new backprop ID
    backprop_set_.emplace_back(next_backprop_ordinal_, std::move(backprop_name));

    return BackpropId{*this, next_backprop_ordinal_++};
}

void Context::ReleaseBackpropId(const BackpropId& backprop_id) {
    CheckValidBackpropId(backprop_id);

    if (backprop_id.ordinal() == 0) {
        throw XchainerError{"The default backprop ID cannot be released."};
    }

    ReleaseBackpropIdNoExcept(backprop_id);
}

void Context::ReleaseBackpropIdNoExcept(const BackpropId& backprop_id) noexcept {
    BackpropSetItem* item = GetBackpropSetItem(backprop_id.ordinal());
    if (item == nullptr) {
        return;
    }

    // Remove the connection pairs involving the backprop ID
    backprop_connections_.erase(
            std::remove_if(
                    backprop_connections_.begin(),
                    backprop_connections_.end(),
                    [ordinal = backprop_id.ordinal()](const std::pair<BackpropOrdinal, BackpropOrdinal>& pair) {
                        return pair.first == ordinal || pair.second == ordinal;
                    }),
            backprop_connections_.end());

    // Remove the backprop ID.
    backprop_set_.erase(
            std::find_if(backprop_set_.begin(), backprop_set_.end(), [ordinal = backprop_id.ordinal()](const BackpropSetItem& item) {
                return item.ordinal == ordinal;
            }));
}

void Context::CheckValidBackpropId(const BackpropId& backprop_id) const {
    if (&backprop_id.context() != this) {
        throw XchainerError{"Invalid context in backprop ID: ", backprop_id};
    }
    const BackpropSetItem* item = GetBackpropSetItem(backprop_id.ordinal());
    if (item == nullptr) {
        throw XchainerError{"Invalid backprop ID, maybe already expired: ", backprop_id};
    }
}

void Context::ConnectBackpropIds(const BackpropId& backprop_id1, const BackpropId& backprop_id2) {
    XCHAINER_ASSERT(&backprop_id1.context() == this);
    XCHAINER_ASSERT(&backprop_id2.context() == this);
    if (backprop_id1 == backprop_id2) {
        // They are identical
        return;
    }
    BackpropSetItem* item1 = GetBackpropSetItem(backprop_id1.ordinal());
    BackpropSetItem* item2 = GetBackpropSetItem(backprop_id2.ordinal());
    if (item1 == nullptr || item2 == nullptr) {
        // At least one cannot be found
        return;
    }

    std::pair<BackpropOrdinal, BackpropOrdinal> pair = std::minmax(backprop_id1.ordinal(), backprop_id2.ordinal());
    if (backprop_connections_.end() != std::find(backprop_connections_.begin(), backprop_connections_.end(), pair)) {
        // Already in connection
        return;
    }

    // Add a new connection
    backprop_connections_.emplace_back(pair);
}

std::string Context::GetBackpropName(const BackpropId& backprop_id) {
    // Note: backprop name cannot be returned by reference, as the reference may be invalidated when a new graph is pushed to the backprop
    // set.

    BackpropSetItem* item = GetBackpropSetItem(backprop_id.ordinal());
    if (item == nullptr) {
        throw XchainerError{"Backprop not found in the context. Ordinal:", backprop_id.ordinal()};
    }
    return item->name;
}

void Context::CheckBackpropAllowed(const BackpropId& backprop_id) {
    BackpropSetItem* item = GetBackpropSetItem(backprop_id.ordinal());
    if (item == nullptr) {
        throw XchainerError{"Backprop ID not found: ", backprop_id};
    }
    if (item->prohibiting_ordinal.has_value()) {
        throw XchainerError{"Cannot backward for backprop ID '",
                            backprop_id,
                            "' because an connected backprop ID '",
                            BackpropId{*this, *item->prohibiting_ordinal},
                            "' which has been created earlier, has already been backpropped."};
    }
}

void Context::SetBackpropDone(const BackpropId& backprop_id) {
    BackpropSetItem* item = GetBackpropSetItem(backprop_id.ordinal());
    XCHAINER_ASSERT(item != nullptr);

    // Find connected backprop IDs
    std::vector<BackpropOrdinal> ordinals_to_prohibit;
    for (const std::pair<BackpropOrdinal, BackpropOrdinal>& pair : backprop_connections_) {
        if (pair.first == backprop_id.ordinal()) {
            ordinals_to_prohibit.emplace_back(pair.second);
        }
    }

    // Mark connected backprop IDs as prohibited.
    for (BackpropOrdinal ord : ordinals_to_prohibit) {
        BackpropSetItem* item2 = GetBackpropSetItem(ord);
        if (!item2->prohibiting_ordinal.has_value()) {
            item2->prohibiting_ordinal = backprop_id.ordinal();
        }
    }
}

std::vector<BackpropId> Context::GetInnerBackpropIds(const BackpropId& backprop_id) {
    std::vector<BackpropId> inner_backprop_ids;
    inner_backprop_ids.reserve(backprop_set_.size());
    for (const std::pair<BackpropOrdinal, BackpropOrdinal>& pair : backprop_connections_) {
        if (pair.first == backprop_id.ordinal()) {
            inner_backprop_ids.emplace_back(BackpropId{*this, pair.second});
        }
    }
    return inner_backprop_ids;
}

template <typename ThisPtr, typename ReturnType>
ReturnType Context::GetBackpropSetItemImpl(ThisPtr this_ptr, BackpropOrdinal ordinal) {
    // using reverse iterator because it's more likely to find earlier
    auto it = std::find_if(this_ptr->backprop_set_.rbegin(), this_ptr->backprop_set_.rend(), [ordinal](const BackpropSetItem& item) {
        return item.ordinal == ordinal;
    });
    if (it == this_ptr->backprop_set_.rend()) {
        return nullptr;
    }
    return &*it;
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
