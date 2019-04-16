#pragma once

#include <memory>
#include <mutex>
#include <typeindex>
#include <unordered_map>

#include "chainerx/error.h"
#include "chainerx/kernel.h"

namespace chainerx {

// Manages dynamic registration and dispatch of kernels.
// This class is hierarchical: it has an optional pointer to a parent KernelRegistry and falls back if a kernel is not found in this
// instance.
class KernelRegistry {
public:
    KernelRegistry() {}

    explicit KernelRegistry(KernelRegistry* parent) : parent_{parent} {}

    // Registers a kernel.
    // Registers an instance of KernelType with the type_index of KeyKernelType as the key.
    // KernelType must be a subclass of KeyKernelType.
    template <typename KeyKernelType, typename KernelType>
    void RegisterKernel() {
        static_assert(std::is_base_of<KeyKernelType, KernelType>::value, "KernelType must be a subclass of KeyKernelType.");
        std::lock_guard<std::mutex> lock{*mutex_};
        auto pair = kernels_.emplace(std::type_index{typeid(KeyKernelType)}, std::make_unique<KernelType>());
        if (!pair.second) {
            throw ChainerxError{"Duplicate kernel: ", KeyKernelType::name()};
        }
    }

    // Looks up a kernel.
    template <typename KeyKernelType>
    Kernel& GetKernel() {
        std::type_index key{typeid(KeyKernelType)};
        {
            std::lock_guard<std::mutex> lock{*mutex_};
            auto it = kernels_.find(key);
            if (it != kernels_.end()) {
                return *it->second;
            }
        }
        if (parent_ != nullptr) {
            return parent_->GetKernel<KeyKernelType>();
        }
        throw ChainerxError{"Kernel not found: ", KeyKernelType::name()};
    }

private:
    std::unique_ptr<std::mutex> mutex_{std::make_unique<std::mutex>()};

    KernelRegistry* parent_{};

    std::unordered_map<std::type_index, std::unique_ptr<Kernel>> kernels_{};
};

namespace internal {

// A facility to register kernels statically.
template <typename BackendType, typename KeyKernelType, typename KernelType>
class KernelRegistrar {
public:
    KernelRegistrar() noexcept {
        KernelRegistry& kernel_registry = BackendType::GetGlobalKernelRegistry();
        kernel_registry.RegisterKernel<KeyKernelType, KernelType>();
    }
};

}  // namespace internal
}  // namespace chainerx
