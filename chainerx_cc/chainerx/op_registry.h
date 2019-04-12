#pragma once

#include <memory>
#include <mutex>
#include <typeindex>
#include <unordered_map>

#include "chainerx/error.h"
#include "chainerx/op.h"

namespace chainerx {

// Manages dynamic registration and dispatch of ops.
// This class is hierarchical: it has an optional pointer to a parent OpRegistry and falls back if an op is not found in this instance.
class OpRegistry {
public:
    OpRegistry() {}

    explicit OpRegistry(OpRegistry* parent) : parent_{parent} {}

    // Registers an op.
    // Registers an instance of OpType with the type_index of KeyOpType as the key.
    // OpType must be a subclass of KeyOpType.
    template <typename KeyOpType, typename OpType>
    void RegisterOp() {
        static_assert(std::is_base_of<KeyOpType, OpType>::value, "OpType must be a subclass of KeyOpType.");
        std::lock_guard<std::mutex> lock{*mutex_};
        auto pair = ops_.emplace(std::type_index{typeid(KeyOpType)}, std::make_unique<OpType>());
        if (!pair.second) {
            throw ChainerxError{"Duplicate op: ", KeyOpType::name()};
        }
    }

    // Looks up an op.
    template <typename KeyOpType>
    Op& GetOp() {
        std::type_index key{typeid(KeyOpType)};
        {
            std::lock_guard<std::mutex> lock{*mutex_};
            auto it = ops_.find(key);
            if (it != ops_.end()) {
                return *it->second;
            }
        }
        if (parent_ != nullptr) {
            return parent_->GetOp<KeyOpType>();
        }
        throw ChainerxError{"Op not found: ", KeyOpType::name()};
    }

private:
    std::unique_ptr<std::mutex> mutex_{std::make_unique<std::mutex>()};

    OpRegistry* parent_{};

    std::unordered_map<std::type_index, std::unique_ptr<Op>> ops_{};
};

namespace internal {

// A facility to register ops statically.
template <typename BackendType, typename KeyOpType, typename OpType>
class OpRegistrar {
public:
    OpRegistrar() noexcept {
        OpRegistry& op_registry = BackendType::GetGlobalOpRegistry();
        op_registry.RegisterOp<KeyOpType, OpType>();
    }
};

}  // namespace internal
}  // namespace chainerx
