#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace native {

class NativeBackend;

}  // namespace native

// TODO(sonots): Hide BackpropId-related functions from users.
// TODO(sonots): Move implementations of BackpropId-releated functions into another class.
class Context {
public:
    Context() = default;
    ~Context();

    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    // Gets the backend specified by the name.
    // If the backend does not exist, this function automatically creates it.
    Backend& GetBackend(const std::string& backend_name);

    // Gets the native backend.
    native::NativeBackend& GetNativeBackend();

    // Gets the device specified by the device ID.
    // If the backend and/or device do not exist, this function automatically creates them.
    Device& GetDevice(const DeviceId& device_id);

    BackpropId MakeNextBackpropId(std::string backprop_name);

    void ReleaseBackpropId(const BackpropId& backprop_id);

    // Checks if the backprop ID is allowed to be backpropped.
    // Backprop is allowed if the order of backprop IDs which have been backpropped is not reversed in any of the previous backprop scopes.
    // XchainerError is thrown if the check fails.
    void CheckBackpropAllowed(const BackpropId& backprop_id);

    // Flags the backprop ID that it has been backpropped.
    void SetBackpropDone(const BackpropId& backprop_id);

    // Returns all backprop IDs created after the queried graph.
    // In many cases, these are also the graphs created in inner scopes.
    // The queried graph is excluded from the returned container.
    std::vector<BackpropId> GetInnerBackpropIds(const BackpropId& backprop_id);

    BackpropId default_backprop_id() {
        // 0 is the graph sub id of the default graph.
        return BackpropId{*this, 0};
    }

private:
    // TODO(niboshi): Support multi-thread usage
    struct GraphStackItem {
        GraphStackItem(GraphSubId sub_id, std::string name) : sub_id{sub_id}, name{std::move(name)} {}

        GraphSubId sub_id;
        std::string name;

        // Indicates whether Backward on any outer graphs (note that this graph is not included) has been called.
        bool is_outer_graph_backpropped{false};
    };

    std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
    std::vector<void*> dlopen_handles_;
    mutable std::mutex mutex_;

    GraphSubId next_graph_sub_id_{1};  // 1 is the first graph sub id after the default graph whose graph sub id is 0.

    std::vector<GraphStackItem> graph_stack_{};
};

// Gets/sets the context that used by default when current context is not set.
Context& GetGlobalDefaultContext();
void SetGlobalDefaultContext(Context* context);

namespace internal {

Context* GetDefaultContextNoExcept() noexcept;

}  // namespace internal

// Gets thread local default context.
Context& GetDefaultContext();

// Sets thread local default context.
//
// The thread local default device is reset to null if given context is different with previous default context.
void SetDefaultContext(Context* context);

// Returns the specified device on the default context.
inline Device& GetDevice(const DeviceId& device_id) { return GetDefaultContext().GetDevice(device_id); }

// Returns the specified backend on the default context.
inline Backend& GetBackend(const std::string& backend_name) { return GetDefaultContext().GetBackend(backend_name); }

// Returns the native backend on the default context.
inline native::NativeBackend& GetNativeBackend() { return GetDefaultContext().GetNativeBackend(); }

// Scope object that switches the default context by RAII.
class ContextScope {
public:
    ContextScope() : orig_ctx_{internal::GetDefaultContextNoExcept()}, orig_device_{internal::GetDefaultDeviceNoExcept()}, exited_{false} {}
    explicit ContextScope(Context& context) : ContextScope{} { SetDefaultContext(&context); }

    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
    ContextScope& operator=(ContextScope&& other) = delete;

    ContextScope(ContextScope&& other) : orig_ctx_{other.orig_ctx_}, orig_device_{other.orig_device_}, exited_{other.exited_} {
        other.exited_ = true;
    }

    ~ContextScope() { Exit(); }

    // Explicitly recovers the original context. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultContext(orig_ctx_);
            SetDefaultDevice(orig_device_);
            exited_ = true;
        }
    }

private:
    Context* orig_ctx_;
    Device* orig_device_;
    bool exited_;
};

}  // namespace xchainer
