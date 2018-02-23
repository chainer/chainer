#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"

namespace xchainer {

class Context {
public:
    // Gets the backend specified by the name.
    // If the backend does not exist, this function automatically creates it.
    Backend& GetBackend(const std::string& backend_name);

    // Gets the device specified by the device ID.
    // If the backend and/or device do not exist, this function automatically creates them.
    Device& GetDevice(const DeviceId& device_id);

    // Gets/sets the default device of this context.
    void set_default_device(Device& device) {
        if (this != &device.backend().context()) {
            throw ContextError("Context mismatch.");
        }
        std::lock_guard<std::mutex> lock{mutex_};
        default_device_ = &device;
    }

    Device& default_device() const {
        std::lock_guard<std::mutex> lock{mutex_};
        if (default_device_ == nullptr) {
            throw ContextError("Global default device is not set.");
        }
        return *default_device_;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
    Device* default_device_ = nullptr;
    mutable std::mutex mutex_;
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

// Scope object that switches the default context by RAII.
class ContextScope {
public:
    ContextScope() : orig_(internal::GetDefaultContextNoExcept()), exited_(false) {}
    explicit ContextScope(Context& context) : ContextScope() { SetDefaultContext(&context); }

    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
    ContextScope& operator=(ContextScope&& other) = delete;

    ContextScope(ContextScope&& other): orig_(other.orig_), exited_(other.exited_) {
		other.exited_ = true;
	}

    ~ContextScope() { Exit(); }

    // Explicitly recovers the original context. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultContext(orig_);
            exited_ = true;
        }
    }

private:
    Context* orig_;
    bool exited_;
};

}  // namespace xchainer
