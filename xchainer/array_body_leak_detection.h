#pragma once

#include <memory>
#include <vector>

#include "xchainer/array_body.h"

namespace xchainer {
namespace internal {

class ArrayBodyLeakTracker {
public:
    void operator()(const std::shared_ptr<internal::ArrayBody>& array_body);

    std::vector<std::shared_ptr<ArrayBody>> GetAliveArrayBodies() const;

private:
    std::vector<std::weak_ptr<internal::ArrayBody>> weak_ptrs_;
};

// A scope object to detect array body leaks.
// It tracks newly created array bodies which are being set to arrays within the scope.
class ArrayBodyLeakDetectionScope {
public:
    explicit ArrayBodyLeakDetectionScope(ArrayBodyLeakTracker& tracker);
    ~ArrayBodyLeakDetectionScope();

    ArrayBodyLeakDetectionScope(const ArrayBodyLeakDetectionScope&) = delete;
    ArrayBodyLeakDetectionScope& operator=(const ArrayBodyLeakDetectionScope&) = delete;
    ArrayBodyLeakDetectionScope(ArrayBodyLeakDetectionScope&& other) {
        exited_ = other.exited_;
        other.exited_ = true;
    }
    ArrayBodyLeakDetectionScope& operator=(ArrayBodyLeakDetectionScope&& other) {
        exited_ = other.exited_;
        other.exited_ = true;
        return *this;
    }

private:
    friend class xchainer::Array;

    static ArrayBodyLeakTracker* GetGlobalTracker() { return array_body_leak_tracker_; }

    // The global array body leak tracker.
    static ArrayBodyLeakTracker* array_body_leak_tracker_;

    bool exited_ = false;
};

}  // namespace internal
}  // namespace xchainer
