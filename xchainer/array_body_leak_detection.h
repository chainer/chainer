#pragma once

#include <memory>
#include <vector>

#include "xchainer/array_body.h"

namespace xchainer {
namespace internal {

// Keep track of array body allocation.
// Used in combination with ArrayBodyLeakDetectionScope to detect leaks.
class ArrayBodyLeakTracker {
public:
    void operator()(const std::shared_ptr<ArrayBody>& array_body);

    // Returns the array bodies which are still alive.
    // It is useful to detect unreleased array bodies, leaking from the scope of ArrayBodyLeakDetectionScope.
    std::vector<std::shared_ptr<ArrayBody>> GetAliveArrayBodies() const;

private:
    std::vector<std::weak_ptr<ArrayBody>> weak_ptrs_;
};

// A scope object to detect array body leaks.
// It tracks newly created array bodies which are being set to arrays within the scope.
// New array bodies are reported to the tracker specified in the constructor.
class ArrayBodyLeakDetectionScope {
public:
    explicit ArrayBodyLeakDetectionScope(ArrayBodyLeakTracker& tracker);
    ~ArrayBodyLeakDetectionScope();

    ArrayBodyLeakDetectionScope(const ArrayBodyLeakDetectionScope&) = delete;
    ArrayBodyLeakDetectionScope& operator=(const ArrayBodyLeakDetectionScope&) = delete;
    ArrayBodyLeakDetectionScope(ArrayBodyLeakDetectionScope&& other) = delete;
    ArrayBodyLeakDetectionScope& operator=(ArrayBodyLeakDetectionScope&& other) = delete;

private:
    friend class xchainer::Array;

    static ArrayBodyLeakTracker* GetGlobalTracker() { return array_body_leak_tracker_; }

    // The global array body leak tracker.
    static ArrayBodyLeakTracker* array_body_leak_tracker_;
};

}  // namespace internal
}  // namespace xchainer
