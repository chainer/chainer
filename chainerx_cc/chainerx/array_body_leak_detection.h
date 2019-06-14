#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <vector>

#include "chainerx/array_body.h"

namespace chainerx {
namespace internal {

// Keep track of array body allocation.
// Used in combination with ArrayBodyLeakDetectionScope to detect leaks.
// This class is thread safe.
class ArrayBodyLeakTracker {
public:
    void operator()(const std::shared_ptr<ArrayBody>& array_body);

    // Returns the array bodies which are still alive.
    // It is useful to detect unreleased array bodies, leaking from the scope of ArrayBodyLeakDetectionScope.
    std::vector<std::shared_ptr<ArrayBody>> GetAliveArrayBodies() const;

    // Asserts all the array bodies are freed in the leak tracker.
    bool IsAllArrayBodiesFreed(std::ostream& os) const;

private:
    std::vector<std::weak_ptr<ArrayBody>> weak_ptrs_;
    mutable std::mutex mutex_;
};

// A scope object to detect array body leaks.
// It tracks newly created array bodies which are being set to arrays within the scope.
// New array bodies are reported to the tracker specified in the constructor.
// Only one leak detection scope can exist at any given moment
class ArrayBodyLeakDetectionScope {
public:
    explicit ArrayBodyLeakDetectionScope(ArrayBodyLeakTracker& tracker);
    ~ArrayBodyLeakDetectionScope();

    ArrayBodyLeakDetectionScope(const ArrayBodyLeakDetectionScope&) = delete;
    ArrayBodyLeakDetectionScope& operator=(const ArrayBodyLeakDetectionScope&) = delete;
    ArrayBodyLeakDetectionScope(ArrayBodyLeakDetectionScope&& other) = delete;
    ArrayBodyLeakDetectionScope& operator=(ArrayBodyLeakDetectionScope&& other) = delete;

    static ArrayBodyLeakTracker* GetGlobalTracker() { return array_body_leak_tracker_; }

private:
    // The global array body leak tracker.
    static ArrayBodyLeakTracker* array_body_leak_tracker_;
};

// Throws an ChainerxError if any leakage is detected. Else, does nothing.
void CheckAllArrayBodiesFreed(ArrayBodyLeakTracker& tracker);

}  // namespace internal
}  // namespace chainerx
