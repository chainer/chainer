#include "xchainer/array_body_leak_detection.h"

#include <memory>
#include <sstream>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

ArrayBodyLeakTracker* ArrayBodyLeakDetectionScope::array_body_leak_tracker_ = nullptr;

void ArrayBodyLeakTracker::operator()(const std::shared_ptr<internal::ArrayBody>& array_body) {
    // Keep weak pointer
    weak_ptrs_.emplace_back(array_body);
}

std::vector<std::shared_ptr<ArrayBody>> ArrayBodyLeakTracker::GetAliveArrayBodies() const {
    std::vector<std::shared_ptr<ArrayBody>> alive_ptrs;
    for (const std::weak_ptr<ArrayBody> weak_ptr : weak_ptrs_) {
        std::shared_ptr<ArrayBody> ptr = weak_ptr.lock();
        if (ptr != nullptr) {
            alive_ptrs.emplace_back(ptr);
        }
    }
    return alive_ptrs;
}

ArrayBodyLeakDetectionScope ::ArrayBodyLeakDetectionScope(ArrayBodyLeakTracker& tracker) {
    assert(array_body_leak_tracker_ == nullptr);  // nested use is not supported
    array_body_leak_tracker_ = &tracker;
}

ArrayBodyLeakDetectionScope ::~ArrayBodyLeakDetectionScope() {
    if (!exited_) {
        array_body_leak_tracker_ = nullptr;
        exited_ = true;
    }
}

}  // namespace internal
}  // namespace xchainer
