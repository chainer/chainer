#include "chainerx/array_body_leak_detection.h"

#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/array_node.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace internal {

ArrayBodyLeakTracker* ArrayBodyLeakDetectionScope::array_body_leak_tracker_ = nullptr;

void ArrayBodyLeakTracker::operator()(const std::shared_ptr<ArrayBody>& array_body) {
    std::lock_guard<std::mutex> lock{mutex_};
    // Keep weak pointer
    weak_ptrs_.emplace_back(array_body);
}

std::vector<std::shared_ptr<ArrayBody>> ArrayBodyLeakTracker::GetAliveArrayBodies() const {
    std::lock_guard<std::mutex> lock{mutex_};
    std::vector<std::shared_ptr<ArrayBody>> alive_ptrs;
    for (const std::weak_ptr<ArrayBody>& weak_ptr : weak_ptrs_) {
        if (std::shared_ptr<ArrayBody> ptr = weak_ptr.lock()) {
            alive_ptrs.emplace_back(ptr);
        }
    }
    return alive_ptrs;
}

bool ArrayBodyLeakTracker::IsAllArrayBodiesFreed(std::ostream& os) const {
    std::vector<std::shared_ptr<internal::ArrayBody>> alive_arr_bodies = GetAliveArrayBodies();
    if (!alive_arr_bodies.empty()) {
        // TODO(niboshi): Output only array bodies that are not referenced from other array bodies
        os << "Some array bodies are not freed.\n";
        os << "Number of alive array bodies: " << alive_arr_bodies.size() << "\n";
        for (const std::shared_ptr<internal::ArrayBody>& array_body : alive_arr_bodies) {
            Array array{array_body};
            os << "- Unreleased array body: " << array_body.get() << "\n";
            os << array << "\n";
            for (const std::shared_ptr<ArrayNode>& array_node : internal::GetArrayBody(array)->nodes()) {
                const BackpropId& backprop_id = array_node->backprop_id();
                DebugDumpComputationalGraph(os, array, backprop_id);
            }
        }
        return false;
    }
    return true;
}

ArrayBodyLeakDetectionScope ::ArrayBodyLeakDetectionScope(ArrayBodyLeakTracker& tracker) {
    CHAINERX_ASSERT(array_body_leak_tracker_ == nullptr);  // nested use is not supported
    array_body_leak_tracker_ = &tracker;
}

ArrayBodyLeakDetectionScope ::~ArrayBodyLeakDetectionScope() { array_body_leak_tracker_ = nullptr; }

void CheckAllArrayBodiesFreed(ArrayBodyLeakTracker& tracker) {
    std::ostringstream os;
    if (!tracker.IsAllArrayBodiesFreed(os)) {
        throw ChainerxError{os.str()};
    }
}

}  // namespace internal
}  // namespace chainerx
