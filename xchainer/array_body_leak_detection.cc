#include "xchainer/array_body_leak_detection.h"

#include <memory>
#include <sstream>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

ArrayBodyLeakTracker* ArrayBodyLeakDetectionScope::array_body_leak_tracker_ = nullptr;

void ArrayBodyLeakTracker::operator()(const std::shared_ptr<internal::ArrayBody>& array_body) {
    // Keep weak pointer
    weak_ptrs_.emplace_back(array_body);
}

void ArrayBodyLeakTracker::CheckAllFreed() {
    std::vector<std::shared_ptr<internal::ArrayBody>> alive_ptrs;

    for (const std::weak_ptr<internal::ArrayBody> weak_ptr : weak_ptrs_) {
        std::shared_ptr<internal::ArrayBody> ptr = weak_ptr.lock();
        if (ptr != nullptr) {
            alive_ptrs.emplace_back(ptr);
        }
    }

    if (!alive_ptrs.empty()) {
        // TODO(niboshi): Output only array bodies that are not referenced from other array bodies
        std::ostringstream os;
        os << "Some array bodies are not freed." << std::endl << "Number of alive array bodies: " << alive_ptrs.size() << std::endl;
        for (const std::shared_ptr<internal::ArrayBody>& array_body : alive_ptrs) {
            Array array{array_body};
            os << "- Unreleased array body: " << array_body.get() << std::endl;
            os << array << std::endl;
            for (const std::shared_ptr<ArrayNode>& array_node : array.nodes()) {
                const GraphId& graph_id = array_node->graph_id();
                DebugDumpComputationalGraph(os, array, graph_id);
            }
        }
        throw GradientCheckError{os.str()};
    }
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
