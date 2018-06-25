#include "xchainer/array_node.h"

#include <cassert>
#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_body.h"

namespace xchainer {

void ArrayNode::ClearGrad() noexcept {
    std::shared_ptr<internal::ArrayBody> body = body_.lock();
    if (body == nullptr) {
        // Array body (and also the grad) is already gone.
        return;
    }
    body->ClearGrad(graph_id_);
}

}  // namespace xchainer
