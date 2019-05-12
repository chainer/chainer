#include "chainerx/array_node.h"

#include "chainerx/array_body.h"

#include <memory>
#include <utility>

namespace chainerx {
namespace internal {

void SetArrayNodeWeakBody(ArrayNode& array_node, std::weak_ptr<ArrayBody> weak_body) { array_node.weak_body_ = std::move(weak_body); }

}  // namespace internal
}  // namespace chainerx
