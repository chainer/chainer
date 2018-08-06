#include "xchainer/graph.h"

#include <ostream>

namespace xchainer {

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id) {
    // TODO(niboshi): Implement backprop name lookup
    return os << backprop_id.sub_id();
}

}  // namespace xchainer
