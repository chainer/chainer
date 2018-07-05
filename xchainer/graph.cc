#include "xchainer/graph.h"

#include <sstream>
#include <string>

namespace xchainer {

bool GraphId::operator==(const GraphId& other) const {
    switch (type_) {
        case kNamed:
            return graph_id_ == other.graph_id_;
        default:
            return type_ == other.type_;
    }
}

std::string GraphId::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const GraphId& graph_id) {
    // TODO(hvy): Fix string representations of "default" and "any".
    switch (graph_id.type()) {
        case GraphId::kNamed:
            return os << graph_id.graph_id();
        case GraphId::kDefault:
            return os << "default";
        default:
            return os << "any";
    }
}

}  // namespace xchainer
