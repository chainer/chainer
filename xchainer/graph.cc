#include "xchainer/graph.h"

#include <ostream>
#include <string>

#include "xchainer/context.h"
#include "xchainer/error.h"

namespace xchainer {

std::ostream& operator<<(std::ostream& os, const GraphId& graph_id) {
    static constexpr const char* kExpiredGraphDisplayName = "<expired>";
    try {
        const std::string& name = graph_id.GetName();
        os << name;
    } catch (const XchainerError& e) {
        os << kExpiredGraphDisplayName;
    }
    return os;
}

std::string GraphId::GetName() const { return context_.get().GetGraphName(*this); }

}  // namespace xchainer
