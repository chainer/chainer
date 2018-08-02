#include "xchainer/graph.h"

#include <ostream>
#include <string>

#include "xchainer/context.h"

namespace xchainer {

std::ostream& operator<<(std::ostream& os, const GraphId& graph_id) { return os << graph_id.name(); }

std::string GraphId::name() const { return context_.get().GetGraphName(*this); }

}  // namespace xchainer
