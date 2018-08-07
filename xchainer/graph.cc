#include "xchainer/graph.h"

#include <ostream>
#include <string>

#include "xchainer/context.h"
#include "xchainer/error.h"

namespace xchainer {

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id) {
    static constexpr const char* kExpiredBackpropDisplayName = "<expired>";
    try {
        const std::string& name = backprop_id.GetName();
        os << name;
    } catch (const XchainerError& e) {
        os << kExpiredBackpropDisplayName;
    }
    return os;
}

std::string BackpropId::GetName() const { return context_.get().GetBackpropName(*this); }

}  // namespace xchainer
