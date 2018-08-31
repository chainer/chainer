#include "chainerx/graph.h"

#include <ostream>
#include <string>

#include "chainerx/context.h"
#include "chainerx/error.h"

namespace chainerx {

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id) {
    static constexpr const char* kExpiredBackpropDisplayName = "<expired>";
    try {
        os << backprop_id.GetName();
    } catch (const ChainerxError&) {
        os << kExpiredBackpropDisplayName;
    }
    return os;
}

std::string BackpropId::GetName() const { return context_.get().GetBackpropName(*this); }

void BackpropId::CheckValid() const { context_.get().CheckValidBackpropId(*this); }

}  // namespace chainerx
