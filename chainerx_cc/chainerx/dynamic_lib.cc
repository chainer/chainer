#include "chainerx/dynamic_lib.h"

#include <string>

#include "chainerx/platform.h"

namespace chainerx {

void* DlOpen(const std::string& filename) { return platform::DlOpen(filename); }

void DlClose(void* handle) { platform::DlClose(handle); }

void* DlSym(void* handle, const std::string& name) { return platform::DlSym(handle, name); }

}  // namespace chainerx
