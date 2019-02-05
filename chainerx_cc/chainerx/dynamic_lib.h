#pragma once

#include <string>

namespace chainerx {

// TODO(niboshi): More generalization is needed.
void* DlOpen(const std::string& filename, int flags);

void DlClose(void* handle);

void* DlSym(void* handle, const std::string& name);

}  // namespace chainerx
