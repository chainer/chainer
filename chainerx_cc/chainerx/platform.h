#pragma once

#include <string>

namespace chainerx {
namespace platform {

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

void* DlOpen(const std::string& filename);

void DlClose(void* handle);

void* DlSym(void* handle, const std::string& name);

}  // namespace platform
}  // namespace chainerx
