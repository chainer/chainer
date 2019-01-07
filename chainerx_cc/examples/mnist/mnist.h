#pragma once

#include <string>

#include "chainerx/array.h"

// Returns MNIST images as a two dimensional array.
chainerx::Array ReadMnistImages(const std::string& filename);

// Returns MNIST labels as a one dimensional array.
chainerx::Array ReadMnistLabels(const std::string& filename);
