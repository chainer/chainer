#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"

namespace chainerx {
/*namespace internal {

Array _stack_weight(std::vector<Array> ws);

std::vector<Array> _lstm(std::vector<Array> xs, Array h, Array c, std::vector<Array> ws, std::vector<Array> b);

template<typename Impl>
std::vector<std::vector<Array>> _one_directional_loop(Impl&& impl, std::vector<Array> xs, Array h, Array c,
	std::vector<Array> ws, std::vector<Array> b);

template <typename Impl>
std::vector<Array> n_step_rnn_impl(Impl&& impl, int64_t n_layers, Array hx, Array cx, std::vector<std::vector<Array>> ws, 
		std::vector<std::vector<Array>> bs, std::vector<Array> xs, int8_t use_bidirection);
}*/

std::vector<std::vector<Array>> n_step_lstm(int64_t n_layers, Array hx, Array cx, const std::vector<std::vector<Array>>& ws, 
		const std::vector<std::vector<Array>>& bs, std::vector<Array>& xs);
}