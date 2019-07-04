#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/backend_util.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/kernels/rnn.h"
#include "chainerx/macro.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace {

__global__ void initGPUData_ker(float* data, int numElements, float* value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        data[tid] = value[tid];
    }
}

void initGPUData(float* data, int numElements, float* value) {
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 1024;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
    initGPUData_ker<<<gridDim, blockDim>>>(data, numElements, value);
}

void weights_forward(
        cuda_internal::DeviceInternals& device_internals,
        cudnnRNNDescriptor_t rnn_desc,
        const std::vector<std::vector<Array>> ws,
        const std::vector<std::vector<Array>> bs,
        int n_layers,
        int num_directions,
        cudnnTensorDescriptor_t x_desc,
        cudnnFilterDescriptor_t w_desc,
        Array& w) {
    for (int layer = 0; layer < n_layers; layer++) {
        for (int8_t di = 0; di < num_directions; di++) {
            for (uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                float* m_offset;
                device_internals.cudnn_handle().Call(
                        cudnnGetRNNLinLayerMatrixParams,
                        rnn_desc,
                        index,
                        x_desc,
                        w_desc,
                        internal::GetRawOffsetData(w),
                        lin_layer_id,
                        linlayermatdesc,
                        reinterpret_cast<void**>(&m_offset));
                cudnnDataType_t dataType;
                cudnnTensorFormat_t format;
                int nbDims;
                int filterDimA[3];

                cudnnGetFilterNdDescriptor(linlayermatdesc, 3, &dataType, &format, &nbDims, filterDimA);
                Array w_temp = AsContiguous(ws[index][lin_layer_id].AsType(Dtype::kFloat32));
                initGPUData(
                        m_offset,
                        filterDimA[0] * filterDimA[1] * filterDimA[2],
                        reinterpret_cast<float*>(internal::GetRawOffsetData(w_temp)));
                cudnnDestroyFilterDescriptor(linlayermatdesc);

                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                float* b_offset;
                device_internals.cudnn_handle().Call(
                        cudnnGetRNNLinLayerBiasParams,
                        rnn_desc,
                        index,
                        x_desc,
                        w_desc,
                        internal::GetRawOffsetData(w),
                        lin_layer_id,
                        linlayerbiasdesc,
                        reinterpret_cast<void**>(&b_offset));
                cudnnGetFilterNdDescriptor(linlayerbiasdesc, 3, &dataType, &format, &nbDims, filterDimA);
                Array b_temp = AsContiguous(bs[index][lin_layer_id].AsType(Dtype::kFloat32));
                initGPUData(
                        b_offset,
                        filterDimA[0] * filterDimA[1] * filterDimA[2],
                        reinterpret_cast<float*>(internal::GetRawOffsetData(b_temp)));
                cudnnDestroyFilterDescriptor(linlayerbiasdesc);
            }
        }
    }
}

std::vector<Array> weights_backward(
        CudaDevice& device,
        cudnnRNNDescriptor_t& rnn_desc,
        cudnnTensorDescriptor_t dummy_x_desc,
        cudnnFilterDescriptor_t w_desc,
        Array w,
        std::vector<std::vector<Array>> ws,
        std::vector<std::vector<Array>> bs,
        int64_t n_layers,
        int64_t num_directions,
        Dtype type) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
    std::vector<Array> ret;
    for (int64_t layer = 0; layer < n_layers; layer++) {
        for (int8_t di = 0; di < num_directions; di++) {
            for (uint lin_layer_id = 0; lin_layer_id < ws[0].size(); lin_layer_id++) {
                int64_t index = num_directions * layer + di;
                cudnnFilterDescriptor_t linlayermatdesc;
                cudnnCreateFilterDescriptor(&linlayermatdesc);
                float* m_offset;
                device_internals.cudnn_handle().Call(
                        cudnnGetRNNLinLayerMatrixParams,
                        rnn_desc,
                        index,
                        dummy_x_desc,
                        w_desc,
                        internal::GetRawOffsetData(w),
                        lin_layer_id,
                        linlayermatdesc,
                        reinterpret_cast<void**>(&m_offset));
                Array m = AsContiguous(Zeros(ws[index][lin_layer_id].shape(), type, ws[index][lin_layer_id].device()));
                cudnnDataType_t dataType;
                cudnnTensorFormat_t format;
                int nbDims;
                int filterDimA[3];

                cudnnGetFilterNdDescriptor(linlayermatdesc, 3, &dataType, &format, &nbDims, filterDimA);
                initGPUData(
                        reinterpret_cast<float*>(internal::GetRawOffsetData(m)), filterDimA[0] * filterDimA[1] * filterDimA[2], m_offset);
                cudnnDestroyFilterDescriptor(linlayermatdesc);
                ret.push_back(m);
                cudnnFilterDescriptor_t linlayerbiasdesc;
                cudnnCreateFilterDescriptor(&linlayerbiasdesc);
                float* b_offset;
                device_internals.cudnn_handle().Call(
                        cudnnGetRNNLinLayerBiasParams,
                        rnn_desc,
                        index,
                        dummy_x_desc,
                        w_desc,
                        internal::GetRawOffsetData(w),
                        lin_layer_id,
                        linlayerbiasdesc,
                        reinterpret_cast<void**>(&b_offset));
                Array b = AsContiguous(Zeros(bs[index][lin_layer_id].shape(), type, bs[index][lin_layer_id].device()));
                cudnnGetFilterNdDescriptor(linlayerbiasdesc, 3, &dataType, &format, &nbDims, filterDimA);
                initGPUData(
                        reinterpret_cast<float*>(internal::GetRawOffsetData(b)), filterDimA[0] * filterDimA[1] * filterDimA[2], b_offset);
                cudnnDestroyFilterDescriptor(linlayerbiasdesc);
                ret.push_back(b);
            }
        }
    }
    return ret;
}

class CudaRnnKernel : public RnnKernel {
public:
    std::tuple<std::vector<std::vector<Array>>, std::unique_ptr<chainerx::RnnGradState>> Call(
            int64_t n_layers,
            Array hx,
            absl::optional<Array> cx,
            const std::vector<std::vector<Array>>& ws,
            const std::vector<std::vector<Array>>& bs,
            const std::vector<Array>& xs,
            const int8_t bidirectional,
            const int8_t mode) override {
        CudaDevice& device = dynamic_cast<CudaDevice&>(hx.device());
        CudaSetDeviceScope scope{device.index()};
        auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT
        Dtype type = hx.dtype();
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);

        const auto input_dim = xs[0].shape()[1];
        const auto hidden_dim = hx.shape()[2];
        const auto num_directions = bidirectional == 1 ? 2 : 1;
        const auto num_layers = n_layers;
        const auto rnn_direction = bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
        const auto rnn_mode = mode == 1 ? CUDNN_LSTM : CUDNN_GRU;
        const auto rnn_input = CUDNN_LINEAR_INPUT;
        cudnnRNNDescriptor_t rnn_desc;
        uint64_t seed = 1337ull;

        cudnnDropoutDescriptor_t dropoutDesc;
        cudnnCreateDropoutDescriptor(&dropoutDesc);

        size_t stateSize;
        void* states;
        device_internals.cudnn_handle().Call(cudnnDropoutGetStatesSize, &stateSize);

        cudaMalloc(&states, stateSize);

        cudnnSetDropoutDescriptor(dropoutDesc, device_internals.cudnn_handle().handle(), 0, states, stateSize, seed);
        cudnnCreateRNNDescriptor(&rnn_desc);
        device_internals.cudnn_handle().Call(
                cudnnSetRNNDescriptor,
                rnn_desc,
                hidden_dim,
                num_layers,
                dropoutDesc,
                rnn_input,
                rnn_direction,
                rnn_mode,
                CUDNN_RNN_ALGO_STANDARD,
                CUDNN_DATA_FLOAT);
        cudnnTensorDescriptor_t *x_desc, *y_desc;
        x_desc = reinterpret_cast<cudnnTensorDescriptor_t*>(malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)));
        y_desc = reinterpret_cast<cudnnTensorDescriptor_t*>(malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)));
        std::vector<cuda_internal::CudnnTensorDescriptor> xs_desc;
        std::vector<cuda_internal::CudnnTensorDescriptor> ys_desc;
        std::vector<Array> ys;
        std::vector<Array> xs_cont;
        for (uint i = 0; i < xs.size(); i++) {
            Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
            Shape ys_shape{xs[i].shape()[0], num_directions * hidden_dim, 1};
            xs_cont.push_back(AsContiguous(xs[i].AsType(Dtype::kFloat32)));
            ys.push_back(
                    AsContiguous(Zeros({xs_cont[i].shape()[0], num_directions * hidden_dim}, xs_cont[i].dtype(), xs_cont[i].device())));
            xs_desc.push_back(cuda_internal::CudnnTensorDescriptor(Reshape(xs_cont[i], xs_shape)));
            ys_desc.push_back(cuda_internal::CudnnTensorDescriptor(Reshape(ys[i], ys_shape)));
            x_desc[i] = *xs_desc[i];
            y_desc[i] = *ys_desc[i];
        }
        Array x = Concatenate(xs_cont, 0);
        Array y = Concatenate(ys, 0);

        size_t weight_size;
        device_internals.cudnn_handle().Call(cudnnGetRNNParamsSize, rnn_desc, x_desc[0], &weight_size, CUDNN_DATA_FLOAT);
        Array w = AsContiguous(Zeros({static_cast<int>(weight_size) / 4, 1, 1}, x.dtype(), x.device()));
        cuda_internal::CudnnFilterDescriptor wDesc{w};

        weights_forward(device_internals, rnn_desc, ws, bs, n_layers, num_directions, x_desc[0], *wDesc, w);
        size_t workSize;
        size_t reserve_size;
        device_internals.cudnn_handle().Call(cudnnGetRNNWorkspaceSize, rnn_desc, xs.size(), x_desc, &workSize);
        Array workspace = AsContiguous(Zeros({static_cast<int64_t>(workSize)}, hx.dtype(), hx.device()));

        device_internals.cudnn_handle().Call(cudnnGetRNNTrainingReserveSize, rnn_desc, xs.size(), x_desc, &reserve_size);
        Array reserve = AsContiguous(Zeros({static_cast<int64_t>(reserve_size)}, hx.dtype(), hx.device()));
        hx = AsContiguous(hx.AsType(Dtype::kFloat32));
        Array hy = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        Array cy = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnTensorDescriptor hxDesc{hx};
        Array _cx;
        if (cx.has_value()) {
            _cx = AsContiguous((*cx).AsType(Dtype::kFloat32));
        } else {
            _cx = Zeros(hx.shape(), hx.dtype(), hx.device());
        }
        cuda_internal::CudnnTensorDescriptor cxDesc{_cx};
        cuda_internal::CudnnTensorDescriptor hyDesc{hy};
        cuda_internal::CudnnTensorDescriptor cyDesc{cy};
        device_internals.cudnn_handle().Call(
                cudnnRNNForwardTraining,
                rnn_desc,
                xs.size(),
                x_desc,
                internal::GetRawOffsetData(x),
                *hxDesc,
                internal::GetRawOffsetData(hx),
                *cxDesc,
                internal::GetRawOffsetData(_cx),
                *wDesc,
                internal::GetRawOffsetData(w),
                y_desc,
                internal::GetRawOffsetData(y),
                *hyDesc,
                internal::GetRawOffsetData(hy),
                *cyDesc,
                internal::GetRawOffsetData(cy),
                internal::GetRawOffsetData(workspace),
                workSize,
                internal::GetRawOffsetData(reserve),
                reserve_size);
        std::vector<int64_t> split_indices;
        for (uint i = 0; i < xs.size() - 1; i++) {
            if (i != 0) {
                split_indices.push_back(split_indices[i - 1] + xs[i].shape()[0]);
            } else {
                split_indices.push_back(xs[i].shape()[0]);
            }
        }

        std::unique_ptr<RnnGradState> state = std::make_unique<GenericRnnGradState>(rnn_desc, *wDesc, w, reserve, workspace);
        y = y.AsType(type);
        ys = Split(y, split_indices, 0);

        std::vector<Array> out_states;
        out_states.push_back(hy.AsType(type));
        if (cx.has_value()) {
            out_states.push_back(cy.AsType(type));
        }
        std::vector<std::vector<Array>> ret;
        ret.push_back(out_states);
        ret.push_back(ys);
        return std::make_tuple(std::move(ret), std::move(state));
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RnnKernel, CudaRnnKernel);

class CudaRnnBackwardKernel : public RnnBackwardKernel {
public:
    std::vector<std::vector<Array>> Call(
            int64_t n_layers,
            Array hx,
            absl::optional<Array> cx,
            const std::vector<std::vector<Array>>& ws,
            const std::vector<std::vector<Array>>& bs,
            const std::vector<Array>& xs,
            Array dhy,
            absl::optional<Array> dcy,
            std::vector<Array> ys,
            std::vector<Array> dys,
            const int8_t bidirectional,
            const std::shared_ptr<chainerx::RnnGradState>& state) override {
        CudaDevice& device = dynamic_cast<CudaDevice&>(hx.device());
        CudaSetDeviceScope scope{device.index()};

        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        auto cuda_state = dynamic_cast<GenericRnnGradState&>(*state);
        Dtype type = hx.dtype();
        const auto input_dim = xs[0].shape()[1];
        const auto hidden_dim = hx.shape()[2];
        const auto num_directions = bidirectional == 1 ? 2 : 1;

        cudnnRNNDescriptor_t rnn_desc = cuda_state.rnn_desc();
        std::vector<Array> dxs;
        std::vector<cuda_internal::CudnnTensorDescriptor> xsDesc, dxsDesc, ysDesc, dysDesc;
        cudnnTensorDescriptor_t* xs_desc = reinterpret_cast<cudnnTensorDescriptor_t*>(malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)));
        cudnnTensorDescriptor_t* dxs_desc = reinterpret_cast<cudnnTensorDescriptor_t*>(malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)));
        cudnnTensorDescriptor_t* ys_desc = reinterpret_cast<cudnnTensorDescriptor_t*>(malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)));
        cudnnTensorDescriptor_t* dys_desc = reinterpret_cast<cudnnTensorDescriptor_t*>(malloc(xs.size() * sizeof(cudnnTensorDescriptor_t)));
        std::vector<Array> xs_cont;
        for (uint i = 0; i < xs.size(); i++) {
            Shape xs_shape{xs[i].shape()[0], xs[i].shape()[1], 1};
            Shape ys_shape{ys[i].shape()[0], ys[i].shape()[1], 1};

            xs_cont.push_back(AsContiguous(xs[i].AsType(Dtype::kFloat32)));
            ys[i] = AsContiguous(ys[i].AsType(Dtype::kFloat32));
            dys[i] = AsContiguous(dys[i].AsType(Dtype::kFloat32));
            xsDesc.push_back(cuda_internal::CudnnTensorDescriptor(Reshape(xs_cont[i], xs_shape)));
            xs_desc[i] = *xsDesc[i];
            dxs.push_back(AsContiguous(Zeros(xs_cont[i].shape(), xs_cont[i].dtype(), xs_cont[i].device())));
            dxsDesc.push_back(cuda_internal::CudnnTensorDescriptor(Reshape(dxs[i], xs_shape)));
            dxs_desc[i] = *dxsDesc[i];
            ysDesc.push_back(cuda_internal::CudnnTensorDescriptor(Reshape(ys[i], ys_shape)));
            ys_desc[i] = *ysDesc[i];
            dysDesc.push_back(cuda_internal::CudnnTensorDescriptor(Reshape(dys[i], ys_shape)));
            dys_desc[i] = *dysDesc[i];
        }
        Array dx = AsContiguous(Concatenate(dxs, 0));
        Array x = AsContiguous(Concatenate(xs_cont, 0));
        Array y = AsContiguous(Concatenate(ys, 0));
        Array dy = AsContiguous(Concatenate(dys, 0));
        cudnnFilterDescriptor_t wDesc = cuda_state.wDesc();
        Array w = AsContiguous(cuda_state.w());
        Array reserve = AsContiguous(cuda_state.reserve());
        Array workspace = AsContiguous(cuda_state.workspace());
        size_t reserve_size = reserve.shape()[0];
        size_t workSize = workspace.shape()[0];
        hx = AsContiguous(hx.AsType(Dtype::kFloat32));
        dhy = AsContiguous(dhy.AsType(Dtype::kFloat32));

        cuda_internal::CudnnTensorDescriptor hx_desc{hx};
        Array _cx;
        Array _dcy;
        if (cx.has_value()) {
            _cx = AsContiguous((*cx).AsType(Dtype::kFloat32));
            _dcy = AsContiguous((*dcy).AsType(Dtype::kFloat32));
        } else {
            _cx = Zeros(hx.shape(), hx.dtype(), hx.device());
            _dcy = Zeros(hx.shape(), hx.dtype(), hx.device());
        }
        cuda_internal::CudnnTensorDescriptor cx_desc{_cx};
        cuda_internal::CudnnTensorDescriptor dcy_desc{_dcy};
        cuda_internal::CudnnTensorDescriptor dhy_desc{dhy};
        Array dhx = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnTensorDescriptor dhx_desc{dhx};
        Array dcx = AsContiguous(Zeros(hx.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnTensorDescriptor dcx_desc{dcx};
        reserve = AsContiguous(reserve);

        device_internals.cudnn_handle().Call(
                cudnnRNNBackwardData,
                rnn_desc,
                xs.size(),
                ys_desc,
                internal::GetRawOffsetData(y),
                dys_desc,
                internal::GetRawOffsetData(dy),
                *dhy_desc,
                internal::GetRawOffsetData(dhy),
                *dcy_desc,
                internal::GetRawOffsetData(_dcy),
                wDesc,
                internal::GetRawOffsetData(w),
                *hx_desc,
                internal::GetRawOffsetData(hx),
                *cx_desc,
                internal::GetRawOffsetData(_cx),
                dxs_desc,
                internal::GetRawOffsetData(dx),
                *dhx_desc,
                internal::GetRawOffsetData(dhx),
                *dcx_desc,
                internal::GetRawOffsetData(dcx),
                internal::GetRawOffsetData(workspace),
                workSize,
                internal::GetRawOffsetData(reserve),
                reserve_size);

        Array dw = AsContiguous(Zeros(w.shape(), hx.dtype(), hx.device()));
        cuda_internal::CudnnFilterDescriptor dwDesc{dw};
        device_internals.cudnn_handle().Call(
                cudnnRNNBackwardWeights,
                rnn_desc,
                xs.size(),
                xs_desc,
                internal::GetRawOffsetData(x),
                *hx_desc,
                internal::GetRawOffsetData(hx),
                ys_desc,
                internal::GetRawOffsetData(y),
                internal::GetRawOffsetData(workspace),
                workSize,
                *dwDesc,
                internal::GetRawOffsetData(dw),
                internal::GetRawOffsetData(reserve),
                reserve_size);

        std::vector<int64_t> split_indices;
        for (uint i = 0; i < xs.size() - 1; i++) {
            if (i != 0) {
                split_indices.push_back(split_indices[i - 1] + xs[i].shape()[0]);
            } else {
                split_indices.push_back(xs[i].shape()[0]);
            }
        }
        dx = dx.AsType(type);
        dxs = Split(dx, split_indices, 0);
        std::vector<Array> dstate;
        dstate.push_back(dhx.AsType(type));
        if (cx.has_value()) {
            dstate.push_back(dcx.AsType(type));
        }
        std::vector<std::vector<Array>> ret;
        ret.push_back(dstate);
        ret.push_back(weights_backward(device, rnn_desc, dxs_desc[0], *dwDesc, dw, ws, bs, n_layers, num_directions, type));
        ret.push_back(dxs);
        return ret;
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RnnBackwardKernel, CudaRnnBackwardKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
