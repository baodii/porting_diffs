// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;
#define MAX_CAP 4
#define MAX_SEQ 2048

inline __device__ float gelu(const float x) {
  const float sqrt_param = 0.79788456080286535587989211986876f;
  const float mul_param = 0.044715;
  return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

/*
In-place gelu(biasAdd(x)) for channels last
*/
template <typename T>
__global__ void fused_bias_gelu(T *input, const T *bias, int total_count,
                                int intermediate_size) {
  // Input restriction: intermediate_size % vals_per_access == 0
  constexpr int granularity = 16;
  constexpr int values_per_access = granularity / sizeof(T);
  const int offset =
      (blockIdx.x * blockDim.x + threadIdx.x) * values_per_access;

  if (offset < total_count) {
    T data[values_per_access];
    T data_bias[values_per_access];
    mem_access::load_global<granularity>(data, input + offset);
    mem_access::load_global<granularity>(data_bias,
                                         bias + (offset % intermediate_size));

#pragma unroll
    for (int i = 0; i < values_per_access; i++) {
      float data_f = conversion::to<float>(data[i]);
      float bias_f = conversion::to<float>(data_bias[i]);
      data[i] = conversion::to<T>(gelu(data_f + bias_f));
    }

    mem_access::store_global<granularity>(input + offset, data);
  }
}

template <typename T>
void launch_bias_gelu(T *input, const T *bias, int intermediate_size,
                      int batch_size, cudaStream_t stream) {
  constexpr int threads = 1024;
  constexpr int granularity = 16;

  const int total_count = batch_size * intermediate_size;
  const int elems_per_block = threads * (granularity / sizeof(T));
  dim3 block_dims(threads);
  dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

  fused_bias_gelu<<<grid_dims, block_dims, 0, stream>>>(
      input, bias, total_count, intermediate_size);
}

template void launch_bias_gelu<float>(float *, const float *, int, int,
                                      cudaStream_t);
#ifdef BF16_AVAILABLE
template void launch_bias_gelu<__nv_bfloat16>(__nv_bfloat16 *,
                                              const __nv_bfloat16 *, int, int,
                                              cudaStream_t);
#endif
template void launch_bias_gelu<__half>(__half *, const __half *, int, int,
                                       cudaStream_t);

/*
In-place channels-last bias add
*/
template <typename T>
__global__ void fused_bias_add(T *input, const T *bias, int total_count,
                               int intermediate_size) {
  // Input restriction: intermediate_size % vals_per_access == 0
  constexpr int granularity = 16;
  constexpr int values_per_access = granularity / sizeof(T);
  const int offset =
      (blockIdx.x * blockDim.x + threadIdx.x) * values_per_access;

  if (offset < total_count) {
    T data[values_per_access];
    T data_bias[values_per_access];
    mem_access::load_global<granularity>(data, input + offset);
    mem_access::load_global<granularity>(data_bias,
                                         bias + (offset % intermediate_size));

#pragma unroll
    for (int i = 0; i < values_per_access; i++) {
      float data_f = conversion::to<float>(data[i]);
      float bias_f = conversion::to<float>(data_bias[i]);
      data[i] = conversion::to<T>(data_f + bias_f);
    }

    mem_access::store_global<granularity>(input + offset, data);
  }
}

template <typename T>
void launch_bias_add(T *input, const T *bias, int intermediate_size,
                     int batch_size, cudaStream_t stream) {
  constexpr int threads = 1024;
  constexpr int granularity = 16;

  const int total_count = batch_size * intermediate_size;
  const int elems_per_block = threads * (granularity / sizeof(T));
  dim3 block_dims(threads);
  dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

  fused_bias_add<<<grid_dims, block_dims, 0, stream>>>(input, bias, total_count,
                                                       intermediate_size);
}

template void launch_bias_add<float>(float *, const float *, int, int,
                                     cudaStream_t);
#ifdef BF16_AVAILABLE
template void launch_bias_add<__nv_bfloat16>(__nv_bfloat16 *,
                                             const __nv_bfloat16 *, int, int,
                                             cudaStream_t);
#endif
template void launch_bias_add<__half>(__half *, const __half *, int, int,
                                      cudaStream_t);

__global__ void fused_bias_residual(float *residual, const float *hidden_state,
                                    const float *attn, const float *bias,
                                    const float *attn_bias,
                                    const int total_count,
                                    const int intermediate_size,
                                    const float mp_scale, const bool preln) {
  float4 *res_fl4_ptr = reinterpret_cast<float4 *>(residual);
  const float4 *hs_fl4_ptr = reinterpret_cast<const float4 *>(hidden_state);
  const float4 *attn_fl4_ptr = reinterpret_cast<const float4 *>(attn);
  const float4 *bias_fl4_ptr = reinterpret_cast<const float4 *>(bias);
  const float4 *attn_bias_fl4_ptr = reinterpret_cast<const float4 *>(attn_bias);
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < total_count) {
    float4 res_fl4 = res_fl4_ptr[offset];
    const float4 hs_fl4 = hs_fl4_ptr[offset];
    const float4 attn_fl4 = attn_fl4_ptr[offset];
    const float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];
    const float4 attn_bias_fl4 = attn_bias_fl4_ptr[offset % intermediate_size];
    if (preln) {
      // residual = (residual + attention + bias + attention_bias) *
      // mp_scale + hidden_state
      res_fl4.x =
          (res_fl4.x + attn_fl4.x + bias_fl4.x + attn_bias_fl4.x) * mp_scale +
          (hs_fl4.x);
      res_fl4.y =
          (res_fl4.y + attn_fl4.y + bias_fl4.y + attn_bias_fl4.y) * mp_scale +
          (hs_fl4.y);
      res_fl4.z =
          (res_fl4.z + attn_fl4.z + bias_fl4.z + attn_bias_fl4.z) * mp_scale +
          (hs_fl4.z);
      res_fl4.w =
          (res_fl4.w + attn_fl4.w + bias_fl4.w + attn_bias_fl4.w) * mp_scale +
          (hs_fl4.w);
    } else {
      // residual += hidden_state + bias
      res_fl4.x = res_fl4.x + hs_fl4.x + bias_fl4.x;
      res_fl4.y = res_fl4.y + hs_fl4.y + bias_fl4.y;
      res_fl4.z = res_fl4.z + hs_fl4.z + bias_fl4.z;
      res_fl4.w = res_fl4.w + hs_fl4.w + bias_fl4.w;
    }
    res_fl4_ptr[offset] = res_fl4;
  }
}

template <typename T>
__global__ void fused_bias_residual(T *residual, const T *hidden_state,
                                    const T *attn, const T *bias,
                                    const T *attn_bias, const int total_count,
                                    const int intermediate_size,
                                    const float mp_scale, const bool preln) {
  using T2 = typename std::conditional<std::is_same<T, __half>::value, __half2,
                                       __nv_bfloat162>::type;
  float2 *res_fl2_ptr = reinterpret_cast<float2 *>(residual);
  const float2 *hs_fl2_ptr = reinterpret_cast<const float2 *>(hidden_state);
  const float2 *attn_fl2_ptr = reinterpret_cast<const float2 *>(attn);
  const float2 *bias_fl2_ptr = reinterpret_cast<const float2 *>(bias);
  const float2 *attn_bias_fl2_ptr = reinterpret_cast<const float2 *>(attn_bias);
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < total_count) {
    float2 res_fl2 = res_fl2_ptr[offset];
    const float2 hs_fl2 = hs_fl2_ptr[offset];
    const float2 attn_fl2 = attn_fl2_ptr[offset];
    const float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];
    const float2 attn_bias_fl2 = attn_bias_fl2_ptr[offset % intermediate_size];

    T2 *res_half2 = reinterpret_cast<T2 *>(&res_fl2);
    const T2 *hs_half2 = reinterpret_cast<const T2 *>(&hs_fl2);
    const T2 *attn_half2 = reinterpret_cast<const T2 *>(&attn_fl2);
    const T2 *bias_half2 = reinterpret_cast<const T2 *>(&bias_fl2);
    const T2 *attn_bias_half2 = reinterpret_cast<const T2 *>(&attn_bias_fl2);

    float2 res_low = conversion::to<float2>(res_half2[0]);
    float2 res_high = conversion::to<float2>(res_half2[1]);

    const float2 hs_low = conversion::to<float2>(hs_half2[0]);
    const float2 hs_high = conversion::to<float2>(hs_half2[1]);

    const float2 attn_low = conversion::to<float2>(attn_half2[0]);
    const float2 attn_high = conversion::to<float2>(attn_half2[1]);

    const float2 bias_low = conversion::to<float2>(bias_half2[0]);
    const float2 bias_high = conversion::to<float2>(bias_half2[1]);

    const float2 attn_bias_low = conversion::to<float2>(attn_bias_half2[0]);
    const float2 attn_bias_high = conversion::to<float2>(attn_bias_half2[1]);

    if (preln) {
      // residual = (residual + attention + bias + attention_bias) *
      // mp_scale + hidden_state
      res_low.x =
          (res_low.x + attn_low.x + bias_low.x + attn_bias_low.x) * mp_scale +
          hs_low.x;
      res_low.y =
          (res_low.y + attn_low.y + bias_low.y + attn_bias_low.y) * mp_scale +
          hs_low.y;
      res_high.x = (res_high.x + attn_high.x + bias_high.x + attn_bias_high.x) *
                       mp_scale +
                   hs_high.x;
      res_high.y = (res_high.y + attn_high.y + bias_high.y + attn_bias_high.y) *
                       mp_scale +
                   hs_high.y;
    } else {
      // residual += hidden_state + bias
      res_low.x = (res_low.x + hs_low.x + bias_low.x);
      res_low.y = (res_low.y + hs_low.y + bias_low.y);
      res_high.x = (res_high.x + hs_high.x + bias_high.x);
      res_high.y = (res_high.y + hs_high.y + bias_high.y);
    }
    res_half2[0] = conversion::to<T2>(res_low);
    res_half2[1] = conversion::to<T2>(res_high);

    res_fl2_ptr[offset] = res_fl2;
  }
}

template <typename T>
void launch_bias_residual(T *residual, T *hidden_state, T *attn, T *bias,
                          T *attn_bias, int batch, int hidden_dim, int mp_size,
                          bool preln, cudaStream_t stream) {
  int total_count = batch * hidden_dim / 4;
  dim3 block_dims(1024);
  dim3 grid_dims((total_count - 1) / 1024 + 1); // (batch_size);

  fused_bias_residual<<<grid_dims, block_dims, 0, stream>>>(
      residual, hidden_state, attn, bias, attn_bias, total_count,
      hidden_dim / 4, 1.0 / mp_size, preln);
}

template void launch_bias_residual<float>(float *, float *, float *, float *,
                                          float *, int, int, int, bool,
                                          cudaStream_t);
#ifdef BF16_AVAILABLE
template void launch_bias_residual<__nv_bfloat16>(
    __nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *,
    __nv_bfloat16 *, int, int, int, bool, cudaStream_t);
#endif
template void launch_bias_residual<__half>(__half *, __half *, __half *,
                                           __half *, __half *, int, int, int,
                                           bool, cudaStream_t);

#ifdef BF16_AVAILABLE
template __global__ void
fused_bias_residual(__nv_bfloat16 *residual, const __nv_bfloat16 *hidden_state,
                    const __nv_bfloat16 *attn, const __nv_bfloat16 *bias,
                    const __nv_bfloat16 *attn_bias, const int total_count,
                    const int intermediate_size, const float mp_scale,
                    const bool preln);
#endif

template __global__ void fused_bias_residual(
    __half *residual, const __half *hidden_state, const __half *attn,
    const __half *bias, const __half *attn_bias, const int total_count,
    const int intermediate_size, const float mp_scale, const bool preln);

__global__ void gptj_residual_add(float *residual, const float *hidden_state,
                                  const float *attn, const float *bias,
                                  const float *attn_bias, const int total_count,
                                  const int intermediate_size,
                                  const float mp_scale) {
  float4 *res_fl4_ptr = reinterpret_cast<float4 *>(residual);
  const float4 *hs_fl4_ptr = reinterpret_cast<const float4 *>(hidden_state);
  const float4 *attn_fl4_ptr = reinterpret_cast<const float4 *>(attn);
  const float4 *bias_fl4_ptr = reinterpret_cast<const float4 *>(bias);
  const float4 *attn_bias_fl4_ptr = reinterpret_cast<const float4 *>(attn_bias);
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (offset < total_count) {
    float4 res_fl4 = res_fl4_ptr[offset];
    const float4 hs_fl4 = hs_fl4_ptr[offset];
    const float4 attn_fl4 = attn_fl4_ptr[offset];
    const float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];

    if (attn_bias) {
      float4 attn_bias_fl4 = attn_bias_fl4_ptr[offset % intermediate_size];
      // residual += attention_bias
      res_fl4.x += attn_bias_fl4.x;
      res_fl4.y += attn_bias_fl4.y;
      res_fl4.z += attn_bias_fl4.z;
      res_fl4.w += attn_bias_fl4.w;
    }
    // residual = hidden_state + attention + (residual + bias) * mp_scale
    res_fl4.x = hs_fl4.x + attn_fl4.x + (res_fl4.x + bias_fl4.x) * mp_scale;
    res_fl4.y = hs_fl4.y + attn_fl4.y + (res_fl4.y + bias_fl4.y) * mp_scale;
    res_fl4.z = hs_fl4.z + attn_fl4.z + (res_fl4.z + bias_fl4.z) * mp_scale;
    res_fl4.w = hs_fl4.w + attn_fl4.w + (res_fl4.w + bias_fl4.w) * mp_scale;

    res_fl4_ptr[offset] = res_fl4;
  }
}