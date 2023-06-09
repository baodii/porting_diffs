// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversion_utils.h"
#include "inference_cuda_layers.h"

#ifndef __HIP_PLATFORM_HCC__
#endif

template <typename T>
void apply_rotary_pos_emb(T* mixed_query,
                                     T* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned num_heads,
                                     unsigned head_size,
                                     unsigned total_count,
                                     int max_out_tokens,
                                     const sycl::nd_item<3> &item_ct1)
{
    auto b = item_ct1.get_group();
    sycl::sub_group g = item_ct1.get_sub_group();

    int id = item_ct1.get_local_id(2);
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = item_ct1.get_group(2) * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned seq_index = head_id % seq_len;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / sycl::pow<float>(10000.0, inv_freq) * (float)seq_id;
            float q = conversion::to<float>(mixed_query[offset + lane]);
            float k = conversion::to<float>(key_layer[k_offset + lane]);
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = sycl::permute_group_by_xor(item_ct1.get_sub_group(), q_rot, 1);
            k_rot = sycl::permute_group_by_xor(item_ct1.get_sub_group(), k_rot, 1);
            q = q * sycl::cos(inv_freq) + q_rot * sycl::sin(inv_freq);
            k = k * sycl::cos(inv_freq) + k_rot * sycl::sin(inv_freq);

            mixed_query[offset + lane] = conversion::to<T>(q);
            key_layer[k_offset + lane] = conversion::to<T>(k);

            lane += WARP_SIZE;
        }
    }
}

void apply_rotary_pos_emb1(float* mixed_query,
                                      float* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      int max_out_tokens,
                                      const sycl::nd_item<3> &item_ct1)
{
    auto b = item_ct1.get_group();
    sycl::sub_group g = item_ct1.get_sub_group();

    int id = item_ct1.get_local_id(2);
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = item_ct1.get_group(2) * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned seq_index = head_id % seq_len;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / sycl::pow<float>(10000.0, inv_freq) * (float)seq_id;
            float q = mixed_query[offset + lane];
            float k = key_layer[k_offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = sycl::permute_group_by_xor(item_ct1.get_sub_group(), q_rot, 1);
            k_rot = sycl::permute_group_by_xor(item_ct1.get_sub_group(), k_rot, 1);
            q = q * sycl::cos(inv_freq) + q_rot * sycl::sin(inv_freq);
            k = k * sycl::cos(inv_freq) + k_rot * sycl::sin(inv_freq);

            mixed_query[offset + lane] = q;
            key_layer[k_offset + lane] = k;

            lane += WARP_SIZE;
        }
    }
}

template <typename T>
/*
DPCT1110:1: The total declared local variable size in device function apply_rotary_pos_emb1 exceeds
128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to avoid high register
pressure.
*/
void apply_rotary_pos_emb1(T* mixed_query,
                           T* key_layer,
                           unsigned rotary_dim,
                           unsigned seq_len,
                           unsigned seq_offset,
                           unsigned num_heads,
                           unsigned head_size,
                           unsigned total_count,
                           int max_out_tokens,
                           const sycl::nd_item<3>& item_ct1)
{
    auto b = item_ct1.get_group();
    sycl::sub_group g = item_ct1.get_sub_group();

    int id = item_ct1.get_local_id(2);
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = item_ct1.get_group(2) * MAX_WARP_NUM + gid;
    unsigned seq_index = head_id % seq_len;
    unsigned offset = head_id * head_size;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    constexpr unsigned mask[32] = {
        0x1 | 0x1000,     0x2 | 0x2000,     0x4 | 0x4000,     0x8 | 0x8000,     0x10 | 0x10000,
        0x20 | 0x20000,   0x40 | 0x40000,   0x80 | 0x80000,   0x100 | 0x100000, 0x200 | 0x200000,
        0x400 | 0x400000, 0x800 | 0x800000, 0x1000 | 0x1,     0x2000 | 0x2,     0x4000 | 0x4,
        0x8000 | 0x8,     0x10000 | 0x10,   0x20000 | 0x20,   0x40000 | 0x40,   0x80000 | 0x80,
        0x100000 | 0x100, 0x200000 | 0x200, 0x400000 | 0x400, 0x800000 | 0x800, 0x1000000,
        0x2000000,        0x4000000,        0x8000000,        0x10000000,       0x20000000,
        0x40000000,       0x80000000};

    unsigned seq_id = (head_id % seq_len) + seq_offset;
    unsigned half_dim = rotary_dim >> 1;
    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane % half_dim) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / sycl::pow<float>(10000.0, inv_freq) * (float)seq_id;
            float q = conversion::to<float>(mixed_query[offset + lane]);
            float k = conversion::to<float>(key_layer[k_offset + lane]);
            float rotary_sign = (lane > (half_dim - 1) ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            /*
            DPCT1023:2: The SYCL sub-group does not support mask options for
            dpct::select_from_sub_group. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the dpct experimental
            helper function to migrate __shfl_sync.
            */
            /*
            DPCT1096:9: The right-most dimension of the work-group used in the SYCL kernel that
            calls this function may be less than "32". The function "dpct::select_from_sub_group"
            may return an unexpected result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of "32".
            */
            auto q_rot_tmp =
                lane < half_dim
                    ? dpct::select_from_sub_group(item_ct1.get_sub_group(), q_rot, lane + half_dim)
                    /*
                    DPCT1023:3: The SYCL sub-group does not support mask options for
                    dpct::select_from_sub_group. You can specify
                    "--use-experimental-features=masked-sub-group-operation" to use the dpct
                    experimental helper function to migrate __shfl_sync.
                    */
                    /*
                    DPCT1096:10: The right-most dimension of the work-group used in the SYCL kernel
                    that calls this function may be less than "32". The function
                    "dpct::select_from_sub_group" may return an unexpected result on the CPU device.
                    Modify the size of the work-group to ensure that the value of the right-most
                    dimension is a multiple of "32".
                    */
                    : dpct::select_from_sub_group(item_ct1.get_sub_group(), q_rot, lane - half_dim);
            /*
            DPCT1023:4: The SYCL sub-group does not support mask options for
            dpct::select_from_sub_group. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the dpct experimental
            helper function to migrate __shfl_sync.
            */
            /*
            DPCT1096:11: The right-most dimension of the work-group used in the SYCL kernel that
            calls this function may be less than "32". The function "dpct::select_from_sub_group"
            may return an unexpected result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of "32".
            */
            auto k_rot_tmp =
                lane < half_dim
                    ? dpct::select_from_sub_group(item_ct1.get_sub_group(), k_rot, lane + half_dim)
                    /*
                    DPCT1023:5: The SYCL sub-group does not support mask options for
                    dpct::select_from_sub_group. You can specify
                    "--use-experimental-features=masked-sub-group-operation" to use the dpct
                    experimental helper function to migrate __shfl_sync.
                    */
                    /*
                    DPCT1096:12: The right-most dimension of the work-group used in the SYCL kernel
                    that calls this function may be less than "32". The function
                    "dpct::select_from_sub_group" may return an unexpected result on the CPU device.
                    Modify the size of the work-group to ensure that the value of the right-most
                    dimension is a multiple of "32".
                    */
                    : dpct::select_from_sub_group(item_ct1.get_sub_group(), k_rot, lane - half_dim);
            q = q * sycl::cos(inv_freq) + q_rot_tmp * sycl::sin(inv_freq);
            k = k * sycl::cos(inv_freq) + k_rot_tmp * sycl::sin(inv_freq);

            mixed_query[offset + lane] = conversion::to<T>(q);
            key_layer[k_offset + lane] = conversion::to<T>(k);

            lane += WARP_SIZE;
        }
    }
}

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 bool rotate_half,
                                 bool rotate_every_two,
                                 dpct::queue_ptr stream,
                                 int max_out_tokens)
{
    int total_count = batch * num_heads * seq_len;
    sycl::range<3> block_dims(1, 1, 1024);
    sycl::range<3> grid_dims(1, 1, (total_count - 1) / MAX_WARP_NUM + 1);  // (batch_size);
    if (rotate_every_two)
        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the limit. To get the
        device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64, sycl::aspect::fp16});
        stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                                 apply_rotary_pos_emb(mixed_query,
                                                      key_layer,
                                                      rotary_dim,
                                                      seq_len,
                                                      offset,
                                                      num_heads,
                                                      head_size,
                                                      total_count,
                                                      max_out_tokens,
                                                      item_ct1);
                             });
    } else if (rotate_half)
        /*
        DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the limit. To get the
        device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64, sycl::aspect::fp16});
        stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                                 apply_rotary_pos_emb1(mixed_query,
                                                       key_layer,
                                                       rotary_dim,
                                                       seq_len,
                                                       offset,
                                                       num_heads,
                                                       head_size,
                                                       total_count,
                                                       max_out_tokens,
                                                       item_ct1);
                             });
    }
}

template void launch_apply_rotary_pos_emb<float>(float*,
                                                 float*,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 bool,
                                                 bool,
                                                 dpct::queue_ptr,
                                                 int);
#ifdef BF16_AVAILABLE
template void launch_apply_rotary_pos_emb<__nv_bfloat16>(__nv_bfloat16*,
                                                         __nv_bfloat16*,
                                                         unsigned,
                                                         unsigned,
                                                         unsigned,
                                                         unsigned,
                                                         unsigned,
                                                         unsigned,
                                                         bool,
                                                         bool,
                                                         cudaStream_t,
                                                         int);
#endif
template void launch_apply_rotary_pos_emb<sycl::half>(sycl::half*,
                                                      sycl::half*,
                                                      unsigned,
                                                      unsigned,
                                                      unsigned,
                                                      unsigned,
                                                      unsigned,
                                                      unsigned,
                                                      bool,
                                                      bool,
                                                      dpct::queue_ptr,
                                                      int);

template void apply_rotary_pos_emb(float* mixed_query,
                                              float* key_layer,
                                              unsigned rotary_dim,
                                              unsigned seq_len,
                                              unsigned seq_offset,
                                              unsigned num_heads,
                                              unsigned head_size,
                                              unsigned total_count,
                                              int max_out_tokens,
                                              const sycl::nd_item<3> &item_ct1);

#ifdef BF16_AVAILABLE
template __global__ void apply_rotary_pos_emb(__nv_bfloat16* mixed_query,
                                              __nv_bfloat16* key_layer,
                                              unsigned rotary_dim,
                                              unsigned seq_len,
                                              unsigned seq_offset,
                                              unsigned num_heads,
                                              unsigned head_size,
                                              unsigned total_count,
                                              int max_out_tokens);
#endif

template void apply_rotary_pos_emb(sycl::half* mixed_query,
                                   sycl::half* key_layer,
                                   unsigned rotary_dim,
                                   unsigned seq_len,
                                   unsigned seq_offset,
                                   unsigned num_heads,
                                   unsigned head_size,
                                   unsigned total_count,
                                   int max_out_tokens,
                                   const sycl::nd_item<3>& item_ct1);

#ifdef BF16_AVAILABLE
template __global__ void apply_rotary_pos_emb1(__nv_bfloat16* mixed_query,
                                               __nv_bfloat16* key_layer,
                                               unsigned rotary_dim,
                                               unsigned seq_len,
                                               unsigned seq_offset,
                                               unsigned num_heads,
                                               unsigned head_size,
                                               unsigned total_count,
                                               int max_out_tokens);
#endif

template void apply_rotary_pos_emb1(sycl::half* mixed_query,
                                    sycl::half* key_layer,
                                    unsigned rotary_dim,
                                    unsigned seq_len,
                                    unsigned seq_offset,
                                    unsigned num_heads,
                                    unsigned head_size,
                                    unsigned total_count,
                                    int max_out_tokens,
                                    const sycl::nd_item<3>& item_ct1);
/*
__global__ void apply_rotary_pos_emb(float* mixed_query,
float* key_layer,
unsigned rotary_dim,
unsigned seq_len,
unsigned seq_offset,
unsigned num_heads,
unsigned head_size,
unsigned total_count)
{
cg::thread_block b = cg::this_thread_block();
cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

int id = threadIdx.x;
int gid = id >> 5;
int lane = id & 0x1f;

unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
unsigned offset = head_id * head_size;

unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;

if (head_id < total_count) {
while (lane < rotary_dim) {
float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
float q = mixed_query[offset + lane];
float k = key_layer[offset + lane];
float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
float q_rot = (q * rotary_sign);
float k_rot = (k * rotary_sign);
q_rot = g.shfl_xor(q_rot, 1);
k_rot = g.shfl_xor(k_rot, 1);
q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

mixed_query[offset + lane] = q;
key_layer[offset + lane] = k;

lane += WARP_SIZE;
}
}
}

__global__ void apply_rotary_pos_emb(__half* mixed_query,
__half* key_layer,
unsigned rotary_dim,
unsigned seq_len,
unsigned seq_offset,
unsigned num_heads,
unsigned head_size,
unsigned total_count)
{
#if __CUDA_ARCH__ >= 700
cg::thread_block b = cg::this_thread_block();
cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

int id = threadIdx.x;
int gid = id >> 5;
int lane = id & 0x1f;

unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
unsigned offset = head_id * head_size;
constexpr unsigned mask[32] = {0x1 | 0x1000, 0x2 | 0x2000, 0x4 | 0x4000, 0x8 | 0x8000,
0x10 | 0x10000, 0x20 | 0x20000, 0x40 | 0x40000, 0x80 | 0x80000,
0x100 | 0x100000, 0x200 | 0x200000, 0x400 | 0x400000, 0x800 | 0x800000,
0x1000 | 0x1, 0x2000 | 0x2, 0x4000 | 0x4, 0x8000 | 0x8,
0x10000 | 0x10, 0x20000 | 0x20, 0x40000 | 0x40, 0x80000 | 0x80,
0x100000 | 0x100, 0x200000 | 0x200, 0x400000 | 0x400, 0x800000 | 0x800,
0x1000000, 0x2000000, 0x4000000, 0x8000000,
0x10000000, 0x20000000, 0x40000000, 0x80000000};
unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;

if (head_id < total_count) {
while (lane < rotary_dim) {
//float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
float inv_freq = (float)((lane % (rotary_dim >> 1)) * 2) / (float)rotary_dim;
inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
float q = (float)mixed_query[offset + lane];
float k = (float)key_layer[offset + lane];
float rotary_sign = (lane > 11 ? -1.0 : 1.0);
float q_rot = (q * rotary_sign);
float k_rot = (k * rotary_sign);
auto q_rot_tmp = lane < 12 ? __shfl_sync(mask[lane], q_rot, lane + 12) : __shfl_sync(mask[lane],
q_rot, lane - 12);//g.shfl_xor(q_rot, 12); auto k_rot_tmp = lane < 12 ? __shfl_sync(mask[lane],
k_rot, lane + 12) : __shfl_sync(mask[lane], k_rot, lane - 12);//g.shfl_xor(k_rot, 12); q = q *
cosf(inv_freq) + q_rot_tmp * sinf(inv_freq); k = k * cosf(inv_freq) + k_rot_tmp * sinf(inv_freq);

mixed_query[offset + lane] = (__half)q;
key_layer[offset + lane] = (__half)k;

lane += WARP_SIZE;
}
}
#endif
}

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
T* key_layer,
unsigned head_size,
unsigned seq_len,
unsigned rotary_dim,
unsigned offset,
unsigned num_heads,
unsigned batch,
cudaStream_t stream)
{
int total_count = batch * num_heads * seq_len;
dim3 block_dims(1024);
dim3 grid_dims((total_count - 1) / MAX_WARP_NUM + 1);  // (batch_size);

apply_rotary_pos_emb<<<grid_dims, block_dims, 0, stream>>>(
mixed_query, key_layer, rotary_dim, seq_len, offset, num_heads, head_size, total_count);
}

template void launch_apply_rotary_pos_emb<float>(float*,
float*,
unsigned,
unsigned,
unsigned,
unsigned,
unsigned,
unsigned,
cudaStream_t);
template void launch_apply_rotary_pos_emb<__half>(__half*,
__half*,
unsigned,
unsigned,
unsigned,
unsigned,
unsigned,
unsigned,
cudaStream_t);
*/
