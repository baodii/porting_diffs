// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds_kernel_utils.h"

#include <stdint.h>

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

namespace conversion {

// Basic primitive for constructing conversions
template <typename TO, typename FROM>
DS_D_INLINE TO to(FROM val)
{
    return to(val);
}

// Specializations

/********************* Identity Conversions *********************/
/*
Identity conversions are useful in templated functions where we might have
a fixed destination type. For example, I might have a kernel that accepts
__half, __nv_bfloat16, and float but always want to do the core computation
at floating point:

T mem_value = input[idx];
float compute_value = conversion::to<float, T>(mem_value);

In practice, we should be able to elide the second template parameter:
float compute_val = conversion::to<float>(mem_value);

In this case, we need an implementation to handle the T = float case

NOTE: The type inferencing system appears to be unable to handle inferring the first
template parameter, even in the trivial case.
*/

// Floating point types
template <>
DS_D_INLINE double to(double val)
{
    return val;
}
template <>
DS_D_INLINE float to(float val)
{
    return val;
}
template <>
DS_D_INLINE sycl::half to(sycl::half val)
{
    return val;
}
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE __nv_bfloat16 to(__nv_bfloat16 val)
{
    return val;
}
#endif

// Integer types
template <>
DS_D_INLINE int8_t to(int8_t val)
{
    return val;
}
template <>
DS_D_INLINE uint8_t to(uint8_t val)
{
    return val;
}
template <>
DS_D_INLINE int16_t to(int16_t val)
{
    return val;
}
template <>
DS_D_INLINE uint16_t to(uint16_t val)
{
    return val;
}
template <>
DS_D_INLINE int32_t to(int32_t val)
{
    return val;
}
template <>
DS_D_INLINE uint32_t to(uint32_t val)
{
    return val;
}
template <>
DS_D_INLINE int64_t to(int64_t val)
{
    return val;
}
template <>
DS_D_INLINE uint64_t to(uint64_t val)
{
    return val;
}

// TODO: evaluate if we want bools

/*********************  To Double Conversions *********************/

// * to double variants

// Would normally like to not use C cast, but this is an important enough conversion
// to keep
template <>
DS_D_INLINE double to(float val)
{
#ifdef PTX_AVAILABLE
    double ret_val;
    /*
    DPCT1053:0: Migration of device assembly code is not supported.
    */
    asm("ctv.rn.f64.f32 %0, %1;\n" : "=d"(ret_val) : "f"(val));
    return ret_val;
#else
    return double(val);
#endif
}
// Note: there is a CVT instruction for __half -> double, but there's no inline interface
// for passing a single half value
template <>
DS_D_INLINE double to(sycl::half val)
{
    return to<double>(
        sycl::vec<sycl::half, 1>{val}.convert<float, sycl::rounding_mode::automatic>()[0]);
}
template <>
DS_D_INLINE double to(int64_t val)
{
    return sycl::vec<long long, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(int32_t val)
{
    return sycl::vec<int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(int16_t val)
{
    return sycl::vec<int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(int8_t val)
{
    return sycl::vec<int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint64_t val)
{
    return sycl::vec<unsigned long long, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint32_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint16_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE double to(uint8_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<double, sycl::rounding_mode::rte>()[0];
}

// Same applies here
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE double to(__nv_bfloat16 val)
{
    return to<double>(__bfloat162float(val));
}
#endif

/*********************  To Float Conversions *********************/

template <>
DS_D_INLINE float to(double val)
{
    return sycl::vec<double, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<float, sycl::rounding_mode::automatic>()[0];
}
template <>
DS_D_INLINE float to(int64_t val)
{
    return sycl::vec<long long, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(int32_t val)
{
    return sycl::vec<int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(int16_t val)
{
    return sycl::vec<int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(int8_t val)
{
    return sycl::vec<int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint64_t val)
{
    return sycl::vec<unsigned long long, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint32_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint16_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE float to(uint8_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<float, sycl::rounding_mode::rte>()[0];
}

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE float to(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}
#endif

/*********************  To Float2 Conversions *********************/
template <>
DS_D_INLINE sycl::float2 to(sycl::half2 val)
{
    return val.convert<float, sycl::rounding_mode::automatic>();
}

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE float2 to(__nv_bfloat162 val)
{
    return __bfloat1622float2(val);
}
#endif

/*********************  To Half Conversions *********************/
template <>
DS_D_INLINE sycl::half to(double val)
{
#ifdef __HIP_PLATFORM_HCC__
    float val_f = __double2float_rn(val);
    return __float2half(val_f);
#else
    /*
    DPCT1007:50: Migration of __double2half is not supported.
    */
    return __double2half(val);
#endif
}
template <>
DS_D_INLINE sycl::half to(float val)
{
    return sycl::vec<float, 1>{val}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}
template <>
DS_D_INLINE sycl::half to(int64_t val)
{
    return sycl::vec<long long, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(int32_t val)
{
    return sycl::vec<int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(int16_t val)
{
    return sycl::vec<short, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(int8_t val)
{
    return sycl::vec<int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint64_t val)
{
    return sycl::vec<unsigned long long, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint32_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint16_t val)
{
    return sycl::vec<unsigned short, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE sycl::half to(uint8_t val)
{
    return sycl::vec<unsigned int, 1>{val}.convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
DS_D_INLINE __half to(__nv_bfloat16 val)
{
    return to<__half>(to<float>(val));
}
#endif

/*********************  To Half2 Conversions *********************/
template <>
DS_D_INLINE sycl::half2 to(sycl::float2 val)
{
    return val.convert<sycl::half, sycl::rounding_mode::rte>();
}
template <>
DS_D_INLINE sycl::half2 to(float val)
{
    return sycl::float2{val, val}.convert<sycl::half, sycl::rounding_mode::rte>();
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
DS_D_INLINE __half2 to(__nv_bfloat162 val)
{
    return to<__half2>(to<float2>(val));
}
#endif

/*********************  To BF16 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE __nv_bfloat16 to(double val)
{
    return __double2bfloat16(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(float val)
{
    return __float2bfloat16(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int64_t val)
{
    return __ll2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int32_t val)
{
    return __int2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int16_t val)
{
    return __short2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int8_t val)
{
    return __int2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint64_t val)
{
    return __ull2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint32_t val)
{
    return __uint2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint16_t val)
{
    return __ushort2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint8_t val)
{
    return __uint2bfloat16_rn(val);
}
#endif

/*********************  To BF162 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE __nv_bfloat162 to(float2 val)
{
    return __float22bfloat162_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat162 to(float val)
{
    return __float2bfloat162_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat162 to(__half2 val)
{
    return to<__nv_bfloat162>(to<float2>(val));
}
#endif

/*********************  To INT64_T Conversions *********************/
template <>
DS_D_INLINE int64_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int64_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int64_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<long long, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int64_t to(__nv_bfloat16 val)
{
    return __bfloat162ll_rn(val);
}
#endif

/*********************  To INT32_T Conversions *********************/
template <>
DS_D_INLINE int32_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int32_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int32_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int32_t to(__nv_bfloat16 val)
{
    return __bfloat162int_rn(val);
}
#endif

/*********************  To INT16_T Conversions *********************/
template <>
DS_D_INLINE int16_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int16_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int16_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int16_t to(__nv_bfloat16 val)
{
    return __bfloat162int_rn(val);
}
#endif

/*********************  To INT8_T Conversions *********************/
template <>
DS_D_INLINE int8_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int8_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE int8_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int8_t to(__nv_bfloat16 val)
{
    return __bfloat162int_rn(val);
}
#endif

/*********************  To UINT64_T Conversions *********************/
template <>
DS_D_INLINE uint64_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint64_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint64_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint64_t to(__nv_bfloat16 val)
{
    return __bfloat162ull_rn(val);
}
#endif

/*********************  To UINT32_T Conversions *********************/
template <>
DS_D_INLINE uint32_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint32_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint32_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint32_t to(__nv_bfloat16 val)
{
    return __bfloat162uint_rn(val);
}
#endif

/*********************  To UINT16_T Conversions *********************/
template <>
DS_D_INLINE uint16_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint16_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint16_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint16_t to(__nv_bfloat16 val)
{
    return __bfloat162uint_rn(val);
}
#endif

/*********************  To UINT8_T Conversions *********************/
template <>
DS_D_INLINE uint8_t to(double val)
{
    return sycl::vec<double, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint8_t to(float val)
{
    return sycl::vec<float, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
template <>
DS_D_INLINE uint8_t to(sycl::half val)
{
    return sycl::vec<sycl::half, 1>{val}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint8_t to(__nv_bfloat16 val)
{
    return __bfloat162uint_rn(val);
}
#endif

}  // namespace conversion
