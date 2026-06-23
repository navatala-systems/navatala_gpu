// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_transformer` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_transformer_gelu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_gelu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfX = (__uint_as_float(0x3f000000u) * x);
    float result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_gelu_tanh_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_gelu_tanh_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfX = (__uint_as_float(0x3f000000u) * x);
    float result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_silu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_silu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (__uint_as_float(0x3f800000u) + expNegX);
    float result = (x / denom);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_relu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_relu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    bool isPositive = (x > __uint_as_float(0x00000000u));
    float result = ((isPositive) ? (x) : (__uint_as_float(0x00000000u)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_quick_gelu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_quick_gelu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float scaled = (__uint_as_float(0x3fd9db23u) * x);
    float negScaled = (-scaled);
    float expNegScaled = exp(negScaled);
    float denom = (__uint_as_float(0x3f800000u) + expNegScaled);
    float sigmoid = (__uint_as_float(0x3f800000u) / denom);
    float result = (x * sigmoid);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_squared_relu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_squared_relu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    bool isPositive = (x > __uint_as_float(0x00000000u));
    float relu = ((isPositive) ? (x) : (__uint_as_float(0x00000000u)));
    float result = (relu * relu);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_bias_gelu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_bias_gelu_f32(const float* _input, const float* bias, const unsigned int* count, const unsigned int* biasSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int bSize = biasSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    unsigned int biasIdx = (gid % bSize);
    float b = bias[biasIdx];
    float xBias = (x + b);
    float x2 = (xBias * xBias);
    float x3 = (x2 * xBias);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (xBias + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfXBias = (__uint_as_float(0x3f000000u) * xBias);
    float result = (halfXBias * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_gelu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_gelu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half x2 = (x * x);
    __half x3 = (x2 * x);
    __half cubeTerm = (__float2half(0.044715f) * x3);
    __half inner = (x + cubeTerm);
    __half tanhArg = (__float2half(0.797885f) * inner);
    __half tanhVal = tanh(tanhArg);
    __half onePlusTanh = (__float2half(1.000000f) + tanhVal);
    __half halfX = (__float2half(0.500000f) * x);
    __half result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_silu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_silu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half negX = (-x);
    __half expNegX = exp(negX);
    __half denom = (__float2half(1.000000f) + expNegX);
    __half result = (x / denom);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_relu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_relu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    bool isPositive = (x > __float2half(0.000000f));
    __half result = ((isPositive) ? (x) : (__float2half(0.000000f)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_quick_gelu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_quick_gelu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half scaled = (__float2half(1.702000f) * x);
    __half negScaled = (-scaled);
    __half expNegScaled = exp(negScaled);
    __half denom = (__float2half(1.000000f) + expNegScaled);
    __half sigmoid = (__float2half(1.000000f) / denom);
    __half result = (x * sigmoid);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_squared_relu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_squared_relu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    bool isPositive = (x > __float2half(0.000000f));
    __half relu = ((isPositive) ? (x) : (__float2half(0.000000f)));
    __half result = (relu * relu);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_gelu_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_gelu_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __hip_bfloat16 xBF16 = _input[gid];
    float x = ((float)(xBF16));
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfX = (__uint_as_float(0x3f000000u) * x);
    float resultF32 = (halfX * onePlusTanh);
    __hip_bfloat16 result = ((__hip_bfloat16)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_silu_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_silu_bf16(const __hip_bfloat16* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __hip_bfloat16 xBF16 = _input[gid];
    float x = ((float)(xBF16));
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (__uint_as_float(0x3f800000u) + expNegX);
    float resultF32 = (x / denom);
    __hip_bfloat16 result = ((__hip_bfloat16)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_geglu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_geglu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    unsigned int valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float gate2 = (gate * gate);
    float gate3 = (gate2 * gate);
    float cubeTerm = (__uint_as_float(0x3d372713u) * gate3);
    float inner = (gate + cubeTerm);
    float tanhArg = (__uint_as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhVal);
    float halfGate = (__uint_as_float(0x3f000000u) * gate);
    float geluGate = (halfGate * onePlusTanh);
    float result = (geluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_swiglu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_swiglu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    unsigned int valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float negGate = (-gate);
    float expNegGate = exp(negGate);
    float denom = (__uint_as_float(0x3f800000u) + expNegGate);
    float siluGate = (gate / denom);
    float result = (siluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_reglu_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_reglu_f32(const float* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    unsigned int valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    bool isPositive = (gate > __uint_as_float(0x00000000u));
    float reluGate = ((isPositive) ? (gate) : (__uint_as_float(0x00000000u)));
    float result = (reluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_geglu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_geglu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    __half gate = _input[gid];
    unsigned int valueIdx = (gid + hidden);
    __half value = _input[valueIdx];
    __half gate2 = (gate * gate);
    __half gate3 = (gate2 * gate);
    __half cubeTerm = (__float2half(0.044715f) * gate3);
    __half inner = (gate + cubeTerm);
    __half tanhArg = (__float2half(0.797885f) * inner);
    __half tanhVal = tanh(tanhArg);
    __half onePlusTanh = (__float2half(1.000000f) + tanhVal);
    __half halfGate = (__float2half(0.500000f) * gate);
    __half geluGate = (halfGate * onePlusTanh);
    __half result = (geluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_swiglu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_swiglu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    __half gate = _input[gid];
    unsigned int valueIdx = (gid + hidden);
    __half value = _input[valueIdx];
    __half negGate = (-gate);
    __half expNegGate = exp(negGate);
    __half denom = (__float2half(1.000000f) + expNegGate);
    __half siluGate = (gate / denom);
    __half result = (siluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_reglu_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_reglu_f16(const __half* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int hidden = (n / 2u);
  bool inBounds = (gid < hidden);
  if (inBounds) {
    __half gate = _input[gid];
    unsigned int valueIdx = (gid + hidden);
    __half value = _input[valueIdx];
    bool isPositive = (gate > __float2half(0.000000f));
    __half reluGate = ((isPositive) ? (gate) : (__float2half(0.000000f)));
    __half result = (reluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_gelu_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_gelu_backward_f32(const float* _input, const float* gradOutput, const unsigned int* count, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float u = (__uint_as_float(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (__uint_as_float(0x3f800000u) - tanh2);
    float threeCoef = __uint_as_float(0x3e095d4fu);
    float derivInner = (__uint_as_float(0x3f800000u) + (threeCoef * x2));
    float term2a = (__uint_as_float(0x3f000000u) * x);
    float term2b = (term2a * sech2);
    float term2c = (term2b * __uint_as_float(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhU);
    float term1 = (__uint_as_float(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dx = (dy * geluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_silu_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_silu_backward_f32(const float* _input, const float* gradOutput, const unsigned int* count, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (__uint_as_float(0x3f800000u) + expNegX);
    float sigmoid = (__uint_as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (__uint_as_float(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (x * oneMinusSigmoid);
    float bracketTerm = (__uint_as_float(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dx = (dy * siluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_relu_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_relu_backward_f32(const float* _input, const float* gradOutput, const unsigned int* count, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    bool isPositive = (x > __uint_as_float(0x00000000u));
    float dx = ((isPositive) ? (dy) : (__uint_as_float(0x00000000u)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_quick_gelu_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_quick_gelu_backward_f32(const float* _input, const float* gradOutput, const unsigned int* count, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float scaledX = (__uint_as_float(0x3fd9db23u) * x);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (__uint_as_float(0x3f800000u) + expNegScaledX);
    float sigmoid = (__uint_as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (__uint_as_float(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dx = (dy * quickGeluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_gelu_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_gelu_backward_f16(const __half* _input, const __half* gradOutput, const unsigned int* count, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float x2 = (xF32 * xF32);
    float x3 = (x2 * xF32);
    float cubeTerm = (__uint_as_float(0x3d372713u) * x3);
    float inner = (xF32 + cubeTerm);
    float u = (__uint_as_float(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (__uint_as_float(0x3f800000u) - tanh2);
    float threeCoef = __uint_as_float(0x3e095d4fu);
    float derivInner = (__uint_as_float(0x3f800000u) + (threeCoef * x2));
    float term2a = (__uint_as_float(0x3f000000u) * xF32);
    float term2b = (term2a * sech2);
    float term2c = (term2b * __uint_as_float(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (__uint_as_float(0x3f800000u) + tanhU);
    float term1 = (__uint_as_float(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dxF32 = (dyF32 * geluPrime);
    __half dx = ((__half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_silu_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_silu_backward_f16(const __half* _input, const __half* gradOutput, const unsigned int* count, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float negX = (-xF32);
    float expNegX = exp(negX);
    float denom = (__uint_as_float(0x3f800000u) + expNegX);
    float sigmoid = (__uint_as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (__uint_as_float(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (xF32 * oneMinusSigmoid);
    float bracketTerm = (__uint_as_float(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dxF32 = (dyF32 * siluPrime);
    __half dx = ((__half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_relu_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_relu_backward_f16(const __half* _input, const __half* gradOutput, const unsigned int* count, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half dy = gradOutput[gid];
    bool isPositive = (x > __float2half(0.000000f));
    __half dx = ((isPositive) ? (dy) : (__float2half(0.000000f)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_quick_gelu_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_quick_gelu_backward_f16(const __half* _input, const __half* gradOutput, const unsigned int* count, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    __half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float scaledX = (__uint_as_float(0x3fd9db23u) * xF32);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (__uint_as_float(0x3f800000u) + expNegScaledX);
    float sigmoid = (__uint_as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (__uint_as_float(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dxF32 = (dyF32 * quickGeluPrime);
    __half dx = ((__half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_forward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rms_norm_forward_f32(const float* _input, const float* gamma, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    unsigned int neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  __syncthreads();
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    unsigned int neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  __syncthreads();
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    unsigned int neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  __syncthreads();
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    unsigned int neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  __syncthreads();
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    unsigned int neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  __syncthreads();
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    unsigned int neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  __syncthreads();
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    unsigned int neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  __syncthreads();
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    unsigned int neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  __syncthreads();
  float totalSumSq = sumSq[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    float g = gamma[lid];
    float xNorm = (x / rms);
    float result = (g * xNorm);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_forward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_rms_norm_forward_f16(const __half* _input, const __half* gamma, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    unsigned int neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  __syncthreads();
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    unsigned int neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  __syncthreads();
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    unsigned int neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  __syncthreads();
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    unsigned int neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  __syncthreads();
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    unsigned int neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  __syncthreads();
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    unsigned int neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  __syncthreads();
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    unsigned int neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  __syncthreads();
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    unsigned int neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  __syncthreads();
  float totalSumSq = sumSq[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    __half gF16 = gamma[lid];
    float g = ((float)(gF16));
    float xNorm = (x / rms);
    float resultF32 = (g * xNorm);
    __half result = ((__half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_forward_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_rms_norm_forward_bf16(const __hip_bfloat16* _input, const __hip_bfloat16* gamma, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __hip_bfloat16 xBF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2bfloat16(0.000000f)));
  float x = ((float)(xBF16));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    unsigned int neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  __syncthreads();
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    unsigned int neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  __syncthreads();
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    unsigned int neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  __syncthreads();
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    unsigned int neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  __syncthreads();
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    unsigned int neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  __syncthreads();
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    unsigned int neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  __syncthreads();
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    unsigned int neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  __syncthreads();
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    unsigned int neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  __syncthreads();
  float totalSumSq = sumSq[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    __hip_bfloat16 gBF16 = gamma[lid];
    float g = ((float)(gBF16));
    float xNorm = (x / rms);
    float resultF32 = (g * xNorm);
    __hip_bfloat16 result = ((__hip_bfloat16)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_forward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_layer_norm_forward_f32(const float* _input, const float* gamma, const float* beta, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumBuf[256];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = ((hiddenValid) ? (x) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    unsigned int neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    unsigned int neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    unsigned int neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    unsigned int neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    unsigned int neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    unsigned int neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    unsigned int neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    unsigned int neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  __syncthreads();
  float totalSum = sumBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float variance = (totalSumSq / hsF32);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  if ((batchValid && hiddenValid)) {
    float g = gamma[lid];
    float b = beta[lid];
    float xNorm = (xMinusMean / std);
    float scaled = (g * xNorm);
    float result = (scaled + b);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_forward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_layer_norm_forward_f16(const __half* _input, const __half* gamma, const __half* beta, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumBuf[256];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float x = ((float)(xF16));
  sumBuf[lid] = ((hiddenValid) ? (x) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    unsigned int neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    unsigned int neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    unsigned int neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    unsigned int neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    unsigned int neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    unsigned int neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    unsigned int neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    unsigned int neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  __syncthreads();
  float totalSum = sumBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float variance = (totalSumSq / hsF32);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  if ((batchValid && hiddenValid)) {
    __half gF16 = gamma[lid];
    __half bF16 = beta[lid];
    float g = ((float)(gF16));
    float b = ((float)(bF16));
    float xNorm = (xMinusMean / std);
    float scaled = (g * xNorm);
    float resultF32 = (scaled + b);
    __half result = ((__half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_forward_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_layer_norm_forward_bf16(const __hip_bfloat16* _input, const __hip_bfloat16* gamma, const __hip_bfloat16* beta, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumBuf[256];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __hip_bfloat16 xBF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2bfloat16(0.000000f)));
  float x = ((float)(xBF16));
  sumBuf[lid] = ((hiddenValid) ? (x) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    unsigned int neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    unsigned int neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    unsigned int neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    unsigned int neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    unsigned int neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    unsigned int neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    unsigned int neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    unsigned int neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  __syncthreads();
  float totalSum = sumBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float variance = (totalSumSq / hsF32);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  if ((batchValid && hiddenValid)) {
    __hip_bfloat16 gBF16 = gamma[lid];
    __hip_bfloat16 bBF16 = beta[lid];
    float g = ((float)(gBF16));
    float b = ((float)(bBF16));
    float xNorm = (xMinusMean / std);
    float scaled = (g * xNorm);
    float resultF32 = (scaled + b);
    __hip_bfloat16 result = ((__hip_bfloat16)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_layer_norm_backward_f32(const float* _input, const float* gradOutput, const float* gamma, const float* mean, const float* rsigma, const unsigned int* batchSize, const unsigned int* hiddenSize, float* gradInput, float* gradGamma, float* gradBeta) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  __shared__ float sumDyGamma[256];
  __shared__ float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (__uint_as_float(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (__uint_as_float(0x00000000u)));
  float mu = ((batchValid) ? (mean[batchIdx]) : (__uint_as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (__uint_as_float(0x3f800000u)));
  float xMinusMean = (x - mu);
  float xHat = (xMinusMean * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGamma[lid] = ((hiddenValid) ? (dyGamma) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGamma_128 = (lid < 128u);
  if (shouldReduce_sumDyGamma_128) {
    unsigned int neighborIdx_sumDyGamma_128 = (lid + 128u);
    float myVal_sumDyGamma_128 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_128 = sumDyGamma[neighborIdx_sumDyGamma_128];
    float sumVal_sumDyGamma_128 = (myVal_sumDyGamma_128 + neighborVal_sumDyGamma_128);
    sumDyGamma[lid] = sumVal_sumDyGamma_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_64 = (lid < 64u);
  if (shouldReduce_sumDyGamma_64) {
    unsigned int neighborIdx_sumDyGamma_64 = (lid + 64u);
    float myVal_sumDyGamma_64 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_64 = sumDyGamma[neighborIdx_sumDyGamma_64];
    float sumVal_sumDyGamma_64 = (myVal_sumDyGamma_64 + neighborVal_sumDyGamma_64);
    sumDyGamma[lid] = sumVal_sumDyGamma_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_32 = (lid < 32u);
  if (shouldReduce_sumDyGamma_32) {
    unsigned int neighborIdx_sumDyGamma_32 = (lid + 32u);
    float myVal_sumDyGamma_32 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_32 = sumDyGamma[neighborIdx_sumDyGamma_32];
    float sumVal_sumDyGamma_32 = (myVal_sumDyGamma_32 + neighborVal_sumDyGamma_32);
    sumDyGamma[lid] = sumVal_sumDyGamma_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_16 = (lid < 16u);
  if (shouldReduce_sumDyGamma_16) {
    unsigned int neighborIdx_sumDyGamma_16 = (lid + 16u);
    float myVal_sumDyGamma_16 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_16 = sumDyGamma[neighborIdx_sumDyGamma_16];
    float sumVal_sumDyGamma_16 = (myVal_sumDyGamma_16 + neighborVal_sumDyGamma_16);
    sumDyGamma[lid] = sumVal_sumDyGamma_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_8 = (lid < 8u);
  if (shouldReduce_sumDyGamma_8) {
    unsigned int neighborIdx_sumDyGamma_8 = (lid + 8u);
    float myVal_sumDyGamma_8 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_8 = sumDyGamma[neighborIdx_sumDyGamma_8];
    float sumVal_sumDyGamma_8 = (myVal_sumDyGamma_8 + neighborVal_sumDyGamma_8);
    sumDyGamma[lid] = sumVal_sumDyGamma_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_4 = (lid < 4u);
  if (shouldReduce_sumDyGamma_4) {
    unsigned int neighborIdx_sumDyGamma_4 = (lid + 4u);
    float myVal_sumDyGamma_4 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_4 = sumDyGamma[neighborIdx_sumDyGamma_4];
    float sumVal_sumDyGamma_4 = (myVal_sumDyGamma_4 + neighborVal_sumDyGamma_4);
    sumDyGamma[lid] = sumVal_sumDyGamma_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_2 = (lid < 2u);
  if (shouldReduce_sumDyGamma_2) {
    unsigned int neighborIdx_sumDyGamma_2 = (lid + 2u);
    float myVal_sumDyGamma_2 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_2 = sumDyGamma[neighborIdx_sumDyGamma_2];
    float sumVal_sumDyGamma_2 = (myVal_sumDyGamma_2 + neighborVal_sumDyGamma_2);
    sumDyGamma[lid] = sumVal_sumDyGamma_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_1 = (lid < 1u);
  if (shouldReduce_sumDyGamma_1) {
    unsigned int neighborIdx_sumDyGamma_1 = (lid + 1u);
    float myVal_sumDyGamma_1 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_1 = sumDyGamma[neighborIdx_sumDyGamma_1];
    float sumVal_sumDyGamma_1 = (myVal_sumDyGamma_1 + neighborVal_sumDyGamma_1);
    sumDyGamma[lid] = sumVal_sumDyGamma_1;
  }
  __syncthreads();
  float totalDyGamma = sumDyGamma[0u];
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    unsigned int neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    unsigned int neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    unsigned int neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    unsigned int neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    unsigned int neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    unsigned int neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    unsigned int neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    unsigned int neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  __syncthreads();
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGamma = (totalDyGamma / hsF32);
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = meanDyGamma;
    float term3 = (xHat * meanDyGammaXhat);
    float inner = ((term1 - term2) - term3);
    float dx = (rs * inner);
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomicAdd(&gradGamma[lid], dyXhat);
    atomicAdd(&gradBeta[lid], dy);
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_layer_norm_backward_f16(const __half* _input, const __half* gradOutput, const __half* gamma, const float* mean, const float* rsigma, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* gradInput, float* gradGamma, float* gradBeta) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  __shared__ float sumDyGamma[256];
  __shared__ float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float x = ((float)(xF16));
  __half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (__float2half(0.000000f)));
  float dy = ((float)(dyF16));
  __half gF16 = ((hiddenValid) ? (gamma[lid]) : (__float2half(0.000000f)));
  float g = ((float)(gF16));
  float mu = ((batchValid) ? (mean[batchIdx]) : (__uint_as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (__uint_as_float(0x3f800000u)));
  float xMinusMean = (x - mu);
  float xHat = (xMinusMean * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGamma[lid] = ((hiddenValid) ? (dyGamma) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGamma_128 = (lid < 128u);
  if (shouldReduce_sumDyGamma_128) {
    unsigned int neighborIdx_sumDyGamma_128 = (lid + 128u);
    float myVal_sumDyGamma_128 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_128 = sumDyGamma[neighborIdx_sumDyGamma_128];
    float sumVal_sumDyGamma_128 = (myVal_sumDyGamma_128 + neighborVal_sumDyGamma_128);
    sumDyGamma[lid] = sumVal_sumDyGamma_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_64 = (lid < 64u);
  if (shouldReduce_sumDyGamma_64) {
    unsigned int neighborIdx_sumDyGamma_64 = (lid + 64u);
    float myVal_sumDyGamma_64 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_64 = sumDyGamma[neighborIdx_sumDyGamma_64];
    float sumVal_sumDyGamma_64 = (myVal_sumDyGamma_64 + neighborVal_sumDyGamma_64);
    sumDyGamma[lid] = sumVal_sumDyGamma_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_32 = (lid < 32u);
  if (shouldReduce_sumDyGamma_32) {
    unsigned int neighborIdx_sumDyGamma_32 = (lid + 32u);
    float myVal_sumDyGamma_32 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_32 = sumDyGamma[neighborIdx_sumDyGamma_32];
    float sumVal_sumDyGamma_32 = (myVal_sumDyGamma_32 + neighborVal_sumDyGamma_32);
    sumDyGamma[lid] = sumVal_sumDyGamma_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_16 = (lid < 16u);
  if (shouldReduce_sumDyGamma_16) {
    unsigned int neighborIdx_sumDyGamma_16 = (lid + 16u);
    float myVal_sumDyGamma_16 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_16 = sumDyGamma[neighborIdx_sumDyGamma_16];
    float sumVal_sumDyGamma_16 = (myVal_sumDyGamma_16 + neighborVal_sumDyGamma_16);
    sumDyGamma[lid] = sumVal_sumDyGamma_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_8 = (lid < 8u);
  if (shouldReduce_sumDyGamma_8) {
    unsigned int neighborIdx_sumDyGamma_8 = (lid + 8u);
    float myVal_sumDyGamma_8 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_8 = sumDyGamma[neighborIdx_sumDyGamma_8];
    float sumVal_sumDyGamma_8 = (myVal_sumDyGamma_8 + neighborVal_sumDyGamma_8);
    sumDyGamma[lid] = sumVal_sumDyGamma_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_4 = (lid < 4u);
  if (shouldReduce_sumDyGamma_4) {
    unsigned int neighborIdx_sumDyGamma_4 = (lid + 4u);
    float myVal_sumDyGamma_4 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_4 = sumDyGamma[neighborIdx_sumDyGamma_4];
    float sumVal_sumDyGamma_4 = (myVal_sumDyGamma_4 + neighborVal_sumDyGamma_4);
    sumDyGamma[lid] = sumVal_sumDyGamma_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_2 = (lid < 2u);
  if (shouldReduce_sumDyGamma_2) {
    unsigned int neighborIdx_sumDyGamma_2 = (lid + 2u);
    float myVal_sumDyGamma_2 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_2 = sumDyGamma[neighborIdx_sumDyGamma_2];
    float sumVal_sumDyGamma_2 = (myVal_sumDyGamma_2 + neighborVal_sumDyGamma_2);
    sumDyGamma[lid] = sumVal_sumDyGamma_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGamma_1 = (lid < 1u);
  if (shouldReduce_sumDyGamma_1) {
    unsigned int neighborIdx_sumDyGamma_1 = (lid + 1u);
    float myVal_sumDyGamma_1 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_1 = sumDyGamma[neighborIdx_sumDyGamma_1];
    float sumVal_sumDyGamma_1 = (myVal_sumDyGamma_1 + neighborVal_sumDyGamma_1);
    sumDyGamma[lid] = sumVal_sumDyGamma_1;
  }
  __syncthreads();
  float totalDyGamma = sumDyGamma[0u];
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    unsigned int neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    unsigned int neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    unsigned int neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    unsigned int neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    unsigned int neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    unsigned int neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    unsigned int neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    unsigned int neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  __syncthreads();
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGamma = (totalDyGamma / hsF32);
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = meanDyGamma;
    float term3 = (xHat * meanDyGammaXhat);
    float inner = ((term1 - term2) - term3);
    float dxF32 = (rs * inner);
    __half dx = ((__half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomicAdd(&gradGamma[lid], dyXhat);
    atomicAdd(&gradBeta[lid], dy);
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rms_norm_backward_f32(const float* _input, const float* gradOutput, const float* gamma, const float* rsigma, const unsigned int* batchSize, const unsigned int* hiddenSize, float* gradInput, float* gradGamma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  __shared__ float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (__uint_as_float(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (__uint_as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (__uint_as_float(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    unsigned int neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    unsigned int neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    unsigned int neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    unsigned int neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    unsigned int neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    unsigned int neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    unsigned int neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    unsigned int neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  __syncthreads();
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dx = (rs * inner);
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomicAdd(&gradGamma[lid], dyXhat);
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_rms_norm_backward_f16(const __half* _input, const __half* gradOutput, const __half* gamma, const float* rsigma, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* gradInput, float* gradGamma) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  __shared__ float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  unsigned int globalIdx = ((batchIdx * hs) + lid);
  __half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float x = ((float)(xF16));
  __half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (__float2half(0.000000f)));
  float dy = ((float)(dyF16));
  __half gF16 = ((hiddenValid) ? (gamma[lid]) : (__float2half(0.000000f)));
  float g = ((float)(gF16));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (__uint_as_float(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (__uint_as_float(0x00000000u)));
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_128 = (lid < 128u);
  if (shouldReduce_sumDyGammaXhat_128) {
    unsigned int neighborIdx_sumDyGammaXhat_128 = (lid + 128u);
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_64 = (lid < 64u);
  if (shouldReduce_sumDyGammaXhat_64) {
    unsigned int neighborIdx_sumDyGammaXhat_64 = (lid + 64u);
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_32 = (lid < 32u);
  if (shouldReduce_sumDyGammaXhat_32) {
    unsigned int neighborIdx_sumDyGammaXhat_32 = (lid + 32u);
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_16 = (lid < 16u);
  if (shouldReduce_sumDyGammaXhat_16) {
    unsigned int neighborIdx_sumDyGammaXhat_16 = (lid + 16u);
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_8 = (lid < 8u);
  if (shouldReduce_sumDyGammaXhat_8) {
    unsigned int neighborIdx_sumDyGammaXhat_8 = (lid + 8u);
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_4 = (lid < 4u);
  if (shouldReduce_sumDyGammaXhat_4) {
    unsigned int neighborIdx_sumDyGammaXhat_4 = (lid + 4u);
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_2 = (lid < 2u);
  if (shouldReduce_sumDyGammaXhat_2) {
    unsigned int neighborIdx_sumDyGammaXhat_2 = (lid + 2u);
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  __syncthreads();
  bool shouldReduce_sumDyGammaXhat_1 = (lid < 1u);
  if (shouldReduce_sumDyGammaXhat_1) {
    unsigned int neighborIdx_sumDyGammaXhat_1 = (lid + 1u);
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  __syncthreads();
  float totalDyGammaXhat = sumDyGammaXhat[0u];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dxF32 = (rs * inner);
    __half dx = ((__half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomicAdd(&gradGamma[lid], dyXhat);
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_multi_pass_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_layer_norm_multi_pass_f32(const float* _input, const float* gamma, const float* beta, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumBuf[256];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * hs);
  float partialSum = __uint_as_float(0x00000000u);
  float partialSumSq = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  unsigned int workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  __syncthreads();
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    unsigned int neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    unsigned int neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    unsigned int neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    unsigned int neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    unsigned int neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    unsigned int neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    unsigned int neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    unsigned int neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  __syncthreads();
  float totalSum = sumBuf[0u];
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float meanSq = (mean * mean);
  float avgSumSq = (totalSumSq / hsF32);
  float variance = (avgSumSq - meanSq);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float g = gamma[iterIdx];
      float b = beta[iterIdx];
      float xNorm = ((xVal2 - mean) / std);
      float scaled = (g * xNorm);
      float result = (scaled + b);
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_hip_navatala_transformer_layer_norm_multi_pass_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_layer_norm_multi_pass_f16(const __half* _input, const __half* gamma, const __half* beta, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumBuf[256];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * hs);
  float partialSum = __uint_as_float(0x00000000u);
  float partialSumSq = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  unsigned int workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    __half xF16 = ((batchValid) ? (_input[globalIdx]) : (__float2half(0.000000f)));
    float xVal = ((float)(xF16));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  __syncthreads();
  bool shouldReduce_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sumBuf_128) {
    unsigned int neighborIdx_sumBuf_128 = (lid + 128u);
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sumBuf_64) {
    unsigned int neighborIdx_sumBuf_64 = (lid + 64u);
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sumBuf_32) {
    unsigned int neighborIdx_sumBuf_32 = (lid + 32u);
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sumBuf_16) {
    unsigned int neighborIdx_sumBuf_16 = (lid + 16u);
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sumBuf_8) {
    unsigned int neighborIdx_sumBuf_8 = (lid + 8u);
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sumBuf_4) {
    unsigned int neighborIdx_sumBuf_4 = (lid + 4u);
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sumBuf_2) {
    unsigned int neighborIdx_sumBuf_2 = (lid + 2u);
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sumBuf_1) {
    unsigned int neighborIdx_sumBuf_1 = (lid + 1u);
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  __syncthreads();
  float totalSum = sumBuf[0u];
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float meanSq = (mean * mean);
  float avgSumSq = (totalSumSq / hsF32);
  float variance = (avgSumSq - meanSq);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      __half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      __half gF16 = gamma[iterIdx];
      __half bF16 = beta[iterIdx];
      float g = ((float)(gF16));
      float b = ((float)(bF16));
      float xNorm = ((xVal2 - mean) / std);
      float scaled = (g * xNorm);
      float resultF32 = (scaled + b);
      __half result = ((__half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_multi_pass_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rms_norm_multi_pass_f32(const float* _input, const float* gamma, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * hs);
  float partialSumSq = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  unsigned int workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumSqBuf[lid] = partialSumSq;
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float g = gamma[iterIdx];
      float xNorm = (xVal2 / rms);
      float result = (g * xNorm);
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_hip_navatala_transformer_rms_norm_multi_pass_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_rms_norm_multi_pass_f16(const __half* _input, const __half* gamma, const float* epsilon, const unsigned int* batchSize, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int hs = hiddenSize[0u];
  float eps = epsilon[0u];
  __shared__ float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * hs);
  float partialSumSq = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  unsigned int workgroupSize = 256u;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    __half xF16 = ((batchValid) ? (_input[globalIdx]) : (__float2half(0.000000f)));
    float xVal = ((float)(xF16));
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumSqBuf[lid] = partialSumSq;
  __syncthreads();
  bool shouldReduce_sumSqBuf_128 = (lid < 128u);
  if (shouldReduce_sumSqBuf_128) {
    unsigned int neighborIdx_sumSqBuf_128 = (lid + 128u);
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_64 = (lid < 64u);
  if (shouldReduce_sumSqBuf_64) {
    unsigned int neighborIdx_sumSqBuf_64 = (lid + 64u);
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_32 = (lid < 32u);
  if (shouldReduce_sumSqBuf_32) {
    unsigned int neighborIdx_sumSqBuf_32 = (lid + 32u);
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_16 = (lid < 16u);
  if (shouldReduce_sumSqBuf_16) {
    unsigned int neighborIdx_sumSqBuf_16 = (lid + 16u);
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_8 = (lid < 8u);
  if (shouldReduce_sumSqBuf_8) {
    unsigned int neighborIdx_sumSqBuf_8 = (lid + 8u);
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_4 = (lid < 4u);
  if (shouldReduce_sumSqBuf_4) {
    unsigned int neighborIdx_sumSqBuf_4 = (lid + 4u);
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_2 = (lid < 2u);
  if (shouldReduce_sumSqBuf_2) {
    unsigned int neighborIdx_sumSqBuf_2 = (lid + 2u);
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sumSqBuf_1 = (lid < 1u);
  if (shouldReduce_sumSqBuf_1) {
    unsigned int neighborIdx_sumSqBuf_1 = (lid + 1u);
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  __syncthreads();
  float totalSumSq = sumSqBuf[0u];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      __half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      __half gF16 = gamma[iterIdx];
      float g = ((float)(gF16));
      float xNorm = (xVal2 / rms);
      float resultF32 = (g * xNorm);
      __half result = ((__half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_forward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_softmax_forward_f32(const float* _input, const unsigned int* batchSize, const unsigned int* seqLen, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  float x = (((batchValid && seqValid)) ? (_input[globalIdx]) : (__uint_as_float(0xf149f2cau)));
  maxBuf[lid] = x;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((batchValid && seqValid)) {
    float result = (expVal / sumExp);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_forward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_softmax_forward_f16(const __half* _input, const unsigned int* batchSize, const unsigned int* seqLen, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  __half xF16 = (((batchValid && seqValid)) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float x = ((seqValid) ? (((float)(xF16))) : (__uint_as_float(0xf149f2cau)));
  maxBuf[lid] = x;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((batchValid && seqValid)) {
    float resultF32 = (expVal / sumExp);
    __half result = ((__half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_forward_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_softmax_forward_bf16(const __hip_bfloat16* _input, const unsigned int* batchSize, const unsigned int* seqLen, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  __hip_bfloat16 xBF16 = (((batchValid && seqValid)) ? (_input[globalIdx]) : (__float2bfloat16(0.000000f)));
  float x = ((seqValid) ? (((float)(xBF16))) : (__uint_as_float(0xf149f2cau)));
  maxBuf[lid] = x;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((batchValid && seqValid)) {
    float resultF32 = (expVal / sumExp);
    __hip_bfloat16 result = ((__hip_bfloat16)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_softmax_backward_f32(const float* _output, const float* gradOutput, const unsigned int* batchSize, const unsigned int* seqLen, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  float y = (((batchValid && seqValid)) ? (_output[globalIdx]) : (__uint_as_float(0x00000000u)));
  float dy = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : (__uint_as_float(0x00000000u)));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  __syncthreads();
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    unsigned int neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    unsigned int neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    unsigned int neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    unsigned int neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    unsigned int neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    unsigned int neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    unsigned int neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    unsigned int neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  __syncthreads();
  float dot = dotBuf[0u];
  if ((batchValid && seqValid)) {
    float dyMinusDot = (dy - dot);
    float dx = (y * dyMinusDot);
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_softmax_backward_f16(const __half* _output, const __half* gradOutput, const unsigned int* batchSize, const unsigned int* seqLen, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  unsigned int globalIdx = ((batchIdx * sl) + lid);
  __half yF16 = (((batchValid && seqValid)) ? (_output[globalIdx]) : (__float2half(0.000000f)));
  __half dyF16 = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : (__float2half(0.000000f)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  __syncthreads();
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    unsigned int neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    unsigned int neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    unsigned int neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    unsigned int neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    unsigned int neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    unsigned int neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    unsigned int neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    unsigned int neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  __syncthreads();
  float dot = dotBuf[0u];
  if ((batchValid && seqValid)) {
    float dyMinusDot = (dy - dot);
    float dxF32 = (y * dyMinusDot);
    __half dx = ((__half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_with_mask_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_softmax_with_mask_f32(const float* _input, const unsigned int* batchSize, const unsigned int* seqLen, const bool* isCausal, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  bool causal = isCausal[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  unsigned int col = lid;
  unsigned int row = (rowIdx % sl);
  unsigned int globalIdx = ((rowIdx * sl) + lid);
  float rawX = ((valid) ? (_input[globalIdx]) : (__uint_as_float(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (__uint_as_float(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float result = ((isMasked) ? (__uint_as_float(0x00000000u)) : ((expVal / sumExp)));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_with_mask_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_softmax_with_mask_f16(const __half* _input, const unsigned int* batchSize, const unsigned int* seqLen, const bool* isCausal, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  bool causal = isCausal[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  unsigned int col = lid;
  unsigned int row = (rowIdx % sl);
  unsigned int globalIdx = ((rowIdx * sl) + lid);
  __half rawXF16 = ((valid) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float rawX = ((colValid) ? (((float)(rawXF16))) : (__uint_as_float(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (__uint_as_float(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float maxVal = maxBuf[0u];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (__uint_as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float resultF32 = ((isMasked) ? (__uint_as_float(0x00000000u)) : ((expVal / sumExp)));
    __half result = ((__half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_with_mask_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_softmax_with_mask_backward_f32(const float* _output, const float* gradOutput, const unsigned int* batchSize, const unsigned int* seqLen, const bool* isCausal, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  bool causal = isCausal[0u];
  __shared__ float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  unsigned int col = lid;
  unsigned int row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  unsigned int globalIdx = ((rowIdx * sl) + lid);
  float y = ((valid) ? (_output[globalIdx]) : (__uint_as_float(0x00000000u)));
  float dy = ((valid) ? (gradOutput[globalIdx]) : (__uint_as_float(0x00000000u)));
  float yDy = ((isMasked) ? (__uint_as_float(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  __syncthreads();
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    unsigned int neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    unsigned int neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    unsigned int neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    unsigned int neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    unsigned int neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    unsigned int neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    unsigned int neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    unsigned int neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  __syncthreads();
  float dot = dotBuf[0u];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxUnmasked = (y * dyMinusDot);
    float dx = ((isMasked) ? (__uint_as_float(0x00000000u)) : (dxUnmasked));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_with_mask_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_softmax_with_mask_backward_f16(const __half* _output, const __half* gradOutput, const unsigned int* batchSize, const unsigned int* seqLen, const bool* isCausal, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int rowIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  bool causal = isCausal[0u];
  __shared__ float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  unsigned int col = lid;
  unsigned int row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  unsigned int globalIdx = ((rowIdx * sl) + lid);
  __half yF16 = ((valid) ? (_output[globalIdx]) : (__float2half(0.000000f)));
  __half dyF16 = ((valid) ? (gradOutput[globalIdx]) : (__float2half(0.000000f)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = ((isMasked) ? (__uint_as_float(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  __syncthreads();
  bool shouldReduce_sum_dotBuf_128 = (lid < 128u);
  if (shouldReduce_sum_dotBuf_128) {
    unsigned int neighborIdx_sum_dotBuf_128 = (lid + 128u);
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_64 = (lid < 64u);
  if (shouldReduce_sum_dotBuf_64) {
    unsigned int neighborIdx_sum_dotBuf_64 = (lid + 64u);
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_32 = (lid < 32u);
  if (shouldReduce_sum_dotBuf_32) {
    unsigned int neighborIdx_sum_dotBuf_32 = (lid + 32u);
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_16 = (lid < 16u);
  if (shouldReduce_sum_dotBuf_16) {
    unsigned int neighborIdx_sum_dotBuf_16 = (lid + 16u);
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_8 = (lid < 8u);
  if (shouldReduce_sum_dotBuf_8) {
    unsigned int neighborIdx_sum_dotBuf_8 = (lid + 8u);
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_4 = (lid < 4u);
  if (shouldReduce_sum_dotBuf_4) {
    unsigned int neighborIdx_sum_dotBuf_4 = (lid + 4u);
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_2 = (lid < 2u);
  if (shouldReduce_sum_dotBuf_2) {
    unsigned int neighborIdx_sum_dotBuf_2 = (lid + 2u);
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_dotBuf_1 = (lid < 1u);
  if (shouldReduce_sum_dotBuf_1) {
    unsigned int neighborIdx_sum_dotBuf_1 = (lid + 1u);
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  __syncthreads();
  float dot = dotBuf[0u];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxF32Unmasked = (y * dyMinusDot);
    float dxF32 = ((isMasked) ? (__uint_as_float(0x00000000u)) : (dxF32Unmasked));
    __half dx = ((__half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_multi_pass_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_softmax_multi_pass_f32(const float* _input, const unsigned int* batchSize, const unsigned int* seqLen, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * sl);
  unsigned int workgroupSize = 256u;
  float partialMax = __uint_as_float(0xf149f2cau);
  float partialSum = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (__uint_as_float(0xf149f2cau)));
    bool isGreater = (xVal > partialMax);
    partialMax = ((isGreater) ? (xVal) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float globalMax = maxBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float globalSum = sumBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      unsigned int globalIdx3 = (baseIdx + iterIdx);
      float xVal3 = _input[globalIdx3];
      float xShifted3 = (xVal3 - globalMax);
      float expVal3 = exp(xShifted3);
      float result = (expVal3 / globalSum);
      _output[globalIdx3] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_hip_navatala_transformer_softmax_multi_pass_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_softmax_multi_pass_f16(const __half* _input, const unsigned int* batchSize, const unsigned int* seqLen, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  __shared__ float maxBuf[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  unsigned int baseIdx = (batchIdx * sl);
  unsigned int workgroupSize = 256u;
  float partialMax = __uint_as_float(0xf149f2cau);
  float partialSum = __uint_as_float(0x00000000u);
  unsigned int iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    unsigned int globalIdx = (baseIdx + iterIdx);
    __half xF16 = ((batchValid) ? (_input[globalIdx]) : (__float2half(0.000000f)));
    float xVal = ((float)(xF16));
    float xValSafe = ((batchValid) ? (xVal) : (__uint_as_float(0xf149f2cau)));
    bool isGreater = (xValSafe > partialMax);
    partialMax = ((isGreater) ? (xValSafe) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  __syncthreads();
  bool shouldReduce_max_maxBuf_128 = (lid < 128u);
  if (shouldReduce_max_maxBuf_128) {
    unsigned int neighborIdx_max_maxBuf_128 = (lid + 128u);
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_64 = (lid < 64u);
  if (shouldReduce_max_maxBuf_64) {
    unsigned int neighborIdx_max_maxBuf_64 = (lid + 64u);
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_32 = (lid < 32u);
  if (shouldReduce_max_maxBuf_32) {
    unsigned int neighborIdx_max_maxBuf_32 = (lid + 32u);
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_16 = (lid < 16u);
  if (shouldReduce_max_maxBuf_16) {
    unsigned int neighborIdx_max_maxBuf_16 = (lid + 16u);
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_8 = (lid < 8u);
  if (shouldReduce_max_maxBuf_8) {
    unsigned int neighborIdx_max_maxBuf_8 = (lid + 8u);
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_4 = (lid < 4u);
  if (shouldReduce_max_maxBuf_4) {
    unsigned int neighborIdx_max_maxBuf_4 = (lid + 4u);
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_2 = (lid < 2u);
  if (shouldReduce_max_maxBuf_2) {
    unsigned int neighborIdx_max_maxBuf_2 = (lid + 2u);
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_maxBuf_1 = (lid < 1u);
  if (shouldReduce_max_maxBuf_1) {
    unsigned int neighborIdx_max_maxBuf_1 = (lid + 1u);
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  __syncthreads();
  float globalMax = maxBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      unsigned int globalIdx2 = (baseIdx + iterIdx);
      __half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  __syncthreads();
  bool shouldReduce_sum_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_sumBuf_128) {
    unsigned int neighborIdx_sum_sumBuf_128 = (lid + 128u);
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_sumBuf_64) {
    unsigned int neighborIdx_sum_sumBuf_64 = (lid + 64u);
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_sumBuf_32) {
    unsigned int neighborIdx_sum_sumBuf_32 = (lid + 32u);
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_sumBuf_16) {
    unsigned int neighborIdx_sum_sumBuf_16 = (lid + 16u);
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_sumBuf_8) {
    unsigned int neighborIdx_sum_sumBuf_8 = (lid + 8u);
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_sumBuf_4) {
    unsigned int neighborIdx_sum_sumBuf_4 = (lid + 4u);
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_sumBuf_2) {
    unsigned int neighborIdx_sum_sumBuf_2 = (lid + 2u);
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_sumBuf_1) {
    unsigned int neighborIdx_sum_sumBuf_1 = (lid + 1u);
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  __syncthreads();
  float globalSum = sumBuf[0u];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      unsigned int globalIdx3 = (baseIdx + iterIdx);
      __half xF16_3 = _input[globalIdx3];
      float xVal3 = ((float)(xF16_3));
      float xShifted3 = (xVal3 - globalMax);
      float expVal3 = exp(xShifted3);
      float resultF32 = (expVal3 / globalSum);
      __half result = ((__half)(resultF32));
      _output[globalIdx3] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f32_to_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_cast_f32_to_f16(const float* _input, const unsigned int* count, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    __half result = ((__half)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f16_to_f32 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_cast_f16_to_f32(const __half* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half x = _input[gid];
    float result = ((float)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f32_to_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_cast_f32_to_bf16(const float* _input, const unsigned int* count, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    __hip_bfloat16 result = ((__hip_bfloat16)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_b_f16_to_f32 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_cast_b_f16_to_f32(const __hip_bfloat16* _input, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __hip_bfloat16 x = _input[gid];
    float result = ((float)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_transpose_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_transpose_f32(const float* _input, const unsigned int* rows, const unsigned int* cols, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int r = rows[0u];
  unsigned int c = cols[0u];
  unsigned int total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / c);
    unsigned int col = (gid % c);
    float x = _input[gid];
    unsigned int outIdx = ((col * r) + row);
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_hip_navatala_transformer_transpose_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_transpose_f16(const __half* _input, const unsigned int* rows, const unsigned int* cols, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int r = rows[0u];
  unsigned int c = cols[0u];
  unsigned int total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / c);
    unsigned int col = (gid % c);
    __half x = _input[gid];
    unsigned int outIdx = ((col * r) + row);
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_hip_navatala_transformer_fused_cast_transpose_f32_to_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_fused_cast_transpose_f32_to_f16(const float* _input, const unsigned int* rows, const unsigned int* cols, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int r = rows[0u];
  unsigned int c = cols[0u];
  unsigned int total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / c);
    unsigned int col = (gid % c);
    float x = _input[gid];
    __half xF16 = ((__half)(x));
    unsigned int outIdx = ((col * r) + row);
    _output[outIdx] = xF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f32_to_f_p8_e4_m3 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_cast_f32_to_f_p8_e4_m3(const float* _input, const float* scale, const unsigned int* count, unsigned char* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = __uint_as_float(0xc3700000u);
    float maxVal = __uint_as_float(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    unsigned char fp8Val = ((unsigned char)(rounded));
    _output[gid] = fp8Val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f_p8_e4_m3_to_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_cast_f_p8_e4_m3_to_f32(const unsigned char* _input, const float* scale, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned char fp8Val = _input[gid];
    float s = scale[0u];
    float f32Val = ((float)(fp8Val));
    float result = (f32Val / s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f32_to_f_p8_e5_m2 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_cast_f32_to_f_p8_e5_m2(const float* _input, const float* scale, const unsigned int* count, unsigned char* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = __uint_as_float(0xc7600000u);
    float maxVal = __uint_as_float(0x47600000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    unsigned char fp8Val = ((unsigned char)(rounded));
    _output[gid] = fp8Val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_cast_f_p8_e5_m2_to_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_cast_f_p8_e5_m2_to_f32(const unsigned char* _input, const float* scale, const unsigned int* count, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned char fp8Val = _input[gid];
    float s = scale[0u];
    float f32Val = ((float)(fp8Val));
    float result = (f32Val / s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_fused_cast_transpose_f_p8 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_fused_cast_transpose_f_p8(const float* _input, const float* scale, const unsigned int* rows, const unsigned int* cols, unsigned char* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int r = rows[0u];
  unsigned int c = cols[0u];
  unsigned int total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / c);
    unsigned int col = (gid % c);
    float x = _input[gid];
    float s = scale[0u];
    float scaled = (x * s);
    float minVal = __uint_as_float(0xc3700000u);
    float maxVal = __uint_as_float(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    unsigned char fp8Val = ((unsigned char)(rounded));
    unsigned int outIdx = ((col * r) + row);
    _output[outIdx] = fp8Val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_transpose4_d_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_transpose4_d_f32(const float* _input, const unsigned int* dim0, const unsigned int* dim1, const unsigned int* dim2, const unsigned int* dim3, const unsigned int* perm0, const unsigned int* perm1, const unsigned int* perm2, const unsigned int* perm3, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d0 = dim0[0u];
  unsigned int d1 = dim1[0u];
  unsigned int d2 = dim2[0u];
  unsigned int d3 = dim3[0u];
  unsigned int p0 = perm0[0u];
  unsigned int p1 = perm1[0u];
  unsigned int p2 = perm2[0u];
  unsigned int p3 = perm3[0u];
  unsigned int total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int stride1 = ((d1 * d2) * d3);
    unsigned int stride2 = (d2 * d3);
    unsigned int stride3 = d3;
    unsigned int i0 = (gid / stride1);
    unsigned int rem0 = (gid % stride1);
    unsigned int i1 = (rem0 / stride2);
    unsigned int rem1 = (rem0 % stride2);
    unsigned int i2 = (rem1 / stride3);
    unsigned int i3 = (rem1 % stride3);
    unsigned int outD0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    unsigned int outD1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    unsigned int outD2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    unsigned int outD3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    unsigned int outI0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    unsigned int outI1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    unsigned int outI2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    unsigned int outI3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    unsigned int outStride1 = ((outD1 * outD2) * outD3);
    unsigned int outStride2 = (outD2 * outD3);
    unsigned int outStride3 = outD3;
    unsigned int outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    float x = _input[gid];
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_hip_navatala_transformer_transpose4_d_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_transpose4_d_f16(const __half* _input, const unsigned int* dim0, const unsigned int* dim1, const unsigned int* dim2, const unsigned int* dim3, const unsigned int* perm0, const unsigned int* perm1, const unsigned int* perm2, const unsigned int* perm3, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d0 = dim0[0u];
  unsigned int d1 = dim1[0u];
  unsigned int d2 = dim2[0u];
  unsigned int d3 = dim3[0u];
  unsigned int p0 = perm0[0u];
  unsigned int p1 = perm1[0u];
  unsigned int p2 = perm2[0u];
  unsigned int p3 = perm3[0u];
  unsigned int total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int stride1 = ((d1 * d2) * d3);
    unsigned int stride2 = (d2 * d3);
    unsigned int stride3 = d3;
    unsigned int i0 = (gid / stride1);
    unsigned int rem0 = (gid % stride1);
    unsigned int i1 = (rem0 / stride2);
    unsigned int rem1 = (rem0 % stride2);
    unsigned int i2 = (rem1 / stride3);
    unsigned int i3 = (rem1 % stride3);
    unsigned int outD0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    unsigned int outD1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    unsigned int outD2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    unsigned int outD3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    unsigned int outI0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    unsigned int outI1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    unsigned int outI2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    unsigned int outI3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    unsigned int outStride1 = ((outD1 * outD2) * outD3);
    unsigned int outStride2 = (outD2 * outD3);
    unsigned int outStride3 = outD3;
    unsigned int outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    __half x = _input[gid];
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_hip_navatala_transformer_dropout_forward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_dropout_forward_f32(const float* _input, const unsigned int* count, const float* prob, const unsigned int* seed, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  float p = prob[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    unsigned int h0 = (gid ^ s);
    unsigned int h1 = (h0 * 3432918353u);
    unsigned int h2 = (h1 ^ (h1 >> 15u));
    unsigned int h3 = (h2 * 461845907u);
    unsigned int hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / __uint_as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (__uint_as_float(0x3f800000u) - p);
    float scale = (__uint_as_float(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float result = ((keep) ? (scaled) : (__uint_as_float(0x00000000u)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_dropout_forward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_dropout_forward_f16(const __half* _input, const unsigned int* count, const float* prob, const unsigned int* seed, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  float p = prob[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half xF16 = _input[gid];
    float x = ((float)(xF16));
    unsigned int h0 = (gid ^ s);
    unsigned int h1 = (h0 * 3432918353u);
    unsigned int h2 = (h1 ^ (h1 >> 15u));
    unsigned int h3 = (h2 * 461845907u);
    unsigned int hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / __uint_as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (__uint_as_float(0x3f800000u) - p);
    float scale = (__uint_as_float(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float resultF32 = ((keep) ? (scaled) : (__uint_as_float(0x00000000u)));
    __half result = ((__half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_dropout_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_dropout_backward_f32(const float* gradOutput, const unsigned int* count, const float* prob, const unsigned int* seed, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  float p = prob[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float dy = gradOutput[gid];
    unsigned int h0 = (gid ^ s);
    unsigned int h1 = (h0 * 3432918353u);
    unsigned int h2 = (h1 ^ (h1 >> 15u));
    unsigned int h3 = (h2 * 461845907u);
    unsigned int hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / __uint_as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (__uint_as_float(0x3f800000u) - p);
    float scale = (__uint_as_float(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dx = ((keep) ? (scaled) : (__uint_as_float(0x00000000u)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_dropout_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_dropout_backward_f16(const __half* gradOutput, const unsigned int* count, const float* prob, const unsigned int* seed, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  float p = prob[0u];
  unsigned int s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    __half dyF16 = gradOutput[gid];
    float dy = ((float)(dyF16));
    unsigned int h0 = (gid ^ s);
    unsigned int h1 = (h0 * 3432918353u);
    unsigned int h2 = (h1 ^ (h1 >> 15u));
    unsigned int h3 = (h2 * 461845907u);
    unsigned int hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / __uint_as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (__uint_as_float(0x3f800000u) - p);
    float scale = (__uint_as_float(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dxF32 = ((keep) ? (scaled) : (__uint_as_float(0x00000000u)));
    __half dx = ((__half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_forward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rope_forward_f32(const float* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1 = (x0Sin + x1Cos);
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_forward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_rope_forward_f16(const __half* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    __half x0F16 = _input[idx0];
    __half x1F16 = _input[idx1];
    float x0 = ((float)(x0F16));
    float x1 = ((float)(x1F16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    __half y0 = ((__half)(y0F32));
    __half y1 = ((__half)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_forward_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_rope_forward_bf16(const __hip_bfloat16* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    __hip_bfloat16 x0BF16 = _input[idx0];
    __hip_bfloat16 x1BF16 = _input[idx1];
    float x0 = ((float)(x0BF16));
    float x1 = ((float)(x1BF16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    __hip_bfloat16 y0 = ((__hip_bfloat16)(y0F32));
    __hip_bfloat16 y1 = ((__hip_bfloat16)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_with_cache_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rope_with_cache_f32(const float* _input, const float* cosCache, const float* sinCache, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    unsigned int cacheIdx = ((pos * halfHd) + dimPair);
    float cosTheta = cosCache[cacheIdx];
    float sinTheta = sinCache[cacheIdx];
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float y0 = ((x0 * cosTheta) - (x1 * sinTheta));
    float y1 = ((x0 * sinTheta) + (x1 * cosTheta));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rope_backward_f32(const float* gradOutput, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, float* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    float dy0 = gradOutput[idx0];
    float dy1 = gradOutput[idx1];
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (__uint_as_float(0x00000000u) - dy0Sin);
    float dx1 = (negDy0Sin + dy1Cos);
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_rope_backward_f16(const __half* gradOutput, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, __half* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    __half dy0F16 = gradOutput[idx0];
    __half dy1F16 = gradOutput[idx1];
    float dy0 = ((float)(dy0F16));
    float dy1 = ((float)(dy1F16));
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0F32 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (__uint_as_float(0x00000000u) - dy0Sin);
    float dx1F32 = (negDy0Sin + dy1Cos);
    __half dx0 = ((__half)(dx0F32));
    __half dx1 = ((__half)(dx1F32));
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_backward_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_rope_backward_bf16(const __hip_bfloat16* gradOutput, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, __hip_bfloat16* gradInput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimPair = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int dim0 = (dimPair * 2u);
    unsigned int dim1 = (dim0 + 1u);
    unsigned int idx0 = (baseOffset + (seqOffset + dim0));
    unsigned int idx1 = (baseOffset + (seqOffset + dim1));
    __hip_bfloat16 dy0BF16 = gradOutput[idx0];
    __hip_bfloat16 dy1BF16 = gradOutput[idx1];
    float dy0 = ((float)(dy0BF16));
    float dy1 = ((float)(dy1BF16));
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0F32 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (__uint_as_float(0x00000000u) - dy0Sin);
    float dx1F32 = (negDy0Sin + dy1Cos);
    __hip_bfloat16 dx0 = ((__hip_bfloat16)(dx0F32));
    __hip_bfloat16 dx1 = ((__hip_bfloat16)(dx1F32));
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_interleaved_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_rope_interleaved_f32(const float* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimIdx = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int idx0 = (baseOffset + (seqOffset + dimIdx));
    unsigned int idx1 = (idx0 + halfHd);
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1 = (x0Sin + x1Cos);
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_interleaved_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_rope_interleaved_f16(const __half* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimIdx = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int idx0 = (baseOffset + (seqOffset + dimIdx));
    unsigned int idx1 = (idx0 + halfHd);
    __half x0F16 = _input[idx0];
    __half x1F16 = _input[idx1];
    float x0 = ((float)(x0F16));
    float x1 = ((float)(x1F16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    __half y0 = ((__half)(y0F32));
    __half y1 = ((__half)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_rope_interleaved_bf16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_transformer_rope_interleaved_bf16(const __hip_bfloat16* _input, const unsigned int* positions, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* headDim, const float* base, __hip_bfloat16* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int batchSeqPairs = (sl * halfHd);
    unsigned int batchIdx = (gid / batchSeqPairs);
    unsigned int remainder = (gid % batchSeqPairs);
    unsigned int seqIdx = (remainder / halfHd);
    unsigned int dimIdx = (remainder % halfHd);
    unsigned int posIdx = ((batchIdx * sl) + seqIdx);
    unsigned int pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    unsigned int baseOffset = (batchIdx * (sl * hd));
    unsigned int seqOffset = (seqIdx * hd);
    unsigned int idx0 = (baseOffset + (seqOffset + dimIdx));
    unsigned int idx1 = (idx0 + halfHd);
    __hip_bfloat16 x0BF16 = _input[idx0];
    __hip_bfloat16 x1BF16 = _input[idx1];
    float x0 = ((float)(x0BF16));
    float x1 = ((float)(x1BF16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    __hip_bfloat16 y0 = ((__hip_bfloat16)(y0F32));
    __hip_bfloat16 y1 = ((__hip_bfloat16)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_simple_attention_f16(const __half* query, const __half* key, const __half* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      __half qValF16 = query[qIdx];
      __half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    __half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  __syncthreads();
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    __half outF16 = ((__half)(outVal));
    unsigned int outIdx = (qBase + 0u);
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_simple_attention_f32(const float* query, const float* key, const float* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  __syncthreads();
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    unsigned int outIdx = (qBase + 0u);
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_paged_attention_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_paged_attention_f32(const float* query, const float* kvCache, const unsigned int* blockTables, const unsigned int* contextLens, const unsigned int* numHeads, const unsigned int* headDim, const unsigned int* pageSize, const unsigned int* maxBlocks, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int ps = pageSize[0u];
  unsigned int mb = maxBlocks[0u];
  unsigned int batchIdx = (groupId / nh);
  unsigned int headIdx = (groupId % nh);
  unsigned int ctxLen = contextLens[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool kvValid = (lid < ctxLen);
  unsigned int qBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  unsigned int blockIdx = (lid / ps);
  unsigned int blockOffset = (lid % ps);
  unsigned int blockTableBase = (batchIdx * mb);
  unsigned int physicalBlock = blockTables[(blockTableBase + blockIdx)];
  unsigned int kvBlockStride = (2u * (nh * (ps * hd)));
  unsigned int kvSlotStride = (nh * (ps * hd));
  unsigned int kvHeadStride = (ps * hd);
  unsigned int kBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  unsigned int vBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  if (kvValid) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float qVal = query[qIdx];
      float kVal = kvCache[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if (kvValid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  for (int outD = 0; outD < (int)(hd); ++outD) {
    if (kvValid) {
      float myExpScore = attnScores[lid];
      float shiftedExp = (myExpScore - maxScore);
      float expVal = exp(shiftedExp);
      float attnWeight = (expVal / sumExp);
      unsigned int vIdx = (vBase + outD);
      float vVal = kvCache[vIdx];
      float weighted = (attnWeight * vVal);
      sumBuf[lid] = weighted;
    } else {
      sumBuf[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
    if (shouldReduce_sum_p2_sumBuf_128) {
      unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
      float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
      float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
    if (shouldReduce_sum_p2_sumBuf_64) {
      unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
      float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
      float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
    if (shouldReduce_sum_p2_sumBuf_32) {
      unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
      float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
      float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
    if (shouldReduce_sum_p2_sumBuf_16) {
      unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
      float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
      float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
    if (shouldReduce_sum_p2_sumBuf_8) {
      unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
      float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
      float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
    if (shouldReduce_sum_p2_sumBuf_4) {
      unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
      float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
      float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
    if (shouldReduce_sum_p2_sumBuf_2) {
      unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
      float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
      float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
    }
    __syncthreads();
    bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
    if (shouldReduce_sum_p2_sumBuf_1) {
      unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
      float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
      float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
    }
    __syncthreads();
    if ((lid == 0u)) {
      float outVal = sumBuf[0u];
      unsigned int outIdx = (qBase + outD);
      _output[outIdx] = outVal;
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_with_mask_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_simple_attention_with_mask_f32(const float* query, const float* key, const float* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool causalMasked = (lid > queryPos);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!causalMasked))) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if ((valid && (!causalMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((valid && (!causalMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  __syncthreads();
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    unsigned int outIdx = (qBase + 0u);
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_with_mask_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_simple_attention_with_mask_f16(const __half* query, const __half* key, const __half* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool causalMasked = (lid > queryPos);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!causalMasked))) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      __half qValF16 = query[qIdx];
      __half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if ((valid && (!causalMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((valid && (!causalMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    __half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  __syncthreads();
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    __half outF16 = ((__half)(outVal));
    unsigned int outIdx = (qBase + 0u);
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_with_padding_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_simple_attention_with_padding_f32(const float* query, const float* key, const float* value, const unsigned int* seqLengths, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* numHeads, const unsigned int* headDim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = maxSeqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  unsigned int actualSeqLen = seqLengths[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool paddingMasked = (lid >= actualSeqLen);
  bool queryMasked = (queryPos >= actualSeqLen);
  bool anyMasked = (paddingMasked || queryMasked);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!anyMasked))) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if ((valid && (!anyMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((valid && (!anyMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  __syncthreads();
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    unsigned int outIdx = (qBase + 0u);
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_with_padding_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_simple_attention_with_padding_f16(const __half* query, const __half* key, const __half* value, const unsigned int* seqLengths, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* numHeads, const unsigned int* headDim, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = maxSeqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  unsigned int actualSeqLen = seqLengths[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool paddingMasked = (lid >= actualSeqLen);
  bool queryMasked = (queryPos >= actualSeqLen);
  bool anyMasked = (paddingMasked || queryMasked);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!anyMasked))) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      __half qValF16 = query[qIdx];
      __half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if ((valid && (!anyMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if ((valid && (!anyMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    __half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p2_sumBuf_128) {
    unsigned int neighborIdx_sum_p2_sumBuf_128 = (lid + 128u);
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p2_sumBuf_64) {
    unsigned int neighborIdx_sum_p2_sumBuf_64 = (lid + 64u);
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p2_sumBuf_32) {
    unsigned int neighborIdx_sum_p2_sumBuf_32 = (lid + 32u);
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p2_sumBuf_16) {
    unsigned int neighborIdx_sum_p2_sumBuf_16 = (lid + 16u);
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p2_sumBuf_8) {
    unsigned int neighborIdx_sum_p2_sumBuf_8 = (lid + 8u);
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p2_sumBuf_4) {
    unsigned int neighborIdx_sum_p2_sumBuf_4 = (lid + 4u);
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p2_sumBuf_2) {
    unsigned int neighborIdx_sum_p2_sumBuf_2 = (lid + 2u);
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p2_sumBuf_1) {
    unsigned int neighborIdx_sum_p2_sumBuf_1 = (lid + 1u);
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  __syncthreads();
  if ((batchValid && (lid == 0u))) {
    float outVal = sumBuf[0u];
    __half outF16 = ((__half)(outVal));
    unsigned int outIdx = (qBase + 0u);
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_simple_attention_backward_f32(const float* gradOutput, const float* query, const float* key, const float* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, float* gradQuery, float* gradKey, float* gradValue) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  __shared__ float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProdQK = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float qVal = query[qIdx];
      float kVal = key[kIdx];
      float prod = (qVal * kVal);
      float oldDotProdQK = dotProdQK;
      dotProdQK = (oldDotProdQK + prod);
    }
    float finalDotProdQK = dotProdQK;
    float scaledScore = (finalDotProdQK * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float dotProdGradAttn = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int gradOutIdx = (qBase + ((unsigned int)(d)));
      unsigned int vIdx = (vBase + ((unsigned int)(d)));
      float gradOutVal = gradOutput[gradOutIdx];
      float vVal = value[vIdx];
      float prodGradV = (gradOutVal * vVal);
      float oldDotProdGradAttn = dotProdGradAttn;
      dotProdGradAttn = (oldDotProdGradAttn + prodGradV);
      float gradVContrib = (attnWeight * gradOutVal);
      float oldGradV = gradValue[vIdx];
      float newGradV = (oldGradV + gradVContrib);
      gradValue[vIdx] = newGradV;
    }
    float finalGradAttn = dotProdGradAttn;
    gradBuf[lid] = finalGradAttn;
  } else {
    gradBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      float kVal = key[kIdx];
      float qVal = query[qIdx];
      float gradQContrib = (scaledGradScore * kVal);
      float oldGradQ = gradQuery[qIdx];
      float newGradQ = (oldGradQ + gradQContrib);
      gradQuery[qIdx] = newGradQ;
      float gradKContrib = (scaledGradScore * qVal);
      float oldGradK = gradKey[kIdx];
      float newGradK = (oldGradK + gradKContrib);
      gradKey[kIdx] = newGradK;
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_simple_attention_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_simple_attention_backward_f16(const __half* gradOutput, const __half* query, const __half* key, const __half* value, const unsigned int* batchSize, const unsigned int* seqLen, const unsigned int* numHeads, const unsigned int* headDim, __half* gradQuery, __half* gradKey, __half* gradValue) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int seqHeads = (sl * nh);
  unsigned int batchIdx = (groupId / seqHeads);
  unsigned int remainder = (groupId % seqHeads);
  unsigned int queryPos = (remainder / nh);
  unsigned int headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (__uint_as_float(0x3f800000u) / sqrtHd);
  __shared__ float attnScores[256];
  __shared__ float sumBuf[256];
  __shared__ float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  unsigned int headDimStride = hd;
  unsigned int headStride = (nh * hd);
  unsigned int seqStride = (sl * headStride);
  unsigned int qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  unsigned int kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      __half qValF16 = query[qIdx];
      __half kValF16 = key[kIdx];
      float qVal = ((float)(qValF16));
      float kVal = ((float)(kValF16));
      float prod = (qVal * kVal);
      float oldDotProd = dotProd;
      dotProd = (oldDotProd + prod);
    }
    float finalDotProd = dotProd;
    float scaledScore = (finalDotProd * scale);
    attnScores[lid] = scaledScore;
  } else {
    attnScores[lid] = __uint_as_float(0xf149f2cau);
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_128 = (lid < 128u);
  if (shouldReduce_max_p1_attnScores_128) {
    unsigned int neighborIdx_max_p1_attnScores_128 = (lid + 128u);
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_64 = (lid < 64u);
  if (shouldReduce_max_p1_attnScores_64) {
    unsigned int neighborIdx_max_p1_attnScores_64 = (lid + 64u);
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_32 = (lid < 32u);
  if (shouldReduce_max_p1_attnScores_32) {
    unsigned int neighborIdx_max_p1_attnScores_32 = (lid + 32u);
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_16 = (lid < 16u);
  if (shouldReduce_max_p1_attnScores_16) {
    unsigned int neighborIdx_max_p1_attnScores_16 = (lid + 16u);
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_8 = (lid < 8u);
  if (shouldReduce_max_p1_attnScores_8) {
    unsigned int neighborIdx_max_p1_attnScores_8 = (lid + 8u);
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_4 = (lid < 4u);
  if (shouldReduce_max_p1_attnScores_4) {
    unsigned int neighborIdx_max_p1_attnScores_4 = (lid + 4u);
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_2 = (lid < 2u);
  if (shouldReduce_max_p1_attnScores_2) {
    unsigned int neighborIdx_max_p1_attnScores_2 = (lid + 2u);
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  __syncthreads();
  bool shouldReduce_max_p1_attnScores_1 = (lid < 1u);
  if (shouldReduce_max_p1_attnScores_1) {
    unsigned int neighborIdx_max_p1_attnScores_1 = (lid + 1u);
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  __syncthreads();
  float maxScore = attnScores[0u];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < 128u);
  if (shouldReduce_sum_p1_sumBuf_128) {
    unsigned int neighborIdx_sum_p1_sumBuf_128 = (lid + 128u);
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < 64u);
  if (shouldReduce_sum_p1_sumBuf_64) {
    unsigned int neighborIdx_sum_p1_sumBuf_64 = (lid + 64u);
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < 32u);
  if (shouldReduce_sum_p1_sumBuf_32) {
    unsigned int neighborIdx_sum_p1_sumBuf_32 = (lid + 32u);
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < 16u);
  if (shouldReduce_sum_p1_sumBuf_16) {
    unsigned int neighborIdx_sum_p1_sumBuf_16 = (lid + 16u);
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < 8u);
  if (shouldReduce_sum_p1_sumBuf_8) {
    unsigned int neighborIdx_sum_p1_sumBuf_8 = (lid + 8u);
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < 4u);
  if (shouldReduce_sum_p1_sumBuf_4) {
    unsigned int neighborIdx_sum_p1_sumBuf_4 = (lid + 4u);
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < 2u);
  if (shouldReduce_sum_p1_sumBuf_2) {
    unsigned int neighborIdx_sum_p1_sumBuf_2 = (lid + 2u);
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < 1u);
  if (shouldReduce_sum_p1_sumBuf_1) {
    unsigned int neighborIdx_sum_p1_sumBuf_1 = (lid + 1u);
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  __syncthreads();
  float sumExp = sumBuf[0u];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    unsigned int vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float gradAttnAccum = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int gradOutIdx = (qBase + ((unsigned int)(d)));
      unsigned int vIdx = (vBase + ((unsigned int)(d)));
      __half gradOutF16 = gradOutput[gradOutIdx];
      float gradOutVal = ((float)(gradOutF16));
      __half vValF16 = value[vIdx];
      float vVal = ((float)(vValF16));
      float gradVContrib = (attnWeight * gradOutVal);
      __half oldGradVF16 = gradValue[vIdx];
      float oldGradV = ((float)(oldGradVF16));
      float newGradV = (oldGradV + gradVContrib);
      __half newGradVF16 = ((__half)(newGradV));
      gradValue[vIdx] = newGradVF16;
      float gradAttnProd = (gradOutVal * vVal);
      float oldGradAttnAccum = gradAttnAccum;
      gradAttnAccum = (oldGradAttnAccum + gradAttnProd);
    }
    float gradAttn = gradAttnAccum;
    gradBuf[lid] = gradAttn;
  } else {
    gradBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int qIdx = (qBase + ((unsigned int)(d)));
      unsigned int kIdx = (kBase + ((unsigned int)(d)));
      __half kValF16 = key[kIdx];
      float kVal = ((float)(kValF16));
      float gradQContrib = (scaledGradScore * kVal);
      __half oldGradQF16 = gradQuery[qIdx];
      float oldGradQ = ((float)(oldGradQF16));
      float newGradQ = (oldGradQ + gradQContrib);
      __half newGradQF16 = ((__half)(newGradQ));
      gradQuery[qIdx] = newGradQF16;
      __half qValF16 = query[qIdx];
      float qVal = ((float)(qValF16));
      float gradKContrib = (scaledGradScore * qVal);
      __half oldGradKF16 = gradKey[kIdx];
      float oldGradK = ((float)(oldGradKF16));
      float newGradK = (oldGradK + gradKContrib);
      __half newGradKF16 = ((__half)(newGradK));
      gradKey[kIdx] = newGradKF16;
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_kv_cache_append_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_kv_cache_append_f16(const __half* newK, const __half* newV, const unsigned int* blockTables, const unsigned int* seqPositions, const unsigned int* numHeads, const unsigned int* headDim, const unsigned int* pageSize, const unsigned int* maxBlocks, __half* kvCache) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int ps = pageSize[0u];
  unsigned int mb = maxBlocks[0u];
  unsigned int batchIdx = (groupId / nh);
  unsigned int headIdx = (groupId % nh);
  unsigned int seqPos = seqPositions[batchIdx];
  unsigned int blockIdx = (seqPos / ps);
  unsigned int blockOffset = (seqPos % ps);
  unsigned int blockTableBase = (batchIdx * mb);
  unsigned int physicalBlock = blockTables[(blockTableBase + blockIdx)];
  unsigned int kvBlockStride = (2u * (nh * (ps * hd)));
  unsigned int kvSlotStride = (nh * (ps * hd));
  unsigned int kvHeadStride = (ps * hd);
  unsigned int kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  unsigned int vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  unsigned int srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    unsigned int srcIdx = (srcBase + lid);
    unsigned int kDstIdx = (kCacheBase + lid);
    unsigned int vDstIdx = (vCacheBase + lid);
    __half kVal = newK[srcIdx];
    __half vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_kv_cache_append_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_kv_cache_append_f32(const float* newK, const float* newV, const unsigned int* blockTables, const unsigned int* seqPositions, const unsigned int* numHeads, const unsigned int* headDim, const unsigned int* pageSize, const unsigned int* maxBlocks, float* kvCache) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int ps = pageSize[0u];
  unsigned int mb = maxBlocks[0u];
  unsigned int batchIdx = (groupId / nh);
  unsigned int headIdx = (groupId % nh);
  unsigned int seqPos = seqPositions[batchIdx];
  unsigned int blockIdx = (seqPos / ps);
  unsigned int blockOffset = (seqPos % ps);
  unsigned int blockTableBase = (batchIdx * mb);
  unsigned int physicalBlock = blockTables[(blockTableBase + blockIdx)];
  unsigned int kvBlockStride = (2u * (nh * (ps * hd)));
  unsigned int kvSlotStride = (nh * (ps * hd));
  unsigned int kvHeadStride = (ps * hd);
  unsigned int kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  unsigned int vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  unsigned int srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    unsigned int srcIdx = (srcBase + lid);
    unsigned int kDstIdx = (kCacheBase + lid);
    unsigned int vDstIdx = (vCacheBase + lid);
    float kVal = newK[srcIdx];
    float vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_kv_cache_rotate_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_kv_cache_rotate_f16(const unsigned int* blockTables, const unsigned int* contextLens, const unsigned int* shiftAmount, const unsigned int* numHeads, const unsigned int* headDim, const unsigned int* pageSize, const unsigned int* maxBlocks, __half* kvCache) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int groupId = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nh = numHeads[0u];
  unsigned int hd = headDim[0u];
  unsigned int ps = pageSize[0u];
  unsigned int mb = maxBlocks[0u];
  unsigned int shift = shiftAmount[0u];
  unsigned int batchIdx = (groupId / nh);
  unsigned int headIdx = (groupId % nh);
  unsigned int ctxLen = contextLens[batchIdx];
  bool posValid = (lid < ctxLen);
  unsigned int srcPos = (lid + shift);
  bool srcValid = (srcPos < ctxLen);
  bool valid = (posValid && srcValid);
  unsigned int blockTableBase = (batchIdx * mb);
  unsigned int kvBlockStride = (2u * (nh * (ps * hd)));
  unsigned int kvSlotStride = (nh * (ps * hd));
  unsigned int kvHeadStride = (ps * hd);
  if (valid) {
    unsigned int srcBlockIdx = (srcPos / ps);
    unsigned int srcBlockOffset = (srcPos % ps);
    unsigned int srcPhysBlock = blockTables[(blockTableBase + srcBlockIdx)];
    unsigned int dstBlockIdx = (lid / ps);
    unsigned int dstBlockOffset = (lid % ps);
    unsigned int dstPhysBlock = blockTables[(blockTableBase + dstBlockIdx)];
    unsigned int kSrcBase = ((srcPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (srcBlockOffset * hd)));
    unsigned int kDstBase = ((dstPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (dstBlockOffset * hd)));
    unsigned int vSrcBase = ((srcPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (srcBlockOffset * hd))));
    unsigned int vDstBase = ((dstPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (dstBlockOffset * hd))));
    for (int d = 0; d < (int)(hd); ++d) {
      unsigned int kSrcIdx = (kSrcBase + ((unsigned int)(d)));
      unsigned int kDstIdx = (kDstBase + ((unsigned int)(d)));
      unsigned int vSrcIdx = (vSrcBase + ((unsigned int)(d)));
      unsigned int vDstIdx = (vDstBase + ((unsigned int)(d)));
      __half kVal = kvCache[kSrcIdx];
      __half vVal = kvCache[vSrcIdx];
      kvCache[kDstIdx] = kVal;
      kvCache[vDstIdx] = vVal;
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_f16(const __half* a, const __half* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, const float* alpha, const float* beta, __half* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileA[256];
  __shared__ float tileB[256];
  float acc = __uint_as_float(0x00000000u);
  unsigned int tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    unsigned int tileIterU32 = ((unsigned int)(tileIter));
    unsigned int tileBase = (tileIterU32 * tileSize);
    unsigned int aK = (tileBase + tidX);
    unsigned int bK = (tileBase + tidY);
    unsigned int tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    unsigned int aIdx = ((row * kDim) + aK);
    unsigned int bIdx = ((bK * nDim) + col);
    __half aValF16 = ((aLoadValid) ? (a[aIdx]) : (__float2half(0.000000f)));
    __half bValF16 = ((bLoadValid) ? (b[bIdx]) : (__float2half(0.000000f)));
    float aTileVal = ((float)(aValF16));
    float bTileVal = ((float)(bValF16));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    __syncthreads();
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      unsigned int kkU32 = ((unsigned int)(kk));
      unsigned int tileAIdx = ((tidY * tileSize) + kkU32);
      unsigned int tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    __syncthreads();
  }
  float finalAcc = acc;
  if (valid) {
    unsigned int cIdx = ((row * nDim) + col);
    __half cOldF16 = c[cIdx];
    float cOld = ((float)(cOldF16));
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    __half resultF16 = ((__half)(result));
    c[cIdx] = resultF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_f32_out = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_f16_f32_out(const __half* a, const __half* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, const float* alpha, const float* beta, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileA[256];
  __shared__ float tileB[256];
  float acc = __uint_as_float(0x00000000u);
  unsigned int tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    unsigned int tileIterU32 = ((unsigned int)(tileIter));
    unsigned int tileBase = (tileIterU32 * tileSize);
    unsigned int aK = (tileBase + tidX);
    unsigned int bK = (tileBase + tidY);
    unsigned int tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    unsigned int aIdx = ((row * kDim) + aK);
    unsigned int bIdx = ((bK * nDim) + col);
    __half aValF16 = ((aLoadValid) ? (a[aIdx]) : (__float2half(0.000000f)));
    __half bValF16 = ((bLoadValid) ? (b[bIdx]) : (__float2half(0.000000f)));
    float aTileVal = ((float)(aValF16));
    float bTileVal = ((float)(bValF16));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    __syncthreads();
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      unsigned int kkU32 = ((unsigned int)(kk));
      unsigned int tileAIdx = ((tidY * tileSize) + kkU32);
      unsigned int tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    __syncthreads();
  }
  float finalAcc = acc;
  if (valid) {
    unsigned int cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_f32(const float* a, const float* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, const float* alpha, const float* beta, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileA[256];
  __shared__ float tileB[256];
  float acc = __uint_as_float(0x00000000u);
  unsigned int tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    unsigned int tileIterU32 = ((unsigned int)(tileIter));
    unsigned int tileBase = (tileIterU32 * tileSize);
    unsigned int aK = (tileBase + tidX);
    unsigned int bK = (tileBase + tidY);
    unsigned int tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    unsigned int aIdx = ((row * kDim) + aK);
    unsigned int bIdx = ((bK * nDim) + col);
    float aTileVal = ((aLoadValid) ? (a[aIdx]) : (__uint_as_float(0x00000000u)));
    float bTileVal = ((bLoadValid) ? (b[bIdx]) : (__uint_as_float(0x00000000u)));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    __syncthreads();
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      unsigned int kkU32 = ((unsigned int)(kk));
      unsigned int tileAIdx = ((tidY * tileSize) + kkU32);
      unsigned int tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    __syncthreads();
  }
  float finalAcc = acc;
  if (valid) {
    unsigned int cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_backward_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_backward_f32(const float* gradC, const float* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, float* gradA) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileGradC[256];
  __shared__ float tileBT[256];
  if (valid) {
    float acc = __uint_as_float(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      unsigned int gradCIdx = ((row * nDim) + jIter);
      unsigned int bIdx = ((col * nDim) + jIter);
      float gradCVal = gradC[gradCIdx];
      float bVal = b[bIdx];
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    unsigned int gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAcc;
  }
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_backward_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_backward_f16(const __half* gradC, const __half* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, __half* gradA) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileGradC[256];
  __shared__ float tileBT[256];
  if (valid) {
    float acc = __uint_as_float(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      unsigned int gradCIdx = ((row * nDim) + jIter);
      unsigned int bIdx = ((col * nDim) + jIter);
      __half gradCValF16 = gradC[gradCIdx];
      __half bValF16 = b[bIdx];
      float gradCVal = ((float)(gradCValF16));
      float bVal = ((float)(bValF16));
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    __half finalAccF16 = ((__half)(finalAcc));
    unsigned int gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAccF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_scale_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_multi_tensor_scale_f32(const float* _input, const float* scale, const unsigned int* totalElements, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float s = scale[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x * s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_scale_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_multi_tensor_scale_f16(const __half* _input, const float* scale, const unsigned int* totalElements, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float s = scale[0u];
  bool valid = (gid < total);
  if (valid) {
    __half xF16 = _input[gid];
    float x = ((float)(xF16));
    float resultF32 = (x * s);
    __half result = ((__half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_add_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_multi_tensor_add_f32(const float* _input, const float* value, const unsigned int* totalElements, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float v = value[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x + v);
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_add_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_multi_tensor_add_f16(const __half* _input, const float* value, const unsigned int* totalElements, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float v = value[0u];
  bool valid = (gid < total);
  if (valid) {
    __half xF16 = _input[gid];
    float x = ((float)(xF16));
    float resultF32 = (x + v);
    __half result = ((__half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_copy_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_multi_tensor_copy_f32(const float* _input, const unsigned int* totalElements, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    _output[gid] = x;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_copy_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_multi_tensor_copy_f16(const __half* _input, const unsigned int* totalElements, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  bool valid = (gid < total);
  if (valid) {
    __half x = _input[gid];
    _output[gid] = x;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_l2_norm_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_multi_tensor_l2_norm_f32(const float* _input, const unsigned int* totalElements, float* partialSums) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int gid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int globalIdx = ((gid * 256u) + lid);
  unsigned int total = totalElements[0u];
  __shared__ float sumSq[256];
  bool valid = (globalIdx < total);
  float x = ((valid) ? (_input[globalIdx]) : (__uint_as_float(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  __syncthreads();
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    unsigned int neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  __syncthreads();
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    unsigned int neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  __syncthreads();
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    unsigned int neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  __syncthreads();
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    unsigned int neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  __syncthreads();
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    unsigned int neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  __syncthreads();
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    unsigned int neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  __syncthreads();
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    unsigned int neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  __syncthreads();
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    unsigned int neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  __syncthreads();
  if ((lid == 0u)) {
    float partialSum = sumSq[0u];
    partialSums[gid] = partialSum;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_l2_norm_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_multi_tensor_l2_norm_f16(const __half* _input, const unsigned int* totalElements, float* partialSums) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int gid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int globalIdx = ((gid * 256u) + lid);
  unsigned int total = totalElements[0u];
  __shared__ float sumSq[256];
  bool valid = (globalIdx < total);
  __half xF16 = ((valid) ? (_input[globalIdx]) : (__float2half(0.000000f)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  __syncthreads();
  bool shouldReduce_sumSq_128 = (lid < 128u);
  if (shouldReduce_sumSq_128) {
    unsigned int neighborIdx_sumSq_128 = (lid + 128u);
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  __syncthreads();
  bool shouldReduce_sumSq_64 = (lid < 64u);
  if (shouldReduce_sumSq_64) {
    unsigned int neighborIdx_sumSq_64 = (lid + 64u);
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  __syncthreads();
  bool shouldReduce_sumSq_32 = (lid < 32u);
  if (shouldReduce_sumSq_32) {
    unsigned int neighborIdx_sumSq_32 = (lid + 32u);
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  __syncthreads();
  bool shouldReduce_sumSq_16 = (lid < 16u);
  if (shouldReduce_sumSq_16) {
    unsigned int neighborIdx_sumSq_16 = (lid + 16u);
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  __syncthreads();
  bool shouldReduce_sumSq_8 = (lid < 8u);
  if (shouldReduce_sumSq_8) {
    unsigned int neighborIdx_sumSq_8 = (lid + 8u);
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  __syncthreads();
  bool shouldReduce_sumSq_4 = (lid < 4u);
  if (shouldReduce_sumSq_4) {
    unsigned int neighborIdx_sumSq_4 = (lid + 4u);
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  __syncthreads();
  bool shouldReduce_sumSq_2 = (lid < 2u);
  if (shouldReduce_sumSq_2) {
    unsigned int neighborIdx_sumSq_2 = (lid + 2u);
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  __syncthreads();
  bool shouldReduce_sumSq_1 = (lid < 1u);
  if (shouldReduce_sumSq_1) {
    unsigned int neighborIdx_sumSq_1 = (lid + 1u);
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  __syncthreads();
  if ((lid == 0u)) {
    float partialSum = sumSq[0u];
    partialSums[gid] = partialSum;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_clip_grad_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_multi_tensor_clip_grad_f32(const float* totalNorm, const float* maxNorm, const unsigned int* totalElements, float* grad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float tNorm = totalNorm[0u];
  float mNorm = maxNorm[0u];
  float epsilon = __uint_as_float(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (__uint_as_float(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    float g = grad[gid];
    float clipped = (g * clipCoef);
    grad[gid] = clipped;
  }
}

)kernel";
const char* k_hip_navatala_transformer_multi_tensor_clip_grad_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_multi_tensor_clip_grad_f16(const float* totalNorm, const float* maxNorm, const unsigned int* totalElements, __half* grad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float tNorm = totalNorm[0u];
  float mNorm = maxNorm[0u];
  float epsilon = __uint_as_float(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (__uint_as_float(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    __half gF16 = grad[gid];
    float g = ((float)(gF16));
    float clippedF32 = (g * clipCoef);
    __half clipped = ((__half)(clippedF32));
    grad[gid] = clipped;
  }
}

)kernel";
const char* k_hip_navatala_transformer_top_k_gating_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_top_k_gating_f32(const float* routerLogits, const unsigned int* numTokens, const unsigned int* numExperts, const unsigned int* topK, int* expertIndices, float* expertWeights) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tokenIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int ne = numExperts[0u];
  unsigned int k = topK[0u];
  __shared__ float logitsBuf[256];
  __shared__ float expBuf[256];
  __shared__ int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  unsigned int globalIdx = ((tokenIdx * ne) + lid);
  float logit = ((valid) ? (routerLogits[globalIdx]) : (__uint_as_float(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  __syncthreads();
  bool shouldReduce_max_logitsBuf_128 = (lid < 128u);
  if (shouldReduce_max_logitsBuf_128) {
    unsigned int neighborIdx_max_logitsBuf_128 = (lid + 128u);
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_64 = (lid < 64u);
  if (shouldReduce_max_logitsBuf_64) {
    unsigned int neighborIdx_max_logitsBuf_64 = (lid + 64u);
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_32 = (lid < 32u);
  if (shouldReduce_max_logitsBuf_32) {
    unsigned int neighborIdx_max_logitsBuf_32 = (lid + 32u);
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_16 = (lid < 16u);
  if (shouldReduce_max_logitsBuf_16) {
    unsigned int neighborIdx_max_logitsBuf_16 = (lid + 16u);
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_8 = (lid < 8u);
  if (shouldReduce_max_logitsBuf_8) {
    unsigned int neighborIdx_max_logitsBuf_8 = (lid + 8u);
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_4 = (lid < 4u);
  if (shouldReduce_max_logitsBuf_4) {
    unsigned int neighborIdx_max_logitsBuf_4 = (lid + 4u);
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_2 = (lid < 2u);
  if (shouldReduce_max_logitsBuf_2) {
    unsigned int neighborIdx_max_logitsBuf_2 = (lid + 2u);
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_1 = (lid < 1u);
  if (shouldReduce_max_logitsBuf_1) {
    unsigned int neighborIdx_max_logitsBuf_1 = (lid + 1u);
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  __syncthreads();
  float maxLogit = logitsBuf[0u];
  float originalLogit = ((valid) ? (routerLogits[globalIdx]) : (__uint_as_float(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (__uint_as_float(0x00000000u)));
  expBuf[lid] = expMasked;
  __syncthreads();
  bool shouldReduce_sum_expBuf_128 = (lid < 128u);
  if (shouldReduce_sum_expBuf_128) {
    unsigned int neighborIdx_sum_expBuf_128 = (lid + 128u);
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_64 = (lid < 64u);
  if (shouldReduce_sum_expBuf_64) {
    unsigned int neighborIdx_sum_expBuf_64 = (lid + 64u);
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_32 = (lid < 32u);
  if (shouldReduce_sum_expBuf_32) {
    unsigned int neighborIdx_sum_expBuf_32 = (lid + 32u);
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_16 = (lid < 16u);
  if (shouldReduce_sum_expBuf_16) {
    unsigned int neighborIdx_sum_expBuf_16 = (lid + 16u);
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_8 = (lid < 8u);
  if (shouldReduce_sum_expBuf_8) {
    unsigned int neighborIdx_sum_expBuf_8 = (lid + 8u);
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_4 = (lid < 4u);
  if (shouldReduce_sum_expBuf_4) {
    unsigned int neighborIdx_sum_expBuf_4 = (lid + 4u);
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_2 = (lid < 2u);
  if (shouldReduce_sum_expBuf_2) {
    unsigned int neighborIdx_sum_expBuf_2 = (lid + 2u);
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_1 = (lid < 1u);
  if (shouldReduce_sum_expBuf_1) {
    unsigned int neighborIdx_sum_expBuf_1 = (lid + 1u);
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  __syncthreads();
  float sumExp = expBuf[0u];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  indexBuf[lid] = lidI32;
  __syncthreads();
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == 0u)) {
      int bestIdx = 0;
      float bestVal = __uint_as_float(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      unsigned int kIterU32 = kIter;
      unsigned int outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValFinal;
      unsigned int bestIdxU32 = ((unsigned int)(bestIdxFinal));
      logitsBuf[bestIdxU32] = __uint_as_float(0xf149f2cau);
    }
    __syncthreads();
  }
}

)kernel";
const char* k_hip_navatala_transformer_top_k_gating_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_top_k_gating_f16(const __half* routerLogits, const unsigned int* numTokens, const unsigned int* numExperts, const unsigned int* topK, int* expertIndices, __half* expertWeights) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tokenIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int ne = numExperts[0u];
  unsigned int k = topK[0u];
  __shared__ float logitsBuf[256];
  __shared__ float expBuf[256];
  __shared__ int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  unsigned int globalIdx = ((tokenIdx * ne) + lid);
  __half logitF16 = ((valid) ? (routerLogits[globalIdx]) : (__float2half(0.000000f)));
  float logit = ((expertValid) ? (((float)(logitF16))) : (__uint_as_float(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  __syncthreads();
  bool shouldReduce_max_logitsBuf_128 = (lid < 128u);
  if (shouldReduce_max_logitsBuf_128) {
    unsigned int neighborIdx_max_logitsBuf_128 = (lid + 128u);
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_64 = (lid < 64u);
  if (shouldReduce_max_logitsBuf_64) {
    unsigned int neighborIdx_max_logitsBuf_64 = (lid + 64u);
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_32 = (lid < 32u);
  if (shouldReduce_max_logitsBuf_32) {
    unsigned int neighborIdx_max_logitsBuf_32 = (lid + 32u);
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_16 = (lid < 16u);
  if (shouldReduce_max_logitsBuf_16) {
    unsigned int neighborIdx_max_logitsBuf_16 = (lid + 16u);
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_8 = (lid < 8u);
  if (shouldReduce_max_logitsBuf_8) {
    unsigned int neighborIdx_max_logitsBuf_8 = (lid + 8u);
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_4 = (lid < 4u);
  if (shouldReduce_max_logitsBuf_4) {
    unsigned int neighborIdx_max_logitsBuf_4 = (lid + 4u);
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_2 = (lid < 2u);
  if (shouldReduce_max_logitsBuf_2) {
    unsigned int neighborIdx_max_logitsBuf_2 = (lid + 2u);
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  __syncthreads();
  bool shouldReduce_max_logitsBuf_1 = (lid < 1u);
  if (shouldReduce_max_logitsBuf_1) {
    unsigned int neighborIdx_max_logitsBuf_1 = (lid + 1u);
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  __syncthreads();
  float maxLogit = logitsBuf[0u];
  __half originalLogitF16 = ((valid) ? (routerLogits[globalIdx]) : (__float2half(0.000000f)));
  float originalLogit = ((expertValid) ? (((float)(originalLogitF16))) : (__uint_as_float(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (__uint_as_float(0x00000000u)));
  expBuf[lid] = expMasked;
  __syncthreads();
  bool shouldReduce_sum_expBuf_128 = (lid < 128u);
  if (shouldReduce_sum_expBuf_128) {
    unsigned int neighborIdx_sum_expBuf_128 = (lid + 128u);
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_64 = (lid < 64u);
  if (shouldReduce_sum_expBuf_64) {
    unsigned int neighborIdx_sum_expBuf_64 = (lid + 64u);
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_32 = (lid < 32u);
  if (shouldReduce_sum_expBuf_32) {
    unsigned int neighborIdx_sum_expBuf_32 = (lid + 32u);
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_16 = (lid < 16u);
  if (shouldReduce_sum_expBuf_16) {
    unsigned int neighborIdx_sum_expBuf_16 = (lid + 16u);
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_8 = (lid < 8u);
  if (shouldReduce_sum_expBuf_8) {
    unsigned int neighborIdx_sum_expBuf_8 = (lid + 8u);
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_4 = (lid < 4u);
  if (shouldReduce_sum_expBuf_4) {
    unsigned int neighborIdx_sum_expBuf_4 = (lid + 4u);
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_2 = (lid < 2u);
  if (shouldReduce_sum_expBuf_2) {
    unsigned int neighborIdx_sum_expBuf_2 = (lid + 2u);
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  __syncthreads();
  bool shouldReduce_sum_expBuf_1 = (lid < 1u);
  if (shouldReduce_sum_expBuf_1) {
    unsigned int neighborIdx_sum_expBuf_1 = (lid + 1u);
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  __syncthreads();
  float sumExp = expBuf[0u];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  int lidI32_2 = ((int)(lid));
  indexBuf[lid] = lidI32_2;
  __syncthreads();
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == 0u)) {
      int bestIdx = 0;
      float bestVal = __uint_as_float(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      unsigned int kIterU32 = kIter;
      unsigned int outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      __half bestValF16 = ((__half)(bestValFinal));
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValF16;
      unsigned int bestIdxU32 = ((unsigned int)(bestIdxFinal));
      logitsBuf[bestIdxU32] = __uint_as_float(0xf149f2cau);
    }
    __syncthreads();
  }
}

)kernel";
const char* k_hip_navatala_transformer_expert_capacity_mask_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_expert_capacity_mask_f32(const int* expertIndices, const unsigned int* numTokens, const unsigned int* numExperts, const unsigned int* topK, const unsigned int* capacity, float* mask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int ne = numExperts[0u];
  unsigned int k = topK[0u];
  unsigned int cap = capacity[0u];
  __shared__ unsigned int expertCounts[256];
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  if ((lid < ne)) {
    expertCounts[lid] = 0u;
  }
  __syncthreads();
  unsigned int totalElements = (nt * k);
  bool valid = (gid < totalElements);
  if (valid) {
    int expertIdx = expertIndices[gid];
    unsigned int expertIdxU32 = ((unsigned int)(expertIdx));
    bool expertIdxValid = (expertIdxU32 < ne);
    if (expertIdxValid) {
      unsigned int prevCount = atomicAdd(&(expertCounts[expertIdxU32]), 1u);
      bool withinCapacity = (prevCount < cap);
      float maskVal = ((withinCapacity) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
      mask[gid] = maskVal;
    } else {
      mask[gid] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_expert_capacity_mask_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_expert_capacity_mask_f16(const int* expertIndices, const unsigned int* numTokens, const unsigned int* numExperts, const unsigned int* topK, const unsigned int* capacity, __half* mask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int ne = numExperts[0u];
  unsigned int k = topK[0u];
  unsigned int cap = capacity[0u];
  __shared__ unsigned int expertCounts[256];
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  if ((lid < ne)) {
    expertCounts[lid] = 0u;
  }
  __syncthreads();
  unsigned int totalElements = (nt * k);
  bool valid = (gid < totalElements);
  if (valid) {
    int expertIdx = expertIndices[gid];
    unsigned int expertIdxU32 = ((unsigned int)(expertIdx));
    bool expertIdxValid = (expertIdxU32 < ne);
    if (expertIdxValid) {
      unsigned int prevCount = atomicAdd(&(expertCounts[expertIdxU32]), 1u);
      bool withinCapacity = (prevCount < cap);
      __half maskVal = ((withinCapacity) ? (__float2half(1.000000f)) : (__float2half(0.000000f)));
      mask[gid] = maskVal;
    } else {
      mask[gid] = __float2half(0.000000f);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_permute_tokens_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_permute_tokens_f32(const float* hiddenStates, const int* expertIndices, const unsigned int* expertOffsets, unsigned int* expertCounters, const unsigned int* numTokens, const unsigned int* hiddenSize, const unsigned int* numExperts, float* permutedStates, int* tokenIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tokenIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int ne = numExperts[0u];
  __shared__ unsigned int sharedDestTokenIdx[1];
  bool tokenValid = (tokenIdx < nt);
  if (tokenValid) {
    int expertIdx = expertIndices[tokenIdx];
    unsigned int expertIdxU32 = ((unsigned int)(expertIdx));
    bool expertValid = (expertIdxU32 < ne);
    if (expertValid) {
      if ((lid == 0u)) {
        unsigned int localPosition = atomicAdd(&(expertCounters[expertIdxU32]), 1u);
        unsigned int expertOffset = expertOffsets[expertIdxU32];
        unsigned int destTokenIdx = (expertOffset + localPosition);
        sharedDestTokenIdx[0u] = destTokenIdx;
        int tokenIdxI32 = ((int)(tokenIdx));
        tokenIndices[destTokenIdx] = tokenIdxI32;
      }
      __syncthreads();
      unsigned int destTokenIdx = sharedDestTokenIdx[0u];
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        unsigned int hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          unsigned int srcIdx = ((tokenIdx * hs) + hIdx);
          unsigned int dstIdx = ((destTokenIdx * hs) + hIdx);
          float val = hiddenStates[srcIdx];
          permutedStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_permute_tokens_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_permute_tokens_f16(const __half* hiddenStates, const int* expertIndices, const unsigned int* expertOffsets, unsigned int* expertCounters, const unsigned int* numTokens, const unsigned int* hiddenSize, const unsigned int* numExperts, __half* permutedStates, int* tokenIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tokenIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int ne = numExperts[0u];
  __shared__ unsigned int sharedDestTokenIdx[1];
  bool tokenValid = (tokenIdx < nt);
  if (tokenValid) {
    int expertIdx = expertIndices[tokenIdx];
    unsigned int expertIdxU32 = ((unsigned int)(expertIdx));
    bool expertValid = (expertIdxU32 < ne);
    if (expertValid) {
      if ((lid == 0u)) {
        unsigned int localPosition = atomicAdd(&(expertCounters[expertIdxU32]), 1u);
        unsigned int expertOffset = expertOffsets[expertIdxU32];
        unsigned int destTokenIdx = (expertOffset + localPosition);
        sharedDestTokenIdx[0u] = destTokenIdx;
        int tokenIdxI32 = ((int)(tokenIdx));
        tokenIndices[destTokenIdx] = tokenIdxI32;
      }
      __syncthreads();
      unsigned int destTokenIdx = sharedDestTokenIdx[0u];
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        unsigned int hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          unsigned int srcIdx = ((tokenIdx * hs) + hIdx);
          unsigned int dstIdx = ((destTokenIdx * hs) + hIdx);
          __half val = hiddenStates[srcIdx];
          permutedStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_unpermute_tokens_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_unpermute_tokens_f32(const float* permutedStates, const int* tokenIndices, const unsigned int* numTokens, const unsigned int* hiddenSize, float* hiddenStates) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int permutedIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int hs = hiddenSize[0u];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    unsigned int origIdx = ((unsigned int)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        unsigned int hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          unsigned int srcIdx = ((permutedIdx * hs) + hIdx);
          unsigned int dstIdx = ((origIdx * hs) + hIdx);
          float val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_unpermute_tokens_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_unpermute_tokens_f16(const __half* permutedStates, const int* tokenIndices, const unsigned int* numTokens, const unsigned int* hiddenSize, __half* hiddenStates) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int permutedIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int nt = numTokens[0u];
  unsigned int hs = hiddenSize[0u];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    unsigned int origIdx = ((unsigned int)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        unsigned int hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          unsigned int srcIdx = ((permutedIdx * hs) + hIdx);
          unsigned int dstIdx = ((origIdx * hs) + hIdx);
          __half val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_pad_sequence_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_pad_sequence_f32(const float* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int seqHidden = (msl * hs);
    unsigned int batchIdx = (gid / seqHidden);
    unsigned int remainder = (gid % seqHidden);
    unsigned int seqIdx = (remainder / hs);
    unsigned int hiddenIdx = (remainder % hs);
    unsigned int seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      unsigned int seqOffset = seqOffsets[batchIdx];
      unsigned int inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_pad_sequence_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_pad_sequence_f16(const __half* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int seqHidden = (msl * hs);
    unsigned int batchIdx = (gid / seqHidden);
    unsigned int remainder = (gid % seqHidden);
    unsigned int seqIdx = (remainder / hs);
    unsigned int hiddenIdx = (remainder % hs);
    unsigned int seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      unsigned int seqOffset = seqOffsets[batchIdx];
      unsigned int inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      __half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = __float2half(0.000000f);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_unpad_sequence_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_unpad_sequence_f32(const float* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, const unsigned int* totalTokens, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int tt = totalTokens[0u];
  unsigned int totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int tokenIdx = (gid / hs);
    unsigned int hiddenIdx = (gid % hs);
    unsigned int foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      unsigned int currOffset = seqOffsets[b];
      unsigned int nextB = (b + 1u);
      unsigned int nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    unsigned int batchIdx = foundBatchIdx;
    unsigned int batchOffset = seqOffsets[batchIdx];
    unsigned int localTokenIdx = (tokenIdx - batchOffset);
    unsigned int batchStride = (msl * hs);
    unsigned int inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_unpad_sequence_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_unpad_sequence_f16(const __half* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, const unsigned int* totalTokens, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int tt = totalTokens[0u];
  unsigned int totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int tokenIdx = (gid / hs);
    unsigned int hiddenIdx = (gid % hs);
    unsigned int foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      unsigned int currOffset = seqOffsets[b];
      unsigned int nextB = (b + 1u);
      unsigned int nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    unsigned int batchIdx = foundBatchIdx;
    unsigned int batchOffset = seqOffsets[batchIdx];
    unsigned int localTokenIdx = (tokenIdx - batchOffset);
    unsigned int batchStride = (msl * hs);
    unsigned int inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    __half val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_swizzle_layout_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_swizzle_layout_f32(const float* _input, const unsigned int* rows, const unsigned int* cols, const unsigned int* tileSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int r = rows[0u];
  unsigned int c = cols[0u];
  unsigned int ts = tileSize[0u];
  unsigned int total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / c);
    unsigned int col = (gid % c);
    float val = _input[gid];
    unsigned int tileRow = (row / ts);
    unsigned int tileCol = (col / ts);
    unsigned int localRow = (row % ts);
    unsigned int localCol = (col % ts);
    unsigned int tilesPerRow = ((c + (ts - 1u)) / ts);
    unsigned int tileIdx = ((tileRow * tilesPerRow) + tileCol);
    unsigned int tileElements = (ts * ts);
    unsigned int localIdx = ((localRow * ts) + localCol);
    unsigned int outIdx = ((tileIdx * tileElements) + localIdx);
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_swizzle_layout_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_swizzle_layout_f16(const __half* _input, const unsigned int* rows, const unsigned int* cols, const unsigned int* tileSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int r = rows[0u];
  unsigned int c = cols[0u];
  unsigned int ts = tileSize[0u];
  unsigned int total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / c);
    unsigned int col = (gid % c);
    __half val = _input[gid];
    unsigned int tileRow = (row / ts);
    unsigned int tileCol = (col / ts);
    unsigned int localRow = (row % ts);
    unsigned int localCol = (col % ts);
    unsigned int tilesPerRow = ((c + (ts - 1u)) / ts);
    unsigned int tileIdx = ((tileRow * tilesPerRow) + tileCol);
    unsigned int tileElements = (ts * ts);
    unsigned int localIdx = ((localRow * ts) + localCol);
    unsigned int outIdx = ((tileIdx * tileElements) + localIdx);
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_hadamard_transform_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_hadamard_transform_f32(const float* _input, const unsigned int* batchSize, const unsigned int* vectorSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int vs = vectorSize[0u];
  __shared__ float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  unsigned int globalIdx = ((batchIdx * vs) + lid);
  if ((batchValid && threadValid)) {
    float val = _input[globalIdx];
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int stride1 = 1u;
  unsigned int partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == 0u);
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  __syncthreads();
  sharedBuf[lid] = newVal1;
  __syncthreads();
  unsigned int stride2 = 2u;
  unsigned int partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == 0u);
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  __syncthreads();
  sharedBuf[lid] = newVal2;
  __syncthreads();
  unsigned int stride3 = 4u;
  unsigned int partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == 0u);
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  __syncthreads();
  sharedBuf[lid] = newVal3;
  __syncthreads();
  unsigned int stride4 = 8u;
  unsigned int partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == 0u);
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  __syncthreads();
  sharedBuf[lid] = newVal4;
  __syncthreads();
  unsigned int stride5 = 16u;
  unsigned int partner5 = (lid ^ stride5);
  float myVal5 = sharedBuf[lid];
  float partnerVal5 = sharedBuf[partner5];
  bool lowBit5 = ((lid & stride5) == 0u);
  float newVal5 = ((lowBit5) ? ((myVal5 + partnerVal5)) : ((partnerVal5 - myVal5)));
  __syncthreads();
  sharedBuf[lid] = newVal5;
  __syncthreads();
  unsigned int stride6 = 32u;
  unsigned int partner6 = (lid ^ stride6);
  float myVal6 = sharedBuf[lid];
  float partnerVal6 = sharedBuf[partner6];
  bool lowBit6 = ((lid & stride6) == 0u);
  float newVal6 = ((lowBit6) ? ((myVal6 + partnerVal6)) : ((partnerVal6 - myVal6)));
  __syncthreads();
  sharedBuf[lid] = newVal6;
  __syncthreads();
  unsigned int stride7 = 64u;
  unsigned int partner7 = (lid ^ stride7);
  float myVal7 = sharedBuf[lid];
  float partnerVal7 = sharedBuf[partner7];
  bool lowBit7 = ((lid & stride7) == 0u);
  float newVal7 = ((lowBit7) ? ((myVal7 + partnerVal7)) : ((partnerVal7 - myVal7)));
  __syncthreads();
  sharedBuf[lid] = newVal7;
  __syncthreads();
  unsigned int stride8 = 128u;
  unsigned int partner8 = (lid ^ stride8);
  float myVal8 = sharedBuf[lid];
  float partnerVal8 = sharedBuf[partner8];
  bool lowBit8 = ((lid & stride8) == 0u);
  float newVal8 = ((lowBit8) ? ((myVal8 + partnerVal8)) : ((partnerVal8 - myVal8)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal8 / scale);
  if ((batchValid && threadValid)) {
    _output[globalIdx] = normalized;
  }
}

)kernel";
const char* k_hip_navatala_transformer_hadamard_transform_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_hadamard_transform_f16(const __half* _input, const unsigned int* batchSize, const unsigned int* vectorSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int vs = vectorSize[0u];
  __shared__ float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  unsigned int globalIdx = ((batchIdx * vs) + lid);
  if ((batchValid && threadValid)) {
    __half valF16 = _input[globalIdx];
    float val = ((float)(valF16));
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int stride1 = 1u;
  unsigned int partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == 0u);
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  __syncthreads();
  sharedBuf[lid] = newVal1;
  __syncthreads();
  unsigned int stride2 = 2u;
  unsigned int partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == 0u);
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  __syncthreads();
  sharedBuf[lid] = newVal2;
  __syncthreads();
  unsigned int stride3 = 4u;
  unsigned int partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == 0u);
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  __syncthreads();
  sharedBuf[lid] = newVal3;
  __syncthreads();
  unsigned int stride4 = 8u;
  unsigned int partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == 0u);
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal4 / scale);
  __half resultF16 = ((__half)(normalized));
  if ((batchValid && threadValid)) {
    _output[globalIdx] = resultF16;
  }
}

)kernel";
const char* k_hip_navatala_transformer_permute_axes_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_permute_axes_f32(const float* _input, const unsigned int* dim0, const unsigned int* dim1, const unsigned int* dim2, const unsigned int* dim3, const unsigned int* perm0, const unsigned int* perm1, const unsigned int* perm2, const unsigned int* perm3, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d0 = dim0[0u];
  unsigned int d1 = dim1[0u];
  unsigned int d2 = dim2[0u];
  unsigned int d3 = dim3[0u];
  unsigned int p0 = perm0[0u];
  unsigned int p1 = perm1[0u];
  unsigned int p2 = perm2[0u];
  unsigned int p3 = perm3[0u];
  unsigned int total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int stride0 = (d1 * (d2 * d3));
    unsigned int stride1 = (d2 * d3);
    unsigned int stride2 = d3;
    unsigned int stride3 = 1u;
    unsigned int i0 = (gid / stride0);
    unsigned int rem0 = (gid % stride0);
    unsigned int i1 = (rem0 / stride1);
    unsigned int rem1 = (rem0 % stride1);
    unsigned int i2 = (rem1 / stride2);
    unsigned int i3 = (rem1 % stride2);
    float val = _input[gid];
    unsigned int od0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    unsigned int od1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    unsigned int od2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    unsigned int od3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    unsigned int oi0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    unsigned int oi1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    unsigned int oi2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    unsigned int oi3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    unsigned int ostride0 = (od1 * (od2 * od3));
    unsigned int ostride1 = (od2 * od3);
    unsigned int ostride2 = od3;
    unsigned int outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_permute_axes_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_permute_axes_f16(const __half* _input, const unsigned int* dim0, const unsigned int* dim1, const unsigned int* dim2, const unsigned int* dim3, const unsigned int* perm0, const unsigned int* perm1, const unsigned int* perm2, const unsigned int* perm3, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d0 = dim0[0u];
  unsigned int d1 = dim1[0u];
  unsigned int d2 = dim2[0u];
  unsigned int d3 = dim3[0u];
  unsigned int p0 = perm0[0u];
  unsigned int p1 = perm1[0u];
  unsigned int p2 = perm2[0u];
  unsigned int p3 = perm3[0u];
  unsigned int total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int stride0 = (d1 * (d2 * d3));
    unsigned int stride1 = (d2 * d3);
    unsigned int stride2 = d3;
    unsigned int i0 = (gid / stride0);
    unsigned int rem0 = (gid % stride0);
    unsigned int i1 = (rem0 / stride1);
    unsigned int rem1 = (rem0 % stride1);
    unsigned int i2 = (rem1 / stride2);
    unsigned int i3 = (rem1 % stride2);
    __half val = _input[gid];
    unsigned int od0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    unsigned int od1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    unsigned int od2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    unsigned int od3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    unsigned int oi0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    unsigned int oi1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    unsigned int oi2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    unsigned int oi3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    unsigned int ostride0 = (od1 * (od2 * od3));
    unsigned int ostride1 = (od2 * od3);
    unsigned int ostride2 = od3;
    unsigned int outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_pack_sequences_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_pack_sequences_f32(const float* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, const unsigned int* totalTokens, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int tt = totalTokens[0u];
  unsigned int totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int tokenIdx = (gid / hs);
    unsigned int hiddenIdx = (gid % hs);
    unsigned int foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      unsigned int currOffset = seqOffsets[b];
      unsigned int nextB = (b + 1u);
      unsigned int nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    unsigned int batchIdx = foundBatchIdx;
    unsigned int batchOffset = seqOffsets[batchIdx];
    unsigned int localTokenIdx = (tokenIdx - batchOffset);
    unsigned int batchStride = (msl * hs);
    unsigned int inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_pack_sequences_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_pack_sequences_f16(const __half* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, const unsigned int* totalTokens, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int tt = totalTokens[0u];
  unsigned int totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int tokenIdx = (gid / hs);
    unsigned int hiddenIdx = (gid % hs);
    unsigned int foundBatchIdx = 0u;
    for (int b = 0; b < (int)(bs); ++b) {
      unsigned int currOffset = seqOffsets[b];
      unsigned int nextB = (b + 1u);
      unsigned int nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    unsigned int batchIdx = foundBatchIdx;
    unsigned int batchOffset = seqOffsets[batchIdx];
    unsigned int localTokenIdx = (tokenIdx - batchOffset);
    unsigned int batchStride = (msl * hs);
    unsigned int inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    __half val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_transformer_unpack_sequences_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_unpack_sequences_f32(const float* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int seqHidden = (msl * hs);
    unsigned int batchIdx = (gid / seqHidden);
    unsigned int remainder = (gid % seqHidden);
    unsigned int seqIdx = (remainder / hs);
    unsigned int hiddenIdx = (remainder % hs);
    unsigned int seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      unsigned int seqOffset = seqOffsets[batchIdx];
      unsigned int inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = __uint_as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_unpack_sequences_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_unpack_sequences_f16(const __half* _input, const unsigned int* seqLengths, const unsigned int* seqOffsets, const unsigned int* batchSize, const unsigned int* maxSeqLen, const unsigned int* hiddenSize, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int msl = maxSeqLen[0u];
  unsigned int hs = hiddenSize[0u];
  unsigned int totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    unsigned int seqHidden = (msl * hs);
    unsigned int batchIdx = (gid / seqHidden);
    unsigned int remainder = (gid % seqHidden);
    unsigned int seqIdx = (remainder / hs);
    unsigned int hiddenIdx = (remainder % hs);
    unsigned int seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      unsigned int seqOffset = seqOffsets[batchIdx];
      unsigned int inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      __half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = __float2half(0.000000f);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_generate_position_ids_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_generate_position_ids_f32(const unsigned int* batchSize, const unsigned int* seqLen, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int total = (bs * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int pos = (gid % sl);
    float posF32 = ((float)(pos));
    _output[gid] = posF32;
  }
}

)kernel";
const char* k_hip_navatala_transformer_generate_position_ids_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_generate_position_ids_u32(const unsigned int* batchSize, const unsigned int* seqLen, unsigned int* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int bs = batchSize[0u];
  unsigned int sl = seqLen[0u];
  unsigned int total = (bs * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int pos = (gid % sl);
    _output[gid] = pos;
  }
}

)kernel";
const char* k_hip_navatala_transformer_generate_causal_mask_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_generate_causal_mask_f32(const unsigned int* seqLen, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int sl = seqLen[0u];
  unsigned int total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / sl);
    unsigned int col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = __uint_as_float(0x00000000u);
    } else {
      _output[gid] = __uint_as_float(0xce6e6b28u);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_generate_causal_mask_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_generate_causal_mask_f16(const unsigned int* seqLen, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int sl = seqLen[0u];
  unsigned int total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / sl);
    unsigned int col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = __float2half(0.000000f);
    } else {
      _output[gid] = __float2half(-65000.000000f);
    }
  }
}

)kernel";
const char* k_hip_navatala_transformer_precompute_rotary_emb_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_transformer_precompute_rotary_emb_f32(const unsigned int* maxSeqLen, const unsigned int* headDim, const float* base, float* cosOutput, float* sinOutput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int msl = maxSeqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int pos = (gid / halfHd);
    unsigned int dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosVal = cos(theta);
    float sinVal = sin(theta);
    cosOutput[gid] = cosVal;
    sinOutput[gid] = sinVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_precompute_rotary_emb_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_precompute_rotary_emb_f16(const unsigned int* maxSeqLen, const unsigned int* headDim, const float* base, __half* cosOutput, __half* sinOutput) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int msl = maxSeqLen[0u];
  unsigned int hd = headDim[0u];
  float ropeBase = base[0u];
  unsigned int halfHd = (hd / 2u);
  unsigned int total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int pos = (gid / halfHd);
    unsigned int dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (__uint_as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosValF32 = cos(theta);
    float sinValF32 = sin(theta);
    __half cosVal = ((__half)(cosValF32));
    __half sinVal = ((__half)(sinValF32));
    cosOutput[gid] = cosVal;
    sinOutput[gid] = sinVal;
  }
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(64) void navatala_transformer_tiled_gemm_f16_mfma(const __half* a, const __half* b, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  unsigned int tileRow = (((unsigned int)((int)(blockIdx.y))) * 32u);
  unsigned int tileCol = (((unsigned int)((int)(blockIdx.x))) * 32u);
  navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(a, tileRow, 0u, kStrideVal);
  navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(b, 0u, tileCol, nStrideVal);
  navatala_mfma_acc_f32_32x32x8 acc0 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc1 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc0, 0, 0, 0);
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc1, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_k_loop = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(64) void navatala_transformer_tiled_gemm_f16_mfma_k_loop(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  unsigned int tileRow = (((unsigned int)((int)(blockIdx.y))) * 32u);
  unsigned int tileCol = (((unsigned int)((int)(blockIdx.x))) * 32u);
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int kBase = (kTileU32 * 8u);
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(a, tileRow, kBase, kStrideVal);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(b, kBase, tileCol, nStrideVal);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_direct = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_direct(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int kBase = (kTileU32 * 8u);
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(a, tileRow, kBase, kStrideVal);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(b, kBase, tileCol, nStrideVal);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_shared(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int waveRowLocal = (waveRowGroup * 32u);
  unsigned int waveColLocal = (waveColGroup * 32u);
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  __shared__ __half tileA[512];
  __shared__ __half tileB[512];
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int kBase = (kTileU32 * 8u);
    for (int loadIter = 0; loadIter < (int)(2u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 8u);
      unsigned int aColLocal = (_flat % 8u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdx = ((aGlobalRow * kStrideVal) + aGlobalCol);
      __half aVal = a[aIdx];
      unsigned int bRowLocal = (_flat / 64u);
      unsigned int bColLocal = (_flat % 64u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdx = ((bGlobalRow * nStrideVal) + bGlobalCol);
      __half bVal = b[bIdx];
      tileA[_flat] = aVal;
      tileB[_flat] = bVal;
    }
    __syncthreads();
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocal, 0u, 8u);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(tileB, 0u, waveColLocal, 64u);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge(const __half* a, const __half* b, const unsigned int* mDim, const unsigned int* nDim, const unsigned int* kDim, const unsigned int* kTiles, const unsigned int* aStride, const unsigned int* bStride, const unsigned int* cStride, const unsigned int* aBatchStride, const unsigned int* bBatchStride, const unsigned int* cBatchStride, const unsigned int* transA, const unsigned int* transB, const float* alpha, const float* beta, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.z)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int waveRowLocal = (waveRowGroup * 32u);
  unsigned int waveColLocal = (waveColGroup * 32u);
  unsigned int mDimVal = mDim[0u];
  unsigned int nDimVal = nDim[0u];
  unsigned int kDimVal = kDim[0u];
  unsigned int kTilesVal = kTiles[0u];
  unsigned int aStrideVal = aStride[0u];
  unsigned int bStrideVal = bStride[0u];
  unsigned int cStrideVal = cStride[0u];
  unsigned int aBatchStrideVal = aBatchStride[0u];
  unsigned int bBatchStrideVal = bBatchStride[0u];
  unsigned int cBatchStrideVal = cBatchStride[0u];
  unsigned int transAVal = transA[0u];
  unsigned int transBVal = transB[0u];
  bool useTransA = (transAVal != 0u);
  bool useTransB = (transBVal != 0u);
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int aBatchBase = (batchIdx * aBatchStrideVal);
  unsigned int bBatchBase = (batchIdx * bBatchStrideVal);
  unsigned int cBatchBase = (batchIdx * cBatchStrideVal);
  unsigned int cTileColBase = (cBatchBase + tileCol);
  unsigned int cColLimit = (cBatchBase + nDimVal);
  __shared__ __half tileA[512];
  __shared__ __half tileB[512];
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int kBase = (kTileU32 * 8u);
    for (int loadIter = 0; loadIter < (int)(2u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 8u);
      unsigned int aColLocal = (_flat % 8u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdxNN = ((aGlobalRow * aStrideVal) + aGlobalCol);
      unsigned int aIdxT = ((aGlobalCol * aStrideVal) + aGlobalRow);
      unsigned int aIdx = ((useTransA) ? (aIdxT) : (aIdxNN));
      unsigned int aIdxBatch = (aBatchBase + aIdx);
      bool aRowValid = (aGlobalRow < mDimVal);
      bool aColValid = (aGlobalCol < kDimVal);
      bool aLoadValid = (aRowValid && aColValid);
      __half aVal = ((aLoadValid) ? (a[aIdxBatch]) : (__float2half(0.000000f)));
      unsigned int bRowLocal = (_flat / 64u);
      unsigned int bColLocal = (_flat % 64u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdxNN = ((bGlobalRow * bStrideVal) + bGlobalCol);
      unsigned int bIdxT = ((bGlobalCol * bStrideVal) + bGlobalRow);
      unsigned int bIdx = ((useTransB) ? (bIdxT) : (bIdxNN));
      unsigned int bIdxBatch = (bBatchBase + bIdx);
      bool bRowValid = (bGlobalRow < kDimVal);
      bool bColValid = (bGlobalCol < nDimVal);
      bool bLoadValid = (bRowValid && bColValid);
      __half bVal = ((bLoadValid) ? (b[bIdxBatch]) : (__float2half(0.000000f)));
      tileA[_flat] = aVal;
      tileB[_flat] = bVal;
    }
    __syncthreads();
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocal, 0u, 8u);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(tileB, 0u, waveColLocal, 64u);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc, tileRow, cTileColBase, cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int waveRowLocal = (waveRowGroup * 32u);
  unsigned int waveColLocal = (waveColGroup * 32u);
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  __shared__ __half tileA[512];
  __shared__ __half tileB[512];
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int kBase = (kTileU32 * 8u);
    for (int loadIter = 0; loadIter < (int)(2u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 8u);
      unsigned int aColLocal = (_flat % 8u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdx = ((aGlobalRow * kStrideVal) + aGlobalCol);
      __half aVal = a[aIdx];
      unsigned int bRowLocal = (_flat / 64u);
      unsigned int bColLocal = (_flat % 64u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdx = ((bGlobalRow * nStrideVal) + bGlobalCol);
      __half bVal = b[bIdx];
      tileA[_flat] = aVal;
      tileB[_flat] = bVal;
    }
    __syncthreads();
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocal, 0u, 8u);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(tileB, 0u, waveColLocal, 64u);
    __syncthreads();
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int waveRowLocal = (waveRowGroup * 32u);
  unsigned int waveColLocal = (waveColGroup * 32u);
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  __shared__ __half tileA[576];
  __shared__ __half tileB[520];
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int kBase = (kTileU32 * 8u);
    for (int loadIter = 0; loadIter < (int)(2u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 8u);
      unsigned int aColLocal = (_flat % 8u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdx = ((aGlobalRow * kStrideVal) + aGlobalCol);
      unsigned int aSharedIdx = ((aRowLocal * 9u) + aColLocal);
      __half aVal = a[aIdx];
      unsigned int bRowLocal = (_flat / 64u);
      unsigned int bColLocal = (_flat % 64u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdx = ((bGlobalRow * nStrideVal) + bGlobalCol);
      unsigned int bSharedIdx = ((bRowLocal * 65u) + bColLocal);
      __half bVal = b[bIdx];
      tileA[aSharedIdx] = aVal;
      tileB[bSharedIdx] = bVal;
    }
    __syncthreads();
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocal, 0u, 9u);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(tileB, 0u, waveColLocal, 65u);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int waveRowLocal = (waveRowGroup * 32u);
  unsigned int waveColLocal = (waveColGroup * 32u);
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  __shared__ __align__(16) __half panelA[1024]; /* R6 shared panel: GPU.MatrixPanelRole.A, slots=2 */
  __shared__ __align__(16) __half panelB[1024]; /* R6 shared panel: GPU.MatrixPanelRole.B, slots=2 */
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int slot = (kTileU32 % 2u);
    unsigned int kBase = (kTileU32 * 8u);
    bool isFirstTile = (kTileU32 == 0u);
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=slot */
      const bool navatala_copy_direct_ok = ((unsigned int)(kStrideVal) % 2u) == 0u && ((unsigned int)(kBase) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (isFirstTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&a[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 8u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 8u);
            const unsigned int navatala_dst_idx = ((slot) * 512u) + (((0u) + navatala_copy_r) * 8u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((blockRow) + navatala_copy_r) * (kStrideVal)) + ((kBase) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelA[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&a[0], &panelA[0], (unsigned int)(slot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(blockRow), (unsigned int)(kBase), (unsigned int)(kStrideVal), 64u, 8u, 8u, 512u);
        }
      }
    }
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=slot */
      const bool navatala_copy_direct_ok = ((unsigned int)(nStrideVal) % 2u) == 0u && ((unsigned int)(blockCol) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (isFirstTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&b[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 64u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 64u);
            const unsigned int navatala_dst_idx = ((slot) * 512u) + (((0u) + navatala_copy_r) * 64u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((kBase) + navatala_copy_r) * (nStrideVal)) + ((blockCol) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelB[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&b[0], &panelB[0], (unsigned int)(slot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(kBase), (unsigned int)(blockCol), (unsigned int)(nStrideVal), 8u, 64u, 64u, 512u);
        }
      }
    }
    /* R6 AwaitCopyGroup HIP direct-LDS wait: group=slot */
    navatala_gpu_wait_direct_lds_copy();
    __syncthreads();
    unsigned int nextTileU32 = (kTileU32 + 1u);
    bool hasNextTile = (nextTileU32 < kTilesVal);
    unsigned int nextSlot = (nextTileU32 % 2u);
    unsigned int nextKBase = (nextTileU32 * 8u);
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=nextSlot */
      const bool navatala_copy_direct_ok = ((unsigned int)(kStrideVal) % 2u) == 0u && ((unsigned int)(nextKBase) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (hasNextTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&a[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 8u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 8u);
            const unsigned int navatala_dst_idx = ((nextSlot) * 512u) + (((0u) + navatala_copy_r) * 8u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((blockRow) + navatala_copy_r) * (kStrideVal)) + ((nextKBase) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelA[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&a[0], &panelA[0], (unsigned int)(nextSlot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(blockRow), (unsigned int)(nextKBase), (unsigned int)(kStrideVal), 64u, 8u, 8u, 512u);
        }
      }
    }
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=nextSlot */
      const bool navatala_copy_direct_ok = ((unsigned int)(nStrideVal) % 2u) == 0u && ((unsigned int)(blockCol) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (hasNextTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&b[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 64u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 64u);
            const unsigned int navatala_dst_idx = ((nextSlot) * 512u) + (((0u) + navatala_copy_r) * 64u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((nextKBase) + navatala_copy_r) * (nStrideVal)) + ((blockCol) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelB[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&b[0], &panelB[0], (unsigned int)(nextSlot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(nextKBase), (unsigned int)(blockCol), (unsigned int)(nStrideVal), 8u, 64u, 64u, 512u);
        }
      }
    }
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(((panelA) + ((slot) * 512u)), waveRowLocal, 0u, 8u);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(((panelB) + ((slot) * 512u)), 0u, waveColLocal, 64u);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta128(const __half* a, const __half* b, const unsigned int* kBlocks, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 128u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 128u);
  unsigned int waveRowLocalBase = (waveRowGroup * 64u);
  unsigned int waveColLocalBase = (waveColGroup * 64u);
  unsigned int waveRowBase = (blockRow + waveRowLocalBase);
  unsigned int waveColBase = (blockCol + waveColLocalBase);
  unsigned int kBlocksVal = kBlocks[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  __shared__ __half tileA[4096];
  __shared__ __half tileB[4096];
  navatala_mfma_acc_f32_32x32x8 acc00 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc01 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc10 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc11 = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kBlock = 0; kBlock < (int)(kBlocksVal); ++kBlock) {
    unsigned int kBlockU32 = ((unsigned int)(kBlock));
    unsigned int kBase = (kBlockU32 * 32u);
    for (int loadIter = 0; loadIter < (int)(16u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 32u);
      unsigned int aColLocal = (_flat % 32u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdx = ((aGlobalRow * kStrideVal) + aGlobalCol);
      __half aVal = a[aIdx];
      unsigned int bRowLocal = (_flat / 128u);
      unsigned int bColLocal = (_flat % 128u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdx = ((bGlobalRow * nStrideVal) + bGlobalCol);
      __half bVal = b[bIdx];
      tileA[_flat] = aVal;
      tileB[_flat] = bVal;
    }
    __syncthreads();
    for (int kkTile = 0; kkTile < (int)(4u); ++kkTile) {
      unsigned int kkTileU32 = ((unsigned int)(kkTile));
      unsigned int kkBase = (kkTileU32 * 8u);
      navatala_mfma_f16x4 aFrag0 = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocalBase, kkBase, 32u);
      navatala_mfma_f16x4 aFrag1 = navatala_mfma_load_a_f16_32x32x8(tileA, (waveRowLocalBase + 32u), kkBase, 32u);
      navatala_mfma_f16x4 bFrag0 = navatala_mfma_load_b_f16_32x32x8(tileB, kkBase, waveColLocalBase, 128u);
      navatala_mfma_f16x4 bFrag1 = navatala_mfma_load_b_f16_32x32x8(tileB, kkBase, (waveColLocalBase + 32u), 128u);
      acc00 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag0, bFrag0, acc00, 0, 0, 0);
      acc01 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag0, bFrag1, acc01, 0, 0, 0);
      acc10 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag1, bFrag0, acc10, 0, 0, 0);
      acc11 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag1, bFrag1, acc11, 0, 0, 0);
    }
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc00, waveRowBase, waveColBase, nStrideVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc01, waveRowBase, (waveColBase + 32u), nStrideVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc10, (waveRowBase + 32u), waveColBase, nStrideVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc11, (waveRowBase + 32u), (waveColBase + 32u), nStrideVal, true);
}

)kernel";
const char* k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128_edge = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta128_edge(const __half* a, const __half* b, const unsigned int* mDim, const unsigned int* nDim, const unsigned int* kDim, const unsigned int* kBlocks, const unsigned int* aStride, const unsigned int* bStride, const unsigned int* cStride, const unsigned int* aBatchStride, const unsigned int* bBatchStride, const unsigned int* cBatchStride, const unsigned int* transA, const unsigned int* transB, const float* alpha, const float* beta, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.z)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 128u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 128u);
  unsigned int waveRowLocalBase = (waveRowGroup * 64u);
  unsigned int waveColLocalBase = (waveColGroup * 64u);
  unsigned int waveRowBase = (blockRow + waveRowLocalBase);
  unsigned int waveColBase = (blockCol + waveColLocalBase);
  unsigned int mDimVal = mDim[0u];
  unsigned int nDimVal = nDim[0u];
  unsigned int kDimVal = kDim[0u];
  unsigned int kBlocksVal = kBlocks[0u];
  unsigned int aStrideVal = aStride[0u];
  unsigned int bStrideVal = bStride[0u];
  unsigned int cStrideVal = cStride[0u];
  unsigned int aBatchStrideVal = aBatchStride[0u];
  unsigned int bBatchStrideVal = bBatchStride[0u];
  unsigned int cBatchStrideVal = cBatchStride[0u];
  unsigned int transAVal = transA[0u];
  unsigned int transBVal = transB[0u];
  bool useTransA = (transAVal != 0u);
  bool useTransB = (transBVal != 0u);
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int aBatchBase = (batchIdx * aBatchStrideVal);
  unsigned int bBatchBase = (batchIdx * bBatchStrideVal);
  unsigned int cBatchBase = (batchIdx * cBatchStrideVal);
  unsigned int cWaveColBase = (cBatchBase + waveColBase);
  unsigned int cColLimit = (cBatchBase + nDimVal);
  __shared__ __half tileA[4096];
  __shared__ __half tileB[4096];
  navatala_mfma_acc_f32_32x32x8 acc00 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc01 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc10 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc11 = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kBlock = 0; kBlock < (int)(kBlocksVal); ++kBlock) {
    unsigned int kBlockU32 = ((unsigned int)(kBlock));
    unsigned int kBase = (kBlockU32 * 32u);
    for (int loadIter = 0; loadIter < (int)(16u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 32u);
      unsigned int aColLocal = (_flat % 32u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdxNN = ((aGlobalRow * aStrideVal) + aGlobalCol);
      unsigned int aIdxT = ((aGlobalCol * aStrideVal) + aGlobalRow);
      unsigned int aIdx = ((useTransA) ? (aIdxT) : (aIdxNN));
      unsigned int aIdxBatch = (aBatchBase + aIdx);
      bool aRowValid = (aGlobalRow < mDimVal);
      bool aColValid = (aGlobalCol < kDimVal);
      bool aLoadValid = (aRowValid && aColValid);
      __half aVal = ((aLoadValid) ? (a[aIdxBatch]) : (__float2half(0.000000f)));
      unsigned int bRowLocal = (_flat / 128u);
      unsigned int bColLocal = (_flat % 128u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdxNN = ((bGlobalRow * bStrideVal) + bGlobalCol);
      unsigned int bIdxT = ((bGlobalCol * bStrideVal) + bGlobalRow);
      unsigned int bIdx = ((useTransB) ? (bIdxT) : (bIdxNN));
      unsigned int bIdxBatch = (bBatchBase + bIdx);
      bool bRowValid = (bGlobalRow < kDimVal);
      bool bColValid = (bGlobalCol < nDimVal);
      bool bLoadValid = (bRowValid && bColValid);
      __half bVal = ((bLoadValid) ? (b[bIdxBatch]) : (__float2half(0.000000f)));
      tileA[_flat] = aVal;
      tileB[_flat] = bVal;
    }
    __syncthreads();
    for (int kkTile = 0; kkTile < (int)(4u); ++kkTile) {
      unsigned int kkTileU32 = ((unsigned int)(kkTile));
      unsigned int kkBase = (kkTileU32 * 8u);
      navatala_mfma_f16x4 aFrag0 = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocalBase, kkBase, 32u);
      navatala_mfma_f16x4 aFrag1 = navatala_mfma_load_a_f16_32x32x8(tileA, (waveRowLocalBase + 32u), kkBase, 32u);
      navatala_mfma_f16x4 bFrag0 = navatala_mfma_load_b_f16_32x32x8(tileB, kkBase, waveColLocalBase, 128u);
      navatala_mfma_f16x4 bFrag1 = navatala_mfma_load_b_f16_32x32x8(tileB, kkBase, (waveColLocalBase + 32u), 128u);
      acc00 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag0, bFrag0, acc00, 0, 0, 0);
      acc01 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag0, bFrag1, acc01, 0, 0, 0);
      acc10 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag1, bFrag0, acc10, 0, 0, 0);
      acc11 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag1, bFrag1, acc11, 0, 0, 0);
    }
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc00, waveRowBase, cWaveColBase, cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc01, waveRowBase, (cWaveColBase + 32u), cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc10, (waveRowBase + 32u), cWaveColBase, cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc11, (waveRowBase + 32u), (cWaveColBase + 32u), cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_gelu_f32 = {
  1,
  "navatala_transformer_gelu_f32",
  "hip",
  "navatala_transformer_gelu_f32",
  "kernel:hip:navatala_transformer_gelu_f32",
  "abi-r1:hip:navatala_transformer_gelu_f32",
  "abi-r1:hip:navatala_transformer_gelu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_gelu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_gelu_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_gelu_tanh_f32 = {
  1,
  "navatala_transformer_gelu_tanh_f32",
  "hip",
  "navatala_transformer_gelu_tanh_f32",
  "kernel:hip:navatala_transformer_gelu_tanh_f32",
  "abi-r1:hip:navatala_transformer_gelu_tanh_f32",
  "abi-r1:hip:navatala_transformer_gelu_tanh_f32",
  3,
  kAbiArgs_hip_navatala_transformer_gelu_tanh_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_silu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_silu_f32 = {
  1,
  "navatala_transformer_silu_f32",
  "hip",
  "navatala_transformer_silu_f32",
  "kernel:hip:navatala_transformer_silu_f32",
  "abi-r1:hip:navatala_transformer_silu_f32",
  "abi-r1:hip:navatala_transformer_silu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_silu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_relu_f32 = {
  1,
  "navatala_transformer_relu_f32",
  "hip",
  "navatala_transformer_relu_f32",
  "kernel:hip:navatala_transformer_relu_f32",
  "abi-r1:hip:navatala_transformer_relu_f32",
  "abi-r1:hip:navatala_transformer_relu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_relu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_quick_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_quick_gelu_f32 = {
  1,
  "navatala_transformer_quick_gelu_f32",
  "hip",
  "navatala_transformer_quick_gelu_f32",
  "kernel:hip:navatala_transformer_quick_gelu_f32",
  "abi-r1:hip:navatala_transformer_quick_gelu_f32",
  "abi-r1:hip:navatala_transformer_quick_gelu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_quick_gelu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_squared_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_squared_relu_f32 = {
  1,
  "navatala_transformer_squared_relu_f32",
  "hip",
  "navatala_transformer_squared_relu_f32",
  "kernel:hip:navatala_transformer_squared_relu_f32",
  "abi-r1:hip:navatala_transformer_squared_relu_f32",
  "abi-r1:hip:navatala_transformer_squared_relu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_squared_relu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_bias_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "biasSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_bias_gelu_f32 = {
  1,
  "navatala_transformer_bias_gelu_f32",
  "hip",
  "navatala_transformer_bias_gelu_f32",
  "kernel:hip:navatala_transformer_bias_gelu_f32",
  "abi-r1:hip:navatala_transformer_bias_gelu_f32",
  "abi-r1:hip:navatala_transformer_bias_gelu_f32",
  5,
  kAbiArgs_hip_navatala_transformer_bias_gelu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_gelu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_gelu_f16 = {
  1,
  "navatala_transformer_gelu_f16",
  "hip",
  "navatala_transformer_gelu_f16",
  "kernel:hip:navatala_transformer_gelu_f16",
  "abi-r1:hip:navatala_transformer_gelu_f16",
  "abi-r1:hip:navatala_transformer_gelu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_gelu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_silu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_silu_f16 = {
  1,
  "navatala_transformer_silu_f16",
  "hip",
  "navatala_transformer_silu_f16",
  "kernel:hip:navatala_transformer_silu_f16",
  "abi-r1:hip:navatala_transformer_silu_f16",
  "abi-r1:hip:navatala_transformer_silu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_silu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_relu_f16 = {
  1,
  "navatala_transformer_relu_f16",
  "hip",
  "navatala_transformer_relu_f16",
  "kernel:hip:navatala_transformer_relu_f16",
  "abi-r1:hip:navatala_transformer_relu_f16",
  "abi-r1:hip:navatala_transformer_relu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_relu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_quick_gelu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_quick_gelu_f16 = {
  1,
  "navatala_transformer_quick_gelu_f16",
  "hip",
  "navatala_transformer_quick_gelu_f16",
  "kernel:hip:navatala_transformer_quick_gelu_f16",
  "abi-r1:hip:navatala_transformer_quick_gelu_f16",
  "abi-r1:hip:navatala_transformer_quick_gelu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_quick_gelu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_squared_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_squared_relu_f16 = {
  1,
  "navatala_transformer_squared_relu_f16",
  "hip",
  "navatala_transformer_squared_relu_f16",
  "kernel:hip:navatala_transformer_squared_relu_f16",
  "abi-r1:hip:navatala_transformer_squared_relu_f16",
  "abi-r1:hip:navatala_transformer_squared_relu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_squared_relu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_gelu_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_gelu_bf16 = {
  1,
  "navatala_transformer_gelu_bf16",
  "hip",
  "navatala_transformer_gelu_bf16",
  "kernel:hip:navatala_transformer_gelu_bf16",
  "abi-r1:hip:navatala_transformer_gelu_bf16",
  "abi-r1:hip:navatala_transformer_gelu_bf16",
  3,
  kAbiArgs_hip_navatala_transformer_gelu_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_silu_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_silu_bf16 = {
  1,
  "navatala_transformer_silu_bf16",
  "hip",
  "navatala_transformer_silu_bf16",
  "kernel:hip:navatala_transformer_silu_bf16",
  "abi-r1:hip:navatala_transformer_silu_bf16",
  "abi-r1:hip:navatala_transformer_silu_bf16",
  3,
  kAbiArgs_hip_navatala_transformer_silu_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_geglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_geglu_f32 = {
  1,
  "navatala_transformer_geglu_f32",
  "hip",
  "navatala_transformer_geglu_f32",
  "kernel:hip:navatala_transformer_geglu_f32",
  "abi-r1:hip:navatala_transformer_geglu_f32",
  "abi-r1:hip:navatala_transformer_geglu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_geglu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_swiglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_swiglu_f32 = {
  1,
  "navatala_transformer_swiglu_f32",
  "hip",
  "navatala_transformer_swiglu_f32",
  "kernel:hip:navatala_transformer_swiglu_f32",
  "abi-r1:hip:navatala_transformer_swiglu_f32",
  "abi-r1:hip:navatala_transformer_swiglu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_swiglu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_reglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_reglu_f32 = {
  1,
  "navatala_transformer_reglu_f32",
  "hip",
  "navatala_transformer_reglu_f32",
  "kernel:hip:navatala_transformer_reglu_f32",
  "abi-r1:hip:navatala_transformer_reglu_f32",
  "abi-r1:hip:navatala_transformer_reglu_f32",
  3,
  kAbiArgs_hip_navatala_transformer_reglu_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_geglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_geglu_f16 = {
  1,
  "navatala_transformer_geglu_f16",
  "hip",
  "navatala_transformer_geglu_f16",
  "kernel:hip:navatala_transformer_geglu_f16",
  "abi-r1:hip:navatala_transformer_geglu_f16",
  "abi-r1:hip:navatala_transformer_geglu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_geglu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_swiglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_swiglu_f16 = {
  1,
  "navatala_transformer_swiglu_f16",
  "hip",
  "navatala_transformer_swiglu_f16",
  "kernel:hip:navatala_transformer_swiglu_f16",
  "abi-r1:hip:navatala_transformer_swiglu_f16",
  "abi-r1:hip:navatala_transformer_swiglu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_swiglu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_reglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_reglu_f16 = {
  1,
  "navatala_transformer_reglu_f16",
  "hip",
  "navatala_transformer_reglu_f16",
  "kernel:hip:navatala_transformer_reglu_f16",
  "abi-r1:hip:navatala_transformer_reglu_f16",
  "abi-r1:hip:navatala_transformer_reglu_f16",
  3,
  kAbiArgs_hip_navatala_transformer_reglu_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_gelu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_gelu_backward_f32 = {
  1,
  "navatala_transformer_gelu_backward_f32",
  "hip",
  "navatala_transformer_gelu_backward_f32",
  "kernel:hip:navatala_transformer_gelu_backward_f32",
  "abi-r1:hip:navatala_transformer_gelu_backward_f32",
  "abi-r1:hip:navatala_transformer_gelu_backward_f32",
  4,
  kAbiArgs_hip_navatala_transformer_gelu_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_silu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_silu_backward_f32 = {
  1,
  "navatala_transformer_silu_backward_f32",
  "hip",
  "navatala_transformer_silu_backward_f32",
  "kernel:hip:navatala_transformer_silu_backward_f32",
  "abi-r1:hip:navatala_transformer_silu_backward_f32",
  "abi-r1:hip:navatala_transformer_silu_backward_f32",
  4,
  kAbiArgs_hip_navatala_transformer_silu_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_relu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_relu_backward_f32 = {
  1,
  "navatala_transformer_relu_backward_f32",
  "hip",
  "navatala_transformer_relu_backward_f32",
  "kernel:hip:navatala_transformer_relu_backward_f32",
  "abi-r1:hip:navatala_transformer_relu_backward_f32",
  "abi-r1:hip:navatala_transformer_relu_backward_f32",
  4,
  kAbiArgs_hip_navatala_transformer_relu_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_quick_gelu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_quick_gelu_backward_f32 = {
  1,
  "navatala_transformer_quick_gelu_backward_f32",
  "hip",
  "navatala_transformer_quick_gelu_backward_f32",
  "kernel:hip:navatala_transformer_quick_gelu_backward_f32",
  "abi-r1:hip:navatala_transformer_quick_gelu_backward_f32",
  "abi-r1:hip:navatala_transformer_quick_gelu_backward_f32",
  4,
  kAbiArgs_hip_navatala_transformer_quick_gelu_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_gelu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_gelu_backward_f16 = {
  1,
  "navatala_transformer_gelu_backward_f16",
  "hip",
  "navatala_transformer_gelu_backward_f16",
  "kernel:hip:navatala_transformer_gelu_backward_f16",
  "abi-r1:hip:navatala_transformer_gelu_backward_f16",
  "abi-r1:hip:navatala_transformer_gelu_backward_f16",
  4,
  kAbiArgs_hip_navatala_transformer_gelu_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_silu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_silu_backward_f16 = {
  1,
  "navatala_transformer_silu_backward_f16",
  "hip",
  "navatala_transformer_silu_backward_f16",
  "kernel:hip:navatala_transformer_silu_backward_f16",
  "abi-r1:hip:navatala_transformer_silu_backward_f16",
  "abi-r1:hip:navatala_transformer_silu_backward_f16",
  4,
  kAbiArgs_hip_navatala_transformer_silu_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_relu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_relu_backward_f16 = {
  1,
  "navatala_transformer_relu_backward_f16",
  "hip",
  "navatala_transformer_relu_backward_f16",
  "kernel:hip:navatala_transformer_relu_backward_f16",
  "abi-r1:hip:navatala_transformer_relu_backward_f16",
  "abi-r1:hip:navatala_transformer_relu_backward_f16",
  4,
  kAbiArgs_hip_navatala_transformer_relu_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_quick_gelu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_quick_gelu_backward_f16 = {
  1,
  "navatala_transformer_quick_gelu_backward_f16",
  "hip",
  "navatala_transformer_quick_gelu_backward_f16",
  "kernel:hip:navatala_transformer_quick_gelu_backward_f16",
  "abi-r1:hip:navatala_transformer_quick_gelu_backward_f16",
  "abi-r1:hip:navatala_transformer_quick_gelu_backward_f16",
  4,
  kAbiArgs_hip_navatala_transformer_quick_gelu_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_forward_f32 = {
  1,
  "navatala_transformer_rms_norm_forward_f32",
  "hip",
  "navatala_transformer_rms_norm_forward_f32",
  "kernel:hip:navatala_transformer_rms_norm_forward_f32",
  "abi-r1:hip:navatala_transformer_rms_norm_forward_f32",
  "abi-r1:hip:navatala_transformer_rms_norm_forward_f32",
  6,
  kAbiArgs_hip_navatala_transformer_rms_norm_forward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_forward_f16 = {
  1,
  "navatala_transformer_rms_norm_forward_f16",
  "hip",
  "navatala_transformer_rms_norm_forward_f16",
  "kernel:hip:navatala_transformer_rms_norm_forward_f16",
  "abi-r1:hip:navatala_transformer_rms_norm_forward_f16",
  "abi-r1:hip:navatala_transformer_rms_norm_forward_f16",
  6,
  kAbiArgs_hip_navatala_transformer_rms_norm_forward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_forward_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_forward_bf16 = {
  1,
  "navatala_transformer_rms_norm_forward_bf16",
  "hip",
  "navatala_transformer_rms_norm_forward_bf16",
  "kernel:hip:navatala_transformer_rms_norm_forward_bf16",
  "abi-r1:hip:navatala_transformer_rms_norm_forward_bf16",
  "abi-r1:hip:navatala_transformer_rms_norm_forward_bf16",
  6,
  kAbiArgs_hip_navatala_transformer_rms_norm_forward_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_forward_f32 = {
  1,
  "navatala_transformer_layer_norm_forward_f32",
  "hip",
  "navatala_transformer_layer_norm_forward_f32",
  "kernel:hip:navatala_transformer_layer_norm_forward_f32",
  "abi-r1:hip:navatala_transformer_layer_norm_forward_f32",
  "abi-r1:hip:navatala_transformer_layer_norm_forward_f32",
  7,
  kAbiArgs_hip_navatala_transformer_layer_norm_forward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_forward_f16 = {
  1,
  "navatala_transformer_layer_norm_forward_f16",
  "hip",
  "navatala_transformer_layer_norm_forward_f16",
  "kernel:hip:navatala_transformer_layer_norm_forward_f16",
  "abi-r1:hip:navatala_transformer_layer_norm_forward_f16",
  "abi-r1:hip:navatala_transformer_layer_norm_forward_f16",
  7,
  kAbiArgs_hip_navatala_transformer_layer_norm_forward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_forward_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_forward_bf16 = {
  1,
  "navatala_transformer_layer_norm_forward_bf16",
  "hip",
  "navatala_transformer_layer_norm_forward_bf16",
  "kernel:hip:navatala_transformer_layer_norm_forward_bf16",
  "abi-r1:hip:navatala_transformer_layer_norm_forward_bf16",
  "abi-r1:hip:navatala_transformer_layer_norm_forward_bf16",
  7,
  kAbiArgs_hip_navatala_transformer_layer_norm_forward_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mean", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradBeta", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_backward_f32 = {
  1,
  "navatala_transformer_layer_norm_backward_f32",
  "hip",
  "navatala_transformer_layer_norm_backward_f32",
  "kernel:hip:navatala_transformer_layer_norm_backward_f32",
  "abi-r1:hip:navatala_transformer_layer_norm_backward_f32",
  "abi-r1:hip:navatala_transformer_layer_norm_backward_f32",
  10,
  kAbiArgs_hip_navatala_transformer_layer_norm_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mean", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradBeta", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_backward_f16 = {
  1,
  "navatala_transformer_layer_norm_backward_f16",
  "hip",
  "navatala_transformer_layer_norm_backward_f16",
  "kernel:hip:navatala_transformer_layer_norm_backward_f16",
  "abi-r1:hip:navatala_transformer_layer_norm_backward_f16",
  "abi-r1:hip:navatala_transformer_layer_norm_backward_f16",
  10,
  kAbiArgs_hip_navatala_transformer_layer_norm_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_backward_f32 = {
  1,
  "navatala_transformer_rms_norm_backward_f32",
  "hip",
  "navatala_transformer_rms_norm_backward_f32",
  "kernel:hip:navatala_transformer_rms_norm_backward_f32",
  "abi-r1:hip:navatala_transformer_rms_norm_backward_f32",
  "abi-r1:hip:navatala_transformer_rms_norm_backward_f32",
  8,
  kAbiArgs_hip_navatala_transformer_rms_norm_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_backward_f16 = {
  1,
  "navatala_transformer_rms_norm_backward_f16",
  "hip",
  "navatala_transformer_rms_norm_backward_f16",
  "kernel:hip:navatala_transformer_rms_norm_backward_f16",
  "abi-r1:hip:navatala_transformer_rms_norm_backward_f16",
  "abi-r1:hip:navatala_transformer_rms_norm_backward_f16",
  8,
  kAbiArgs_hip_navatala_transformer_rms_norm_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_multi_pass_f32 = {
  1,
  "navatala_transformer_layer_norm_multi_pass_f32",
  "hip",
  "navatala_transformer_layer_norm_multi_pass_f32",
  "kernel:hip:navatala_transformer_layer_norm_multi_pass_f32",
  "abi-r1:hip:navatala_transformer_layer_norm_multi_pass_f32",
  "abi-r1:hip:navatala_transformer_layer_norm_multi_pass_f32",
  7,
  kAbiArgs_hip_navatala_transformer_layer_norm_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_layer_norm_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_layer_norm_multi_pass_f16 = {
  1,
  "navatala_transformer_layer_norm_multi_pass_f16",
  "hip",
  "navatala_transformer_layer_norm_multi_pass_f16",
  "kernel:hip:navatala_transformer_layer_norm_multi_pass_f16",
  "abi-r1:hip:navatala_transformer_layer_norm_multi_pass_f16",
  "abi-r1:hip:navatala_transformer_layer_norm_multi_pass_f16",
  7,
  kAbiArgs_hip_navatala_transformer_layer_norm_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_multi_pass_f32 = {
  1,
  "navatala_transformer_rms_norm_multi_pass_f32",
  "hip",
  "navatala_transformer_rms_norm_multi_pass_f32",
  "kernel:hip:navatala_transformer_rms_norm_multi_pass_f32",
  "abi-r1:hip:navatala_transformer_rms_norm_multi_pass_f32",
  "abi-r1:hip:navatala_transformer_rms_norm_multi_pass_f32",
  6,
  kAbiArgs_hip_navatala_transformer_rms_norm_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rms_norm_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rms_norm_multi_pass_f16 = {
  1,
  "navatala_transformer_rms_norm_multi_pass_f16",
  "hip",
  "navatala_transformer_rms_norm_multi_pass_f16",
  "kernel:hip:navatala_transformer_rms_norm_multi_pass_f16",
  "abi-r1:hip:navatala_transformer_rms_norm_multi_pass_f16",
  "abi-r1:hip:navatala_transformer_rms_norm_multi_pass_f16",
  6,
  kAbiArgs_hip_navatala_transformer_rms_norm_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_forward_f32 = {
  1,
  "navatala_transformer_softmax_forward_f32",
  "hip",
  "navatala_transformer_softmax_forward_f32",
  "kernel:hip:navatala_transformer_softmax_forward_f32",
  "abi-r1:hip:navatala_transformer_softmax_forward_f32",
  "abi-r1:hip:navatala_transformer_softmax_forward_f32",
  4,
  kAbiArgs_hip_navatala_transformer_softmax_forward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_forward_f16 = {
  1,
  "navatala_transformer_softmax_forward_f16",
  "hip",
  "navatala_transformer_softmax_forward_f16",
  "kernel:hip:navatala_transformer_softmax_forward_f16",
  "abi-r1:hip:navatala_transformer_softmax_forward_f16",
  "abi-r1:hip:navatala_transformer_softmax_forward_f16",
  4,
  kAbiArgs_hip_navatala_transformer_softmax_forward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_forward_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_forward_bf16 = {
  1,
  "navatala_transformer_softmax_forward_bf16",
  "hip",
  "navatala_transformer_softmax_forward_bf16",
  "kernel:hip:navatala_transformer_softmax_forward_bf16",
  "abi-r1:hip:navatala_transformer_softmax_forward_bf16",
  "abi-r1:hip:navatala_transformer_softmax_forward_bf16",
  4,
  kAbiArgs_hip_navatala_transformer_softmax_forward_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_backward_f32[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_backward_f32 = {
  1,
  "navatala_transformer_softmax_backward_f32",
  "hip",
  "navatala_transformer_softmax_backward_f32",
  "kernel:hip:navatala_transformer_softmax_backward_f32",
  "abi-r1:hip:navatala_transformer_softmax_backward_f32",
  "abi-r1:hip:navatala_transformer_softmax_backward_f32",
  5,
  kAbiArgs_hip_navatala_transformer_softmax_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_backward_f16[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_backward_f16 = {
  1,
  "navatala_transformer_softmax_backward_f16",
  "hip",
  "navatala_transformer_softmax_backward_f16",
  "kernel:hip:navatala_transformer_softmax_backward_f16",
  "abi-r1:hip:navatala_transformer_softmax_backward_f16",
  "abi-r1:hip:navatala_transformer_softmax_backward_f16",
  5,
  kAbiArgs_hip_navatala_transformer_softmax_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_with_mask_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_with_mask_f32 = {
  1,
  "navatala_transformer_softmax_with_mask_f32",
  "hip",
  "navatala_transformer_softmax_with_mask_f32",
  "kernel:hip:navatala_transformer_softmax_with_mask_f32",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_f32",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_f32",
  5,
  kAbiArgs_hip_navatala_transformer_softmax_with_mask_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_with_mask_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_with_mask_f16 = {
  1,
  "navatala_transformer_softmax_with_mask_f16",
  "hip",
  "navatala_transformer_softmax_with_mask_f16",
  "kernel:hip:navatala_transformer_softmax_with_mask_f16",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_f16",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_f16",
  5,
  kAbiArgs_hip_navatala_transformer_softmax_with_mask_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_with_mask_backward_f32[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_with_mask_backward_f32 = {
  1,
  "navatala_transformer_softmax_with_mask_backward_f32",
  "hip",
  "navatala_transformer_softmax_with_mask_backward_f32",
  "kernel:hip:navatala_transformer_softmax_with_mask_backward_f32",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_backward_f32",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_backward_f32",
  6,
  kAbiArgs_hip_navatala_transformer_softmax_with_mask_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_with_mask_backward_f16[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_with_mask_backward_f16 = {
  1,
  "navatala_transformer_softmax_with_mask_backward_f16",
  "hip",
  "navatala_transformer_softmax_with_mask_backward_f16",
  "kernel:hip:navatala_transformer_softmax_with_mask_backward_f16",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_backward_f16",
  "abi-r1:hip:navatala_transformer_softmax_with_mask_backward_f16",
  6,
  kAbiArgs_hip_navatala_transformer_softmax_with_mask_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_multi_pass_f32 = {
  1,
  "navatala_transformer_softmax_multi_pass_f32",
  "hip",
  "navatala_transformer_softmax_multi_pass_f32",
  "kernel:hip:navatala_transformer_softmax_multi_pass_f32",
  "abi-r1:hip:navatala_transformer_softmax_multi_pass_f32",
  "abi-r1:hip:navatala_transformer_softmax_multi_pass_f32",
  4,
  kAbiArgs_hip_navatala_transformer_softmax_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_softmax_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_softmax_multi_pass_f16 = {
  1,
  "navatala_transformer_softmax_multi_pass_f16",
  "hip",
  "navatala_transformer_softmax_multi_pass_f16",
  "kernel:hip:navatala_transformer_softmax_multi_pass_f16",
  "abi-r1:hip:navatala_transformer_softmax_multi_pass_f16",
  "abi-r1:hip:navatala_transformer_softmax_multi_pass_f16",
  4,
  kAbiArgs_hip_navatala_transformer_softmax_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f32_to_f16 = {
  1,
  "navatala_transformer_cast_f32_to_f16",
  "hip",
  "navatala_transformer_cast_f32_to_f16",
  "kernel:hip:navatala_transformer_cast_f32_to_f16",
  "abi-r1:hip:navatala_transformer_cast_f32_to_f16",
  "abi-r1:hip:navatala_transformer_cast_f32_to_f16",
  3,
  kAbiArgs_hip_navatala_transformer_cast_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f16_to_f32 = {
  1,
  "navatala_transformer_cast_f16_to_f32",
  "hip",
  "navatala_transformer_cast_f16_to_f32",
  "kernel:hip:navatala_transformer_cast_f16_to_f32",
  "abi-r1:hip:navatala_transformer_cast_f16_to_f32",
  "abi-r1:hip:navatala_transformer_cast_f16_to_f32",
  3,
  kAbiArgs_hip_navatala_transformer_cast_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f32_to_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f32_to_bf16 = {
  1,
  "navatala_transformer_cast_f32_to_bf16",
  "hip",
  "navatala_transformer_cast_f32_to_bf16",
  "kernel:hip:navatala_transformer_cast_f32_to_bf16",
  "abi-r1:hip:navatala_transformer_cast_f32_to_bf16",
  "abi-r1:hip:navatala_transformer_cast_f32_to_bf16",
  3,
  kAbiArgs_hip_navatala_transformer_cast_f32_to_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_b_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_b_f16_to_f32 = {
  1,
  "navatala_transformer_cast_b_f16_to_f32",
  "hip",
  "navatala_transformer_cast_b_f16_to_f32",
  "kernel:hip:navatala_transformer_cast_b_f16_to_f32",
  "abi-r1:hip:navatala_transformer_cast_b_f16_to_f32",
  "abi-r1:hip:navatala_transformer_cast_b_f16_to_f32",
  3,
  kAbiArgs_hip_navatala_transformer_cast_b_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_transpose_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_transpose_f32 = {
  1,
  "navatala_transformer_transpose_f32",
  "hip",
  "navatala_transformer_transpose_f32",
  "kernel:hip:navatala_transformer_transpose_f32",
  "abi-r1:hip:navatala_transformer_transpose_f32",
  "abi-r1:hip:navatala_transformer_transpose_f32",
  4,
  kAbiArgs_hip_navatala_transformer_transpose_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_transpose_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_transpose_f16 = {
  1,
  "navatala_transformer_transpose_f16",
  "hip",
  "navatala_transformer_transpose_f16",
  "kernel:hip:navatala_transformer_transpose_f16",
  "abi-r1:hip:navatala_transformer_transpose_f16",
  "abi-r1:hip:navatala_transformer_transpose_f16",
  4,
  kAbiArgs_hip_navatala_transformer_transpose_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_fused_cast_transpose_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_fused_cast_transpose_f32_to_f16 = {
  1,
  "navatala_transformer_fused_cast_transpose_f32_to_f16",
  "hip",
  "navatala_transformer_fused_cast_transpose_f32_to_f16",
  "kernel:hip:navatala_transformer_fused_cast_transpose_f32_to_f16",
  "abi-r1:hip:navatala_transformer_fused_cast_transpose_f32_to_f16",
  "abi-r1:hip:navatala_transformer_fused_cast_transpose_f32_to_f16",
  4,
  kAbiArgs_hip_navatala_transformer_fused_cast_transpose_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f32_to_f_p8_e4_m3[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f32_to_f_p8_e4_m3 = {
  1,
  "navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "hip",
  "navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "kernel:hip:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "abi-r1:hip:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "abi-r1:hip:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  4,
  kAbiArgs_hip_navatala_transformer_cast_f32_to_f_p8_e4_m3
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f_p8_e4_m3_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f_p8_e4_m3_to_f32 = {
  1,
  "navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "hip",
  "navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "kernel:hip:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "abi-r1:hip:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "abi-r1:hip:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  4,
  kAbiArgs_hip_navatala_transformer_cast_f_p8_e4_m3_to_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f32_to_f_p8_e5_m2[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f32_to_f_p8_e5_m2 = {
  1,
  "navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "hip",
  "navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "kernel:hip:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "abi-r1:hip:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "abi-r1:hip:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  4,
  kAbiArgs_hip_navatala_transformer_cast_f32_to_f_p8_e5_m2
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_cast_f_p8_e5_m2_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_cast_f_p8_e5_m2_to_f32 = {
  1,
  "navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "hip",
  "navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "kernel:hip:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "abi-r1:hip:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "abi-r1:hip:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  4,
  kAbiArgs_hip_navatala_transformer_cast_f_p8_e5_m2_to_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_fused_cast_transpose_f_p8[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_fused_cast_transpose_f_p8 = {
  1,
  "navatala_transformer_fused_cast_transpose_f_p8",
  "hip",
  "navatala_transformer_fused_cast_transpose_f_p8",
  "kernel:hip:navatala_transformer_fused_cast_transpose_f_p8",
  "abi-r1:hip:navatala_transformer_fused_cast_transpose_f_p8",
  "abi-r1:hip:navatala_transformer_fused_cast_transpose_f_p8",
  5,
  kAbiArgs_hip_navatala_transformer_fused_cast_transpose_f_p8
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_transpose4_d_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_transpose4_d_f32 = {
  1,
  "navatala_transformer_transpose4_d_f32",
  "hip",
  "navatala_transformer_transpose4_d_f32",
  "kernel:hip:navatala_transformer_transpose4_d_f32",
  "abi-r1:hip:navatala_transformer_transpose4_d_f32",
  "abi-r1:hip:navatala_transformer_transpose4_d_f32",
  10,
  kAbiArgs_hip_navatala_transformer_transpose4_d_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_transpose4_d_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_transpose4_d_f16 = {
  1,
  "navatala_transformer_transpose4_d_f16",
  "hip",
  "navatala_transformer_transpose4_d_f16",
  "kernel:hip:navatala_transformer_transpose4_d_f16",
  "abi-r1:hip:navatala_transformer_transpose4_d_f16",
  "abi-r1:hip:navatala_transformer_transpose4_d_f16",
  10,
  kAbiArgs_hip_navatala_transformer_transpose4_d_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_dropout_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_dropout_forward_f32 = {
  1,
  "navatala_transformer_dropout_forward_f32",
  "hip",
  "navatala_transformer_dropout_forward_f32",
  "kernel:hip:navatala_transformer_dropout_forward_f32",
  "abi-r1:hip:navatala_transformer_dropout_forward_f32",
  "abi-r1:hip:navatala_transformer_dropout_forward_f32",
  5,
  kAbiArgs_hip_navatala_transformer_dropout_forward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_dropout_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_dropout_forward_f16 = {
  1,
  "navatala_transformer_dropout_forward_f16",
  "hip",
  "navatala_transformer_dropout_forward_f16",
  "kernel:hip:navatala_transformer_dropout_forward_f16",
  "abi-r1:hip:navatala_transformer_dropout_forward_f16",
  "abi-r1:hip:navatala_transformer_dropout_forward_f16",
  5,
  kAbiArgs_hip_navatala_transformer_dropout_forward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_dropout_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_dropout_backward_f32 = {
  1,
  "navatala_transformer_dropout_backward_f32",
  "hip",
  "navatala_transformer_dropout_backward_f32",
  "kernel:hip:navatala_transformer_dropout_backward_f32",
  "abi-r1:hip:navatala_transformer_dropout_backward_f32",
  "abi-r1:hip:navatala_transformer_dropout_backward_f32",
  5,
  kAbiArgs_hip_navatala_transformer_dropout_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_dropout_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_dropout_backward_f16 = {
  1,
  "navatala_transformer_dropout_backward_f16",
  "hip",
  "navatala_transformer_dropout_backward_f16",
  "kernel:hip:navatala_transformer_dropout_backward_f16",
  "abi-r1:hip:navatala_transformer_dropout_backward_f16",
  "abi-r1:hip:navatala_transformer_dropout_backward_f16",
  5,
  kAbiArgs_hip_navatala_transformer_dropout_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_forward_f32 = {
  1,
  "navatala_transformer_rope_forward_f32",
  "hip",
  "navatala_transformer_rope_forward_f32",
  "kernel:hip:navatala_transformer_rope_forward_f32",
  "abi-r1:hip:navatala_transformer_rope_forward_f32",
  "abi-r1:hip:navatala_transformer_rope_forward_f32",
  7,
  kAbiArgs_hip_navatala_transformer_rope_forward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_forward_f16 = {
  1,
  "navatala_transformer_rope_forward_f16",
  "hip",
  "navatala_transformer_rope_forward_f16",
  "kernel:hip:navatala_transformer_rope_forward_f16",
  "abi-r1:hip:navatala_transformer_rope_forward_f16",
  "abi-r1:hip:navatala_transformer_rope_forward_f16",
  7,
  kAbiArgs_hip_navatala_transformer_rope_forward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_forward_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_forward_bf16 = {
  1,
  "navatala_transformer_rope_forward_bf16",
  "hip",
  "navatala_transformer_rope_forward_bf16",
  "kernel:hip:navatala_transformer_rope_forward_bf16",
  "abi-r1:hip:navatala_transformer_rope_forward_bf16",
  "abi-r1:hip:navatala_transformer_rope_forward_bf16",
  7,
  kAbiArgs_hip_navatala_transformer_rope_forward_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_with_cache_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cosCache", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinCache", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_with_cache_f32 = {
  1,
  "navatala_transformer_rope_with_cache_f32",
  "hip",
  "navatala_transformer_rope_with_cache_f32",
  "kernel:hip:navatala_transformer_rope_with_cache_f32",
  "abi-r1:hip:navatala_transformer_rope_with_cache_f32",
  "abi-r1:hip:navatala_transformer_rope_with_cache_f32",
  8,
  kAbiArgs_hip_navatala_transformer_rope_with_cache_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_backward_f32 = {
  1,
  "navatala_transformer_rope_backward_f32",
  "hip",
  "navatala_transformer_rope_backward_f32",
  "kernel:hip:navatala_transformer_rope_backward_f32",
  "abi-r1:hip:navatala_transformer_rope_backward_f32",
  "abi-r1:hip:navatala_transformer_rope_backward_f32",
  7,
  kAbiArgs_hip_navatala_transformer_rope_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_backward_f16 = {
  1,
  "navatala_transformer_rope_backward_f16",
  "hip",
  "navatala_transformer_rope_backward_f16",
  "kernel:hip:navatala_transformer_rope_backward_f16",
  "abi-r1:hip:navatala_transformer_rope_backward_f16",
  "abi-r1:hip:navatala_transformer_rope_backward_f16",
  7,
  kAbiArgs_hip_navatala_transformer_rope_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_backward_bf16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_backward_bf16 = {
  1,
  "navatala_transformer_rope_backward_bf16",
  "hip",
  "navatala_transformer_rope_backward_bf16",
  "kernel:hip:navatala_transformer_rope_backward_bf16",
  "abi-r1:hip:navatala_transformer_rope_backward_bf16",
  "abi-r1:hip:navatala_transformer_rope_backward_bf16",
  7,
  kAbiArgs_hip_navatala_transformer_rope_backward_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_interleaved_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_interleaved_f32 = {
  1,
  "navatala_transformer_rope_interleaved_f32",
  "hip",
  "navatala_transformer_rope_interleaved_f32",
  "kernel:hip:navatala_transformer_rope_interleaved_f32",
  "abi-r1:hip:navatala_transformer_rope_interleaved_f32",
  "abi-r1:hip:navatala_transformer_rope_interleaved_f32",
  7,
  kAbiArgs_hip_navatala_transformer_rope_interleaved_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_interleaved_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_interleaved_f16 = {
  1,
  "navatala_transformer_rope_interleaved_f16",
  "hip",
  "navatala_transformer_rope_interleaved_f16",
  "kernel:hip:navatala_transformer_rope_interleaved_f16",
  "abi-r1:hip:navatala_transformer_rope_interleaved_f16",
  "abi-r1:hip:navatala_transformer_rope_interleaved_f16",
  7,
  kAbiArgs_hip_navatala_transformer_rope_interleaved_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_rope_interleaved_bf16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_rope_interleaved_bf16 = {
  1,
  "navatala_transformer_rope_interleaved_bf16",
  "hip",
  "navatala_transformer_rope_interleaved_bf16",
  "kernel:hip:navatala_transformer_rope_interleaved_bf16",
  "abi-r1:hip:navatala_transformer_rope_interleaved_bf16",
  "abi-r1:hip:navatala_transformer_rope_interleaved_bf16",
  7,
  kAbiArgs_hip_navatala_transformer_rope_interleaved_bf16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_f16 = {
  1,
  "navatala_transformer_simple_attention_f16",
  "hip",
  "navatala_transformer_simple_attention_f16",
  "kernel:hip:navatala_transformer_simple_attention_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_f16",
  8,
  kAbiArgs_hip_navatala_transformer_simple_attention_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_f32 = {
  1,
  "navatala_transformer_simple_attention_f32",
  "hip",
  "navatala_transformer_simple_attention_f32",
  "kernel:hip:navatala_transformer_simple_attention_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_f32",
  8,
  kAbiArgs_hip_navatala_transformer_simple_attention_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_paged_attention_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kvCache", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "blockTables", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contextLens", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numHeads", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_paged_attention_f32 = {
  1,
  "navatala_transformer_paged_attention_f32",
  "hip",
  "navatala_transformer_paged_attention_f32",
  "kernel:hip:navatala_transformer_paged_attention_f32",
  "abi-r1:hip:navatala_transformer_paged_attention_f32",
  "abi-r1:hip:navatala_transformer_paged_attention_f32",
  9,
  kAbiArgs_hip_navatala_transformer_paged_attention_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_with_mask_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_with_mask_f32 = {
  1,
  "navatala_transformer_simple_attention_with_mask_f32",
  "hip",
  "navatala_transformer_simple_attention_with_mask_f32",
  "kernel:hip:navatala_transformer_simple_attention_with_mask_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_with_mask_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_with_mask_f32",
  8,
  kAbiArgs_hip_navatala_transformer_simple_attention_with_mask_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_with_mask_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_with_mask_f16 = {
  1,
  "navatala_transformer_simple_attention_with_mask_f16",
  "hip",
  "navatala_transformer_simple_attention_with_mask_f16",
  "kernel:hip:navatala_transformer_simple_attention_with_mask_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_with_mask_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_with_mask_f16",
  8,
  kAbiArgs_hip_navatala_transformer_simple_attention_with_mask_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_with_padding_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_with_padding_f32 = {
  1,
  "navatala_transformer_simple_attention_with_padding_f32",
  "hip",
  "navatala_transformer_simple_attention_with_padding_f32",
  "kernel:hip:navatala_transformer_simple_attention_with_padding_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_with_padding_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_with_padding_f32",
  9,
  kAbiArgs_hip_navatala_transformer_simple_attention_with_padding_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_with_padding_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_with_padding_f16 = {
  1,
  "navatala_transformer_simple_attention_with_padding_f16",
  "hip",
  "navatala_transformer_simple_attention_with_padding_f16",
  "kernel:hip:navatala_transformer_simple_attention_with_padding_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_with_padding_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_with_padding_f16",
  9,
  kAbiArgs_hip_navatala_transformer_simple_attention_with_padding_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradQuery", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKey", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradValue", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_backward_f32 = {
  1,
  "navatala_transformer_simple_attention_backward_f32",
  "hip",
  "navatala_transformer_simple_attention_backward_f32",
  "kernel:hip:navatala_transformer_simple_attention_backward_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_backward_f32",
  "abi-r1:hip:navatala_transformer_simple_attention_backward_f32",
  11,
  kAbiArgs_hip_navatala_transformer_simple_attention_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_simple_attention_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradQuery", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKey", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradValue", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_simple_attention_backward_f16 = {
  1,
  "navatala_transformer_simple_attention_backward_f16",
  "hip",
  "navatala_transformer_simple_attention_backward_f16",
  "kernel:hip:navatala_transformer_simple_attention_backward_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_backward_f16",
  "abi-r1:hip:navatala_transformer_simple_attention_backward_f16",
  11,
  kAbiArgs_hip_navatala_transformer_simple_attention_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_kv_cache_append_f16[] = {
  { "newK", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "blockTables", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqPositions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numHeads", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_kv_cache_append_f16 = {
  1,
  "navatala_transformer_kv_cache_append_f16",
  "hip",
  "navatala_transformer_kv_cache_append_f16",
  "kernel:hip:navatala_transformer_kv_cache_append_f16",
  "abi-r1:hip:navatala_transformer_kv_cache_append_f16",
  "abi-r1:hip:navatala_transformer_kv_cache_append_f16",
  9,
  kAbiArgs_hip_navatala_transformer_kv_cache_append_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_kv_cache_append_f32[] = {
  { "newK", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "newV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "blockTables", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqPositions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numHeads", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_kv_cache_append_f32 = {
  1,
  "navatala_transformer_kv_cache_append_f32",
  "hip",
  "navatala_transformer_kv_cache_append_f32",
  "kernel:hip:navatala_transformer_kv_cache_append_f32",
  "abi-r1:hip:navatala_transformer_kv_cache_append_f32",
  "abi-r1:hip:navatala_transformer_kv_cache_append_f32",
  9,
  kAbiArgs_hip_navatala_transformer_kv_cache_append_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_kv_cache_rotate_f16[] = {
  { "blockTables", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contextLens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shiftAmount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_kv_cache_rotate_f16 = {
  1,
  "navatala_transformer_kv_cache_rotate_f16",
  "hip",
  "navatala_transformer_kv_cache_rotate_f16",
  "kernel:hip:navatala_transformer_kv_cache_rotate_f16",
  "abi-r1:hip:navatala_transformer_kv_cache_rotate_f16",
  "abi-r1:hip:navatala_transformer_kv_cache_rotate_f16",
  8,
  kAbiArgs_hip_navatala_transformer_kv_cache_rotate_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16 = {
  1,
  "navatala_transformer_tiled_gemm_f16",
  "hip",
  "navatala_transformer_tiled_gemm_f16",
  "kernel:hip:navatala_transformer_tiled_gemm_f16",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16",
  8,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_f32_out[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_f32_out = {
  1,
  "navatala_transformer_tiled_gemm_f16_f32_out",
  "hip",
  "navatala_transformer_tiled_gemm_f16_f32_out",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_f32_out",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_f32_out",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_f32_out",
  8,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_f32_out
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f32 = {
  1,
  "navatala_transformer_tiled_gemm_f32",
  "hip",
  "navatala_transformer_tiled_gemm_f32",
  "kernel:hip:navatala_transformer_tiled_gemm_f32",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f32",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f32",
  8,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_backward_f32[] = {
  { "gradC", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradA", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_backward_f32 = {
  1,
  "navatala_transformer_tiled_gemm_backward_f32",
  "hip",
  "navatala_transformer_tiled_gemm_backward_f32",
  "kernel:hip:navatala_transformer_tiled_gemm_backward_f32",
  "abi-r1:hip:navatala_transformer_tiled_gemm_backward_f32",
  "abi-r1:hip:navatala_transformer_tiled_gemm_backward_f32",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_backward_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_backward_f16[] = {
  { "gradC", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradA", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_backward_f16 = {
  1,
  "navatala_transformer_tiled_gemm_backward_f16",
  "hip",
  "navatala_transformer_tiled_gemm_backward_f16",
  "kernel:hip:navatala_transformer_tiled_gemm_backward_f16",
  "abi-r1:hip:navatala_transformer_tiled_gemm_backward_f16",
  "abi-r1:hip:navatala_transformer_tiled_gemm_backward_f16",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_backward_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_scale_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_scale_f32 = {
  1,
  "navatala_transformer_multi_tensor_scale_f32",
  "hip",
  "navatala_transformer_multi_tensor_scale_f32",
  "kernel:hip:navatala_transformer_multi_tensor_scale_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_scale_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_scale_f32",
  4,
  kAbiArgs_hip_navatala_transformer_multi_tensor_scale_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_scale_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_scale_f16 = {
  1,
  "navatala_transformer_multi_tensor_scale_f16",
  "hip",
  "navatala_transformer_multi_tensor_scale_f16",
  "kernel:hip:navatala_transformer_multi_tensor_scale_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_scale_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_scale_f16",
  4,
  kAbiArgs_hip_navatala_transformer_multi_tensor_scale_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_add_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_add_f32 = {
  1,
  "navatala_transformer_multi_tensor_add_f32",
  "hip",
  "navatala_transformer_multi_tensor_add_f32",
  "kernel:hip:navatala_transformer_multi_tensor_add_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_add_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_add_f32",
  4,
  kAbiArgs_hip_navatala_transformer_multi_tensor_add_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_add_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_add_f16 = {
  1,
  "navatala_transformer_multi_tensor_add_f16",
  "hip",
  "navatala_transformer_multi_tensor_add_f16",
  "kernel:hip:navatala_transformer_multi_tensor_add_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_add_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_add_f16",
  4,
  kAbiArgs_hip_navatala_transformer_multi_tensor_add_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_copy_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_copy_f32 = {
  1,
  "navatala_transformer_multi_tensor_copy_f32",
  "hip",
  "navatala_transformer_multi_tensor_copy_f32",
  "kernel:hip:navatala_transformer_multi_tensor_copy_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_copy_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_copy_f32",
  3,
  kAbiArgs_hip_navatala_transformer_multi_tensor_copy_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_copy_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_copy_f16 = {
  1,
  "navatala_transformer_multi_tensor_copy_f16",
  "hip",
  "navatala_transformer_multi_tensor_copy_f16",
  "kernel:hip:navatala_transformer_multi_tensor_copy_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_copy_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_copy_f16",
  3,
  kAbiArgs_hip_navatala_transformer_multi_tensor_copy_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_l2_norm_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partialSums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_l2_norm_f32 = {
  1,
  "navatala_transformer_multi_tensor_l2_norm_f32",
  "hip",
  "navatala_transformer_multi_tensor_l2_norm_f32",
  "kernel:hip:navatala_transformer_multi_tensor_l2_norm_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_l2_norm_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_l2_norm_f32",
  3,
  kAbiArgs_hip_navatala_transformer_multi_tensor_l2_norm_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_l2_norm_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partialSums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_l2_norm_f16 = {
  1,
  "navatala_transformer_multi_tensor_l2_norm_f16",
  "hip",
  "navatala_transformer_multi_tensor_l2_norm_f16",
  "kernel:hip:navatala_transformer_multi_tensor_l2_norm_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_l2_norm_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_l2_norm_f16",
  3,
  kAbiArgs_hip_navatala_transformer_multi_tensor_l2_norm_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_clip_grad_f32[] = {
  { "totalNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_clip_grad_f32 = {
  1,
  "navatala_transformer_multi_tensor_clip_grad_f32",
  "hip",
  "navatala_transformer_multi_tensor_clip_grad_f32",
  "kernel:hip:navatala_transformer_multi_tensor_clip_grad_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_clip_grad_f32",
  "abi-r1:hip:navatala_transformer_multi_tensor_clip_grad_f32",
  4,
  kAbiArgs_hip_navatala_transformer_multi_tensor_clip_grad_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_multi_tensor_clip_grad_f16[] = {
  { "totalNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_multi_tensor_clip_grad_f16 = {
  1,
  "navatala_transformer_multi_tensor_clip_grad_f16",
  "hip",
  "navatala_transformer_multi_tensor_clip_grad_f16",
  "kernel:hip:navatala_transformer_multi_tensor_clip_grad_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_clip_grad_f16",
  "abi-r1:hip:navatala_transformer_multi_tensor_clip_grad_f16",
  4,
  kAbiArgs_hip_navatala_transformer_multi_tensor_clip_grad_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_top_k_gating_f32[] = {
  { "routerLogits", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "expertIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertWeights", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_top_k_gating_f32 = {
  1,
  "navatala_transformer_top_k_gating_f32",
  "hip",
  "navatala_transformer_top_k_gating_f32",
  "kernel:hip:navatala_transformer_top_k_gating_f32",
  "abi-r1:hip:navatala_transformer_top_k_gating_f32",
  "abi-r1:hip:navatala_transformer_top_k_gating_f32",
  6,
  kAbiArgs_hip_navatala_transformer_top_k_gating_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_top_k_gating_f16[] = {
  { "routerLogits", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "expertIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertWeights", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_top_k_gating_f16 = {
  1,
  "navatala_transformer_top_k_gating_f16",
  "hip",
  "navatala_transformer_top_k_gating_f16",
  "kernel:hip:navatala_transformer_top_k_gating_f16",
  "abi-r1:hip:navatala_transformer_top_k_gating_f16",
  "abi-r1:hip:navatala_transformer_top_k_gating_f16",
  6,
  kAbiArgs_hip_navatala_transformer_top_k_gating_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_expert_capacity_mask_f32[] = {
  { "expertIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "capacity", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mask", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_expert_capacity_mask_f32 = {
  1,
  "navatala_transformer_expert_capacity_mask_f32",
  "hip",
  "navatala_transformer_expert_capacity_mask_f32",
  "kernel:hip:navatala_transformer_expert_capacity_mask_f32",
  "abi-r1:hip:navatala_transformer_expert_capacity_mask_f32",
  "abi-r1:hip:navatala_transformer_expert_capacity_mask_f32",
  6,
  kAbiArgs_hip_navatala_transformer_expert_capacity_mask_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_expert_capacity_mask_f16[] = {
  { "expertIndices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "capacity", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mask", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_expert_capacity_mask_f16 = {
  1,
  "navatala_transformer_expert_capacity_mask_f16",
  "hip",
  "navatala_transformer_expert_capacity_mask_f16",
  "kernel:hip:navatala_transformer_expert_capacity_mask_f16",
  "abi-r1:hip:navatala_transformer_expert_capacity_mask_f16",
  "abi-r1:hip:navatala_transformer_expert_capacity_mask_f16",
  6,
  kAbiArgs_hip_navatala_transformer_expert_capacity_mask_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_permute_tokens_f32[] = {
  { "hiddenStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertCounters", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "permutedStates", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_permute_tokens_f32 = {
  1,
  "navatala_transformer_permute_tokens_f32",
  "hip",
  "navatala_transformer_permute_tokens_f32",
  "kernel:hip:navatala_transformer_permute_tokens_f32",
  "abi-r1:hip:navatala_transformer_permute_tokens_f32",
  "abi-r1:hip:navatala_transformer_permute_tokens_f32",
  9,
  kAbiArgs_hip_navatala_transformer_permute_tokens_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_permute_tokens_f16[] = {
  { "hiddenStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertCounters", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "permutedStates", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_permute_tokens_f16 = {
  1,
  "navatala_transformer_permute_tokens_f16",
  "hip",
  "navatala_transformer_permute_tokens_f16",
  "kernel:hip:navatala_transformer_permute_tokens_f16",
  "abi-r1:hip:navatala_transformer_permute_tokens_f16",
  "abi-r1:hip:navatala_transformer_permute_tokens_f16",
  9,
  kAbiArgs_hip_navatala_transformer_permute_tokens_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_unpermute_tokens_f32[] = {
  { "permutedStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenStates", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_unpermute_tokens_f32 = {
  1,
  "navatala_transformer_unpermute_tokens_f32",
  "hip",
  "navatala_transformer_unpermute_tokens_f32",
  "kernel:hip:navatala_transformer_unpermute_tokens_f32",
  "abi-r1:hip:navatala_transformer_unpermute_tokens_f32",
  "abi-r1:hip:navatala_transformer_unpermute_tokens_f32",
  5,
  kAbiArgs_hip_navatala_transformer_unpermute_tokens_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_unpermute_tokens_f16[] = {
  { "permutedStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenStates", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_unpermute_tokens_f16 = {
  1,
  "navatala_transformer_unpermute_tokens_f16",
  "hip",
  "navatala_transformer_unpermute_tokens_f16",
  "kernel:hip:navatala_transformer_unpermute_tokens_f16",
  "abi-r1:hip:navatala_transformer_unpermute_tokens_f16",
  "abi-r1:hip:navatala_transformer_unpermute_tokens_f16",
  5,
  kAbiArgs_hip_navatala_transformer_unpermute_tokens_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_pad_sequence_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_pad_sequence_f32 = {
  1,
  "navatala_transformer_pad_sequence_f32",
  "hip",
  "navatala_transformer_pad_sequence_f32",
  "kernel:hip:navatala_transformer_pad_sequence_f32",
  "abi-r1:hip:navatala_transformer_pad_sequence_f32",
  "abi-r1:hip:navatala_transformer_pad_sequence_f32",
  7,
  kAbiArgs_hip_navatala_transformer_pad_sequence_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_pad_sequence_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_pad_sequence_f16 = {
  1,
  "navatala_transformer_pad_sequence_f16",
  "hip",
  "navatala_transformer_pad_sequence_f16",
  "kernel:hip:navatala_transformer_pad_sequence_f16",
  "abi-r1:hip:navatala_transformer_pad_sequence_f16",
  "abi-r1:hip:navatala_transformer_pad_sequence_f16",
  7,
  kAbiArgs_hip_navatala_transformer_pad_sequence_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_unpad_sequence_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_unpad_sequence_f32 = {
  1,
  "navatala_transformer_unpad_sequence_f32",
  "hip",
  "navatala_transformer_unpad_sequence_f32",
  "kernel:hip:navatala_transformer_unpad_sequence_f32",
  "abi-r1:hip:navatala_transformer_unpad_sequence_f32",
  "abi-r1:hip:navatala_transformer_unpad_sequence_f32",
  8,
  kAbiArgs_hip_navatala_transformer_unpad_sequence_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_unpad_sequence_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_unpad_sequence_f16 = {
  1,
  "navatala_transformer_unpad_sequence_f16",
  "hip",
  "navatala_transformer_unpad_sequence_f16",
  "kernel:hip:navatala_transformer_unpad_sequence_f16",
  "abi-r1:hip:navatala_transformer_unpad_sequence_f16",
  "abi-r1:hip:navatala_transformer_unpad_sequence_f16",
  8,
  kAbiArgs_hip_navatala_transformer_unpad_sequence_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_swizzle_layout_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tileSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_swizzle_layout_f32 = {
  1,
  "navatala_transformer_swizzle_layout_f32",
  "hip",
  "navatala_transformer_swizzle_layout_f32",
  "kernel:hip:navatala_transformer_swizzle_layout_f32",
  "abi-r1:hip:navatala_transformer_swizzle_layout_f32",
  "abi-r1:hip:navatala_transformer_swizzle_layout_f32",
  5,
  kAbiArgs_hip_navatala_transformer_swizzle_layout_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_swizzle_layout_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tileSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_swizzle_layout_f16 = {
  1,
  "navatala_transformer_swizzle_layout_f16",
  "hip",
  "navatala_transformer_swizzle_layout_f16",
  "kernel:hip:navatala_transformer_swizzle_layout_f16",
  "abi-r1:hip:navatala_transformer_swizzle_layout_f16",
  "abi-r1:hip:navatala_transformer_swizzle_layout_f16",
  5,
  kAbiArgs_hip_navatala_transformer_swizzle_layout_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_hadamard_transform_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vectorSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_hadamard_transform_f32 = {
  1,
  "navatala_transformer_hadamard_transform_f32",
  "hip",
  "navatala_transformer_hadamard_transform_f32",
  "kernel:hip:navatala_transformer_hadamard_transform_f32",
  "abi-r1:hip:navatala_transformer_hadamard_transform_f32",
  "abi-r1:hip:navatala_transformer_hadamard_transform_f32",
  4,
  kAbiArgs_hip_navatala_transformer_hadamard_transform_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_hadamard_transform_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vectorSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_hadamard_transform_f16 = {
  1,
  "navatala_transformer_hadamard_transform_f16",
  "hip",
  "navatala_transformer_hadamard_transform_f16",
  "kernel:hip:navatala_transformer_hadamard_transform_f16",
  "abi-r1:hip:navatala_transformer_hadamard_transform_f16",
  "abi-r1:hip:navatala_transformer_hadamard_transform_f16",
  4,
  kAbiArgs_hip_navatala_transformer_hadamard_transform_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_permute_axes_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_permute_axes_f32 = {
  1,
  "navatala_transformer_permute_axes_f32",
  "hip",
  "navatala_transformer_permute_axes_f32",
  "kernel:hip:navatala_transformer_permute_axes_f32",
  "abi-r1:hip:navatala_transformer_permute_axes_f32",
  "abi-r1:hip:navatala_transformer_permute_axes_f32",
  10,
  kAbiArgs_hip_navatala_transformer_permute_axes_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_permute_axes_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dim0", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim1", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim2", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim3", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm0", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm2", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "perm3", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_permute_axes_f16 = {
  1,
  "navatala_transformer_permute_axes_f16",
  "hip",
  "navatala_transformer_permute_axes_f16",
  "kernel:hip:navatala_transformer_permute_axes_f16",
  "abi-r1:hip:navatala_transformer_permute_axes_f16",
  "abi-r1:hip:navatala_transformer_permute_axes_f16",
  10,
  kAbiArgs_hip_navatala_transformer_permute_axes_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_pack_sequences_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_pack_sequences_f32 = {
  1,
  "navatala_transformer_pack_sequences_f32",
  "hip",
  "navatala_transformer_pack_sequences_f32",
  "kernel:hip:navatala_transformer_pack_sequences_f32",
  "abi-r1:hip:navatala_transformer_pack_sequences_f32",
  "abi-r1:hip:navatala_transformer_pack_sequences_f32",
  8,
  kAbiArgs_hip_navatala_transformer_pack_sequences_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_pack_sequences_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_pack_sequences_f16 = {
  1,
  "navatala_transformer_pack_sequences_f16",
  "hip",
  "navatala_transformer_pack_sequences_f16",
  "kernel:hip:navatala_transformer_pack_sequences_f16",
  "abi-r1:hip:navatala_transformer_pack_sequences_f16",
  "abi-r1:hip:navatala_transformer_pack_sequences_f16",
  8,
  kAbiArgs_hip_navatala_transformer_pack_sequences_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_unpack_sequences_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_unpack_sequences_f32 = {
  1,
  "navatala_transformer_unpack_sequences_f32",
  "hip",
  "navatala_transformer_unpack_sequences_f32",
  "kernel:hip:navatala_transformer_unpack_sequences_f32",
  "abi-r1:hip:navatala_transformer_unpack_sequences_f32",
  "abi-r1:hip:navatala_transformer_unpack_sequences_f32",
  7,
  kAbiArgs_hip_navatala_transformer_unpack_sequences_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_unpack_sequences_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_unpack_sequences_f16 = {
  1,
  "navatala_transformer_unpack_sequences_f16",
  "hip",
  "navatala_transformer_unpack_sequences_f16",
  "kernel:hip:navatala_transformer_unpack_sequences_f16",
  "abi-r1:hip:navatala_transformer_unpack_sequences_f16",
  "abi-r1:hip:navatala_transformer_unpack_sequences_f16",
  7,
  kAbiArgs_hip_navatala_transformer_unpack_sequences_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_generate_position_ids_f32[] = {
  { "batchSize", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_generate_position_ids_f32 = {
  1,
  "navatala_transformer_generate_position_ids_f32",
  "hip",
  "navatala_transformer_generate_position_ids_f32",
  "kernel:hip:navatala_transformer_generate_position_ids_f32",
  "abi-r1:hip:navatala_transformer_generate_position_ids_f32",
  "abi-r1:hip:navatala_transformer_generate_position_ids_f32",
  3,
  kAbiArgs_hip_navatala_transformer_generate_position_ids_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_generate_position_ids_u32[] = {
  { "batchSize", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_generate_position_ids_u32 = {
  1,
  "navatala_transformer_generate_position_ids_u32",
  "hip",
  "navatala_transformer_generate_position_ids_u32",
  "kernel:hip:navatala_transformer_generate_position_ids_u32",
  "abi-r1:hip:navatala_transformer_generate_position_ids_u32",
  "abi-r1:hip:navatala_transformer_generate_position_ids_u32",
  3,
  kAbiArgs_hip_navatala_transformer_generate_position_ids_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_generate_causal_mask_f32[] = {
  { "seqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_generate_causal_mask_f32 = {
  1,
  "navatala_transformer_generate_causal_mask_f32",
  "hip",
  "navatala_transformer_generate_causal_mask_f32",
  "kernel:hip:navatala_transformer_generate_causal_mask_f32",
  "abi-r1:hip:navatala_transformer_generate_causal_mask_f32",
  "abi-r1:hip:navatala_transformer_generate_causal_mask_f32",
  2,
  kAbiArgs_hip_navatala_transformer_generate_causal_mask_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_generate_causal_mask_f16[] = {
  { "seqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_generate_causal_mask_f16 = {
  1,
  "navatala_transformer_generate_causal_mask_f16",
  "hip",
  "navatala_transformer_generate_causal_mask_f16",
  "kernel:hip:navatala_transformer_generate_causal_mask_f16",
  "abi-r1:hip:navatala_transformer_generate_causal_mask_f16",
  "abi-r1:hip:navatala_transformer_generate_causal_mask_f16",
  2,
  kAbiArgs_hip_navatala_transformer_generate_causal_mask_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_precompute_rotary_emb_f32[] = {
  { "maxSeqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cosOutput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinOutput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_precompute_rotary_emb_f32 = {
  1,
  "navatala_transformer_precompute_rotary_emb_f32",
  "hip",
  "navatala_transformer_precompute_rotary_emb_f32",
  "kernel:hip:navatala_transformer_precompute_rotary_emb_f32",
  "abi-r1:hip:navatala_transformer_precompute_rotary_emb_f32",
  "abi-r1:hip:navatala_transformer_precompute_rotary_emb_f32",
  5,
  kAbiArgs_hip_navatala_transformer_precompute_rotary_emb_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_precompute_rotary_emb_f16[] = {
  { "maxSeqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cosOutput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinOutput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_precompute_rotary_emb_f16 = {
  1,
  "navatala_transformer_precompute_rotary_emb_f16",
  "hip",
  "navatala_transformer_precompute_rotary_emb_f16",
  "kernel:hip:navatala_transformer_precompute_rotary_emb_f16",
  "abi-r1:hip:navatala_transformer_precompute_rotary_emb_f16",
  "abi-r1:hip:navatala_transformer_precompute_rotary_emb_f16",
  5,
  kAbiArgs_hip_navatala_transformer_precompute_rotary_emb_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kStride", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma",
  5,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_k_loop[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kTiles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_k_loop = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_k_loop",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_k_loop",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_k_loop",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_k_loop",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_k_loop",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_k_loop
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_direct[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kTiles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_direct = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_direct",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_direct",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_direct",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_direct",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_direct",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_direct
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kTiles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mDim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nDim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kTiles", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aStride", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bStride", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cStride", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aBatchStride", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bBatchStride", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cBatchStride", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transA", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transB", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 16, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge",
  17,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kTiles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kTiles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kTiles", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kBlocks", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kStride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nStride", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128 = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta128",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta128",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta128",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta128",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta128",
  6,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128
};

const KernelArgumentInfo kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128_edge[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mDim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nDim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kBlocks", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aStride", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bStride", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cStride", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "aBatchStride", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bBatchStride", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cBatchStride", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transA", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "transB", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 16, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128_edge = {
  1,
  "navatala_transformer_tiled_gemm_f16_mfma_cta128_edge",
  "hip",
  "navatala_transformer_tiled_gemm_f16_mfma_cta128_edge",
  "kernel:hip:navatala_transformer_tiled_gemm_f16_mfma_cta128_edge",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta128_edge",
  "abi-r1:hip:navatala_transformer_tiled_gemm_f16_mfma_cta128_edge",
  17,
  kAbiArgs_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128_edge
};

bool tryGetKernelAbiManifest_hip_transformer(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_gelu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_tanh_f32") {
    out = &kAbiManifest_hip_navatala_transformer_gelu_tanh_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_silu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_relu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_quick_gelu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_squared_relu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_squared_relu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_bias_gelu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_bias_gelu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_gelu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_silu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_relu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_quick_gelu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_squared_relu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_squared_relu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_gelu_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_silu_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_geglu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_geglu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swiglu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_swiglu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_reglu_f32") {
    out = &kAbiManifest_hip_navatala_transformer_reglu_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_geglu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_geglu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swiglu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_swiglu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_reglu_f16") {
    out = &kAbiManifest_hip_navatala_transformer_reglu_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_gelu_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_silu_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_relu_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_quick_gelu_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_gelu_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_silu_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_relu_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_quick_gelu_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_forward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_forward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_forward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_forward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_forward_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_forward_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_forward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_forward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_forward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_forward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_forward_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_forward_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_multi_pass_f32") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_multi_pass_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_multi_pass_f16") {
    out = &kAbiManifest_hip_navatala_transformer_layer_norm_multi_pass_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_multi_pass_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_multi_pass_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_multi_pass_f16") {
    out = &kAbiManifest_hip_navatala_transformer_rms_norm_multi_pass_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_forward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_forward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_forward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_forward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_forward_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_forward_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_f32") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_with_mask_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_f16") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_with_mask_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_with_mask_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_with_mask_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_multi_pass_f32") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_multi_pass_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_multi_pass_f16") {
    out = &kAbiManifest_hip_navatala_transformer_softmax_multi_pass_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_f16") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f32_to_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f16_to_f32") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f16_to_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f32_to_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_b_f16_to_f32") {
    out = &kAbiManifest_hip_navatala_transformer_cast_b_f16_to_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose_f32") {
    out = &kAbiManifest_hip_navatala_transformer_transpose_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose_f16") {
    out = &kAbiManifest_hip_navatala_transformer_transpose_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_fused_cast_transpose_f32_to_f16") {
    out = &kAbiManifest_hip_navatala_transformer_fused_cast_transpose_f32_to_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e4_m3") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f32_to_f_p8_e4_m3;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f_p8_e4_m3_to_f32") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f_p8_e4_m3_to_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e5_m2") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f32_to_f_p8_e5_m2;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f_p8_e5_m2_to_f32") {
    out = &kAbiManifest_hip_navatala_transformer_cast_f_p8_e5_m2_to_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_fused_cast_transpose_f_p8") {
    out = &kAbiManifest_hip_navatala_transformer_fused_cast_transpose_f_p8;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose4_d_f32") {
    out = &kAbiManifest_hip_navatala_transformer_transpose4_d_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose4_d_f16") {
    out = &kAbiManifest_hip_navatala_transformer_transpose4_d_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_forward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_dropout_forward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_forward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_dropout_forward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_dropout_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_dropout_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_forward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rope_forward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_forward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_rope_forward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_forward_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_rope_forward_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_with_cache_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rope_with_cache_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rope_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_rope_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_backward_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_rope_backward_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_interleaved_f32") {
    out = &kAbiManifest_hip_navatala_transformer_rope_interleaved_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_interleaved_f16") {
    out = &kAbiManifest_hip_navatala_transformer_rope_interleaved_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_interleaved_bf16") {
    out = &kAbiManifest_hip_navatala_transformer_rope_interleaved_bf16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_f16") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_f32") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_paged_attention_f32") {
    out = &kAbiManifest_hip_navatala_transformer_paged_attention_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_mask_f32") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_with_mask_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_mask_f16") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_with_mask_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_padding_f32") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_with_padding_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_padding_f16") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_with_padding_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_simple_attention_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_kv_cache_append_f16") {
    out = &kAbiManifest_hip_navatala_transformer_kv_cache_append_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_kv_cache_append_f32") {
    out = &kAbiManifest_hip_navatala_transformer_kv_cache_append_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_kv_cache_rotate_f16") {
    out = &kAbiManifest_hip_navatala_transformer_kv_cache_rotate_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_f32_out;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f32") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_backward_f32") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_backward_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_backward_f16") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_backward_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_scale_f32") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_scale_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_scale_f16") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_scale_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_add_f32") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_add_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_add_f16") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_add_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_copy_f32") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_copy_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_copy_f16") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_copy_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f32") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_l2_norm_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f16") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_l2_norm_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f32") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_clip_grad_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f16") {
    out = &kAbiManifest_hip_navatala_transformer_multi_tensor_clip_grad_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_top_k_gating_f32") {
    out = &kAbiManifest_hip_navatala_transformer_top_k_gating_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_top_k_gating_f16") {
    out = &kAbiManifest_hip_navatala_transformer_top_k_gating_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_expert_capacity_mask_f32") {
    out = &kAbiManifest_hip_navatala_transformer_expert_capacity_mask_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_expert_capacity_mask_f16") {
    out = &kAbiManifest_hip_navatala_transformer_expert_capacity_mask_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_tokens_f32") {
    out = &kAbiManifest_hip_navatala_transformer_permute_tokens_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_tokens_f16") {
    out = &kAbiManifest_hip_navatala_transformer_permute_tokens_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpermute_tokens_f32") {
    out = &kAbiManifest_hip_navatala_transformer_unpermute_tokens_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpermute_tokens_f16") {
    out = &kAbiManifest_hip_navatala_transformer_unpermute_tokens_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pad_sequence_f32") {
    out = &kAbiManifest_hip_navatala_transformer_pad_sequence_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pad_sequence_f16") {
    out = &kAbiManifest_hip_navatala_transformer_pad_sequence_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpad_sequence_f32") {
    out = &kAbiManifest_hip_navatala_transformer_unpad_sequence_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpad_sequence_f16") {
    out = &kAbiManifest_hip_navatala_transformer_unpad_sequence_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swizzle_layout_f32") {
    out = &kAbiManifest_hip_navatala_transformer_swizzle_layout_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swizzle_layout_f16") {
    out = &kAbiManifest_hip_navatala_transformer_swizzle_layout_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_hadamard_transform_f32") {
    out = &kAbiManifest_hip_navatala_transformer_hadamard_transform_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_hadamard_transform_f16") {
    out = &kAbiManifest_hip_navatala_transformer_hadamard_transform_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_axes_f32") {
    out = &kAbiManifest_hip_navatala_transformer_permute_axes_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_axes_f16") {
    out = &kAbiManifest_hip_navatala_transformer_permute_axes_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pack_sequences_f32") {
    out = &kAbiManifest_hip_navatala_transformer_pack_sequences_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pack_sequences_f16") {
    out = &kAbiManifest_hip_navatala_transformer_pack_sequences_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpack_sequences_f32") {
    out = &kAbiManifest_hip_navatala_transformer_unpack_sequences_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpack_sequences_f16") {
    out = &kAbiManifest_hip_navatala_transformer_unpack_sequences_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_position_ids_f32") {
    out = &kAbiManifest_hip_navatala_transformer_generate_position_ids_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_position_ids_u32") {
    out = &kAbiManifest_hip_navatala_transformer_generate_position_ids_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_causal_mask_f32") {
    out = &kAbiManifest_hip_navatala_transformer_generate_causal_mask_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_causal_mask_f16") {
    out = &kAbiManifest_hip_navatala_transformer_generate_causal_mask_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_precompute_rotary_emb_f32") {
    out = &kAbiManifest_hip_navatala_transformer_precompute_rotary_emb_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_precompute_rotary_emb_f16") {
    out = &kAbiManifest_hip_navatala_transformer_precompute_rotary_emb_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_k_loop") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_k_loop;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_direct") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_direct;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta128") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta128_edge") {
    out = &kAbiManifest_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128_edge;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_transformer(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_gelu_f32";
    std::string_view sv(k_hip_navatala_transformer_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_gelu_tanh_f32";
    std::string_view sv(k_hip_navatala_transformer_gelu_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_silu_f32";
    std::string_view sv(k_hip_navatala_transformer_silu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_relu_f32";
    std::string_view sv(k_hip_navatala_transformer_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_quick_gelu_f32";
    std::string_view sv(k_hip_navatala_transformer_quick_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_squared_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_squared_relu_f32";
    std::string_view sv(k_hip_navatala_transformer_squared_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_bias_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_bias_gelu_f32";
    std::string_view sv(k_hip_navatala_transformer_bias_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_gelu_f16";
    std::string_view sv(k_hip_navatala_transformer_gelu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_silu_f16";
    std::string_view sv(k_hip_navatala_transformer_silu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_relu_f16";
    std::string_view sv(k_hip_navatala_transformer_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_quick_gelu_f16";
    std::string_view sv(k_hip_navatala_transformer_quick_gelu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_squared_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_squared_relu_f16";
    std::string_view sv(k_hip_navatala_transformer_squared_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_gelu_bf16";
    std::string_view sv(k_hip_navatala_transformer_gelu_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_silu_bf16";
    std::string_view sv(k_hip_navatala_transformer_silu_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_geglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_geglu_f32";
    std::string_view sv(k_hip_navatala_transformer_geglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swiglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_swiglu_f32";
    std::string_view sv(k_hip_navatala_transformer_swiglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_reglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_reglu_f32";
    std::string_view sv(k_hip_navatala_transformer_reglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_geglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_geglu_f16";
    std::string_view sv(k_hip_navatala_transformer_geglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swiglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_swiglu_f16";
    std::string_view sv(k_hip_navatala_transformer_swiglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_reglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_reglu_f16";
    std::string_view sv(k_hip_navatala_transformer_reglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_gelu_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_gelu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_silu_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_silu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_relu_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_relu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_quick_gelu_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_quick_gelu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_gelu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_gelu_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_gelu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_silu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_silu_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_silu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_relu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_relu_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_relu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_quick_gelu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_quick_gelu_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_quick_gelu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_forward_f32";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_forward_f16";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_forward_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_forward_bf16";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_forward_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_forward_f32";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_forward_f16";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_forward_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_forward_bf16";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_forward_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_multi_pass_f32";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_layer_norm_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_layer_norm_multi_pass_f16";
    std::string_view sv(k_hip_navatala_transformer_layer_norm_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_multi_pass_f32";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rms_norm_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rms_norm_multi_pass_f16";
    std::string_view sv(k_hip_navatala_transformer_rms_norm_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_forward_f32";
    std::string_view sv(k_hip_navatala_transformer_softmax_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_forward_f16";
    std::string_view sv(k_hip_navatala_transformer_softmax_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_forward_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_forward_bf16";
    std::string_view sv(k_hip_navatala_transformer_softmax_forward_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_softmax_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_softmax_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_with_mask_f32";
    std::string_view sv(k_hip_navatala_transformer_softmax_with_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_with_mask_f16";
    std::string_view sv(k_hip_navatala_transformer_softmax_with_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_with_mask_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_softmax_with_mask_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_with_mask_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_with_mask_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_softmax_with_mask_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_multi_pass_f32";
    std::string_view sv(k_hip_navatala_transformer_softmax_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_softmax_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_softmax_multi_pass_f16";
    std::string_view sv(k_hip_navatala_transformer_softmax_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f32_to_f16";
    std::string_view sv(k_hip_navatala_transformer_cast_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f16_to_f32";
    std::string_view sv(k_hip_navatala_transformer_cast_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f32_to_bf16";
    std::string_view sv(k_hip_navatala_transformer_cast_f32_to_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_b_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_b_f16_to_f32";
    std::string_view sv(k_hip_navatala_transformer_cast_b_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_transpose_f32";
    std::string_view sv(k_hip_navatala_transformer_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_transpose_f16";
    std::string_view sv(k_hip_navatala_transformer_transpose_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_fused_cast_transpose_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_fused_cast_transpose_f32_to_f16";
    std::string_view sv(k_hip_navatala_transformer_fused_cast_transpose_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e4_m3") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f32_to_f_p8_e4_m3";
    std::string_view sv(k_hip_navatala_transformer_cast_f32_to_f_p8_e4_m3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f_p8_e4_m3_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f_p8_e4_m3_to_f32";
    std::string_view sv(k_hip_navatala_transformer_cast_f_p8_e4_m3_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e5_m2") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f32_to_f_p8_e5_m2";
    std::string_view sv(k_hip_navatala_transformer_cast_f32_to_f_p8_e5_m2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_cast_f_p8_e5_m2_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_cast_f_p8_e5_m2_to_f32";
    std::string_view sv(k_hip_navatala_transformer_cast_f_p8_e5_m2_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_fused_cast_transpose_f_p8") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_fused_cast_transpose_f_p8";
    std::string_view sv(k_hip_navatala_transformer_fused_cast_transpose_f_p8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose4_d_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_transpose4_d_f32";
    std::string_view sv(k_hip_navatala_transformer_transpose4_d_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_transpose4_d_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_transpose4_d_f16";
    std::string_view sv(k_hip_navatala_transformer_transpose4_d_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_dropout_forward_f32";
    std::string_view sv(k_hip_navatala_transformer_dropout_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_dropout_forward_f16";
    std::string_view sv(k_hip_navatala_transformer_dropout_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_dropout_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_dropout_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_dropout_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_dropout_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_dropout_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_forward_f32";
    std::string_view sv(k_hip_navatala_transformer_rope_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_forward_f16";
    std::string_view sv(k_hip_navatala_transformer_rope_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_forward_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_forward_bf16";
    std::string_view sv(k_hip_navatala_transformer_rope_forward_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_with_cache_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_with_cache_f32";
    std::string_view sv(k_hip_navatala_transformer_rope_with_cache_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_rope_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_rope_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_backward_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_backward_bf16";
    std::string_view sv(k_hip_navatala_transformer_rope_backward_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_interleaved_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_interleaved_f32";
    std::string_view sv(k_hip_navatala_transformer_rope_interleaved_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_interleaved_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_interleaved_f16";
    std::string_view sv(k_hip_navatala_transformer_rope_interleaved_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_rope_interleaved_bf16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_rope_interleaved_bf16";
    std::string_view sv(k_hip_navatala_transformer_rope_interleaved_bf16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_f16";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_f32";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_paged_attention_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_paged_attention_f32";
    std::string_view sv(k_hip_navatala_transformer_paged_attention_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_with_mask_f32";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_with_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_with_mask_f16";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_with_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_padding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_with_padding_f32";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_with_padding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_with_padding_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_with_padding_f16";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_with_padding_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_simple_attention_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_simple_attention_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_simple_attention_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_kv_cache_append_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_kv_cache_append_f16";
    std::string_view sv(k_hip_navatala_transformer_kv_cache_append_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_kv_cache_append_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_kv_cache_append_f32";
    std::string_view sv(k_hip_navatala_transformer_kv_cache_append_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_kv_cache_rotate_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_kv_cache_rotate_f16";
    std::string_view sv(k_hip_navatala_transformer_kv_cache_rotate_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f32";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_backward_f32";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_backward_f16";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_scale_f32";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_scale_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_scale_f16";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_scale_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_add_f32";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_add_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_add_f16";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_add_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_copy_f32";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_copy_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_copy_f16";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_copy_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_l2_norm_f32";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_l2_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_l2_norm_f16";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_l2_norm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_clip_grad_f32";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_clip_grad_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_multi_tensor_clip_grad_f16";
    std::string_view sv(k_hip_navatala_transformer_multi_tensor_clip_grad_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_top_k_gating_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_top_k_gating_f32";
    std::string_view sv(k_hip_navatala_transformer_top_k_gating_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_top_k_gating_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_top_k_gating_f16";
    std::string_view sv(k_hip_navatala_transformer_top_k_gating_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_expert_capacity_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_expert_capacity_mask_f32";
    std::string_view sv(k_hip_navatala_transformer_expert_capacity_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_expert_capacity_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_expert_capacity_mask_f16";
    std::string_view sv(k_hip_navatala_transformer_expert_capacity_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_tokens_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_permute_tokens_f32";
    std::string_view sv(k_hip_navatala_transformer_permute_tokens_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_tokens_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_permute_tokens_f16";
    std::string_view sv(k_hip_navatala_transformer_permute_tokens_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpermute_tokens_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_unpermute_tokens_f32";
    std::string_view sv(k_hip_navatala_transformer_unpermute_tokens_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpermute_tokens_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_unpermute_tokens_f16";
    std::string_view sv(k_hip_navatala_transformer_unpermute_tokens_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pad_sequence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_pad_sequence_f32";
    std::string_view sv(k_hip_navatala_transformer_pad_sequence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pad_sequence_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_pad_sequence_f16";
    std::string_view sv(k_hip_navatala_transformer_pad_sequence_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpad_sequence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_unpad_sequence_f32";
    std::string_view sv(k_hip_navatala_transformer_unpad_sequence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpad_sequence_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_unpad_sequence_f16";
    std::string_view sv(k_hip_navatala_transformer_unpad_sequence_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swizzle_layout_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_swizzle_layout_f32";
    std::string_view sv(k_hip_navatala_transformer_swizzle_layout_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_swizzle_layout_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_swizzle_layout_f16";
    std::string_view sv(k_hip_navatala_transformer_swizzle_layout_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_hadamard_transform_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_hadamard_transform_f32";
    std::string_view sv(k_hip_navatala_transformer_hadamard_transform_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_hadamard_transform_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_hadamard_transform_f16";
    std::string_view sv(k_hip_navatala_transformer_hadamard_transform_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_axes_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_permute_axes_f32";
    std::string_view sv(k_hip_navatala_transformer_permute_axes_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_permute_axes_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_permute_axes_f16";
    std::string_view sv(k_hip_navatala_transformer_permute_axes_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pack_sequences_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_pack_sequences_f32";
    std::string_view sv(k_hip_navatala_transformer_pack_sequences_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_pack_sequences_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_pack_sequences_f16";
    std::string_view sv(k_hip_navatala_transformer_pack_sequences_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpack_sequences_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_unpack_sequences_f32";
    std::string_view sv(k_hip_navatala_transformer_unpack_sequences_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_unpack_sequences_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_unpack_sequences_f16";
    std::string_view sv(k_hip_navatala_transformer_unpack_sequences_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_position_ids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_generate_position_ids_f32";
    std::string_view sv(k_hip_navatala_transformer_generate_position_ids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_position_ids_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_generate_position_ids_u32";
    std::string_view sv(k_hip_navatala_transformer_generate_position_ids_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_causal_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_generate_causal_mask_f32";
    std::string_view sv(k_hip_navatala_transformer_generate_causal_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_generate_causal_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_generate_causal_mask_f16";
    std::string_view sv(k_hip_navatala_transformer_generate_causal_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_precompute_rotary_emb_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_precompute_rotary_emb_f32";
    std::string_view sv(k_hip_navatala_transformer_precompute_rotary_emb_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_precompute_rotary_emb_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_precompute_rotary_emb_f16";
    std::string_view sv(k_hip_navatala_transformer_precompute_rotary_emb_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_k_loop") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_k_loop";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_k_loop);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_direct") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta64_direct";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_direct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta128") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta128";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_mfma_cta128_edge") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_mfma_cta128_edge";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_mfma_cta128_edge);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_transformer_tiled_gemm_f16_f32_out") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16_f32_out";
    std::string_view sv(k_hip_navatala_transformer_tiled_gemm_f16_f32_out);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

