// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_transformer` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_transformer_gelu_f32 = R"kernel(
__kernel void navatala_transformer_gelu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_float(0x3f800000u) + tanhVal);
    float halfX = (as_float(0x3f000000u) * x);
    float result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_gelu_tanh_f32 = R"kernel(
__kernel void navatala_transformer_gelu_tanh_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float tanhArg = (as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_float(0x3f800000u) + tanhVal);
    float halfX = (as_float(0x3f000000u) * x);
    float result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_silu_f32 = R"kernel(
__kernel void navatala_transformer_silu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (as_float(0x3f800000u) + expNegX);
    float result = (x / denom);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_relu_f32 = R"kernel(
__kernel void navatala_transformer_relu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    bool isPositive = (x > as_float(0x00000000u));
    float result = ((isPositive) ? (x) : (as_float(0x00000000u)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_quick_gelu_f32 = R"kernel(
__kernel void navatala_transformer_quick_gelu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float scaled = (as_float(0x3fd9db23u) * x);
    float negScaled = (-scaled);
    float expNegScaled = exp(negScaled);
    float denom = (as_float(0x3f800000u) + expNegScaled);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float result = (x * sigmoid);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_squared_relu_f32 = R"kernel(
__kernel void navatala_transformer_squared_relu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    bool isPositive = (x > as_float(0x00000000u));
    float relu = ((isPositive) ? (x) : (as_float(0x00000000u)));
    float result = (relu * relu);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_bias_gelu_f32 = R"kernel(
__kernel void navatala_transformer_bias_gelu_f32(__global const float* _input, __global const float* bias, __global const uint* count, __global const uint* biasSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint bSize = biasSize[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    uint biasIdx = (gid % bSize);
    float b = bias[biasIdx];
    float xBias = (x + b);
    float x2 = (xBias * xBias);
    float x3 = (x2 * xBias);
    float cubeTerm = (as_float(0x3d372713u) * x3);
    float inner = (xBias + cubeTerm);
    float tanhArg = (as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_float(0x3f800000u) + tanhVal);
    float halfXBias = (as_float(0x3f000000u) * xBias);
    float result = (halfXBias * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_gelu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_gelu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half x2 = (x * x);
    half x3 = (x2 * x);
    half cubeTerm = ((half)(0.044715f) * x3);
    half inner = (x + cubeTerm);
    half tanhArg = ((half)(0.797885f) * inner);
    half tanhVal = tanh(tanhArg);
    half onePlusTanh = ((half)(1.000000f) + tanhVal);
    half halfX = ((half)(0.500000f) * x);
    half result = (halfX * onePlusTanh);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_silu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_silu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half negX = (-x);
    half expNegX = exp(negX);
    half denom = ((half)(1.000000f) + expNegX);
    half result = (x / denom);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_relu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_relu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    bool isPositive = (x > (half)(0.000000f));
    half result = ((isPositive) ? (x) : ((half)(0.000000f)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_quick_gelu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_quick_gelu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half scaled = ((half)(1.702000f) * x);
    half negScaled = (-scaled);
    half expNegScaled = exp(negScaled);
    half denom = ((half)(1.000000f) + expNegScaled);
    half sigmoid = ((half)(1.000000f) / denom);
    half result = (x * sigmoid);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_squared_relu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_squared_relu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    bool isPositive = (x > (half)(0.000000f));
    half relu = ((isPositive) ? (x) : ((half)(0.000000f)));
    half result = (relu * relu);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_geglu_f32 = R"kernel(
__kernel void navatala_transformer_geglu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float gate2 = (gate * gate);
    float gate3 = (gate2 * gate);
    float cubeTerm = (as_float(0x3d372713u) * gate3);
    float inner = (gate + cubeTerm);
    float tanhArg = (as_float(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_float(0x3f800000u) + tanhVal);
    float halfGate = (as_float(0x3f000000u) * gate);
    float geluGate = (halfGate * onePlusTanh);
    float result = (geluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_swiglu_f32 = R"kernel(
__kernel void navatala_transformer_swiglu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    float negGate = (-gate);
    float expNegGate = exp(negGate);
    float denom = (as_float(0x3f800000u) + expNegGate);
    float siluGate = (gate / denom);
    float result = (siluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_reglu_f32 = R"kernel(
__kernel void navatala_transformer_reglu_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    float gate = _input[gid];
    uint valueIdx = (gid + hidden);
    float value = _input[valueIdx];
    bool isPositive = (gate > as_float(0x00000000u));
    float reluGate = ((isPositive) ? (gate) : (as_float(0x00000000u)));
    float result = (reluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_geglu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_geglu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    half gate = _input[gid];
    uint valueIdx = (gid + hidden);
    half value = _input[valueIdx];
    half gate2 = (gate * gate);
    half gate3 = (gate2 * gate);
    half cubeTerm = ((half)(0.044715f) * gate3);
    half inner = (gate + cubeTerm);
    half tanhArg = ((half)(0.797885f) * inner);
    half tanhVal = tanh(tanhArg);
    half onePlusTanh = ((half)(1.000000f) + tanhVal);
    half halfGate = ((half)(0.500000f) * gate);
    half geluGate = (halfGate * onePlusTanh);
    half result = (geluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_swiglu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_swiglu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    half gate = _input[gid];
    uint valueIdx = (gid + hidden);
    half value = _input[valueIdx];
    half negGate = (-gate);
    half expNegGate = exp(negGate);
    half denom = ((half)(1.000000f) + expNegGate);
    half siluGate = (gate / denom);
    half result = (siluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_reglu_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_reglu_f16(__global const half* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint hidden = (n / (uint)(2u));
  bool inBounds = (gid < hidden);
  if (inBounds) {
    half gate = _input[gid];
    uint valueIdx = (gid + hidden);
    half value = _input[valueIdx];
    bool isPositive = (gate > (half)(0.000000f));
    half reluGate = ((isPositive) ? (gate) : ((half)(0.000000f)));
    half result = (reluGate * value);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_gelu_backward_f32 = R"kernel(
__kernel void navatala_transformer_gelu_backward_f32(__global const float* _input, __global const float* gradOutput, __global const uint* count, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_float(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float u = (as_float(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (as_float(0x3f800000u) - tanh2);
    float threeCoef = as_float(0x3e095d4fu);
    float derivInner = (as_float(0x3f800000u) + (threeCoef * x2));
    float term2a = (as_float(0x3f000000u) * x);
    float term2b = (term2a * sech2);
    float term2c = (term2b * as_float(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (as_float(0x3f800000u) + tanhU);
    float term1 = (as_float(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dx = (dy * geluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_silu_backward_f32 = R"kernel(
__kernel void navatala_transformer_silu_backward_f32(__global const float* _input, __global const float* gradOutput, __global const uint* count, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float negX = (-x);
    float expNegX = exp(negX);
    float denom = (as_float(0x3f800000u) + expNegX);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_float(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (x * oneMinusSigmoid);
    float bracketTerm = (as_float(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dx = (dy * siluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_relu_backward_f32 = R"kernel(
__kernel void navatala_transformer_relu_backward_f32(__global const float* _input, __global const float* gradOutput, __global const uint* count, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    bool isPositive = (x > as_float(0x00000000u));
    float dx = ((isPositive) ? (dy) : (as_float(0x00000000u)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_quick_gelu_backward_f32 = R"kernel(
__kernel void navatala_transformer_quick_gelu_backward_f32(__global const float* _input, __global const float* gradOutput, __global const uint* count, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float scaledX = (as_float(0x3fd9db23u) * x);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_float(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_float(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dx = (dy * quickGeluPrime);
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_gelu_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_gelu_backward_f16(__global const half* _input, __global const half* gradOutput, __global const uint* count, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float x2 = (xF32 * xF32);
    float x3 = (x2 * xF32);
    float cubeTerm = (as_float(0x3d372713u) * x3);
    float inner = (xF32 + cubeTerm);
    float u = (as_float(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (as_float(0x3f800000u) - tanh2);
    float threeCoef = as_float(0x3e095d4fu);
    float derivInner = (as_float(0x3f800000u) + (threeCoef * x2));
    float term2a = (as_float(0x3f000000u) * xF32);
    float term2b = (term2a * sech2);
    float term2c = (term2b * as_float(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (as_float(0x3f800000u) + tanhU);
    float term1 = (as_float(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dxF32 = (dyF32 * geluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_silu_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_silu_backward_f16(__global const half* _input, __global const half* gradOutput, __global const uint* count, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float negX = (-xF32);
    float expNegX = exp(negX);
    float denom = (as_float(0x3f800000u) + expNegX);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_float(0x3f800000u) - sigmoid);
    float xTimesOneMinusSig = (xF32 * oneMinusSigmoid);
    float bracketTerm = (as_float(0x3f800000u) + xTimesOneMinusSig);
    float siluPrime = (sigmoid * bracketTerm);
    float dxF32 = (dyF32 * siluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_relu_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_relu_backward_f16(__global const half* _input, __global const half* gradOutput, __global const uint* count, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    bool isPositive = (x > (half)(0.000000f));
    half dx = ((isPositive) ? (dy) : ((half)(0.000000f)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_quick_gelu_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_quick_gelu_backward_f16(__global const half* _input, __global const half* gradOutput, __global const uint* count, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float scaledX = (as_float(0x3fd9db23u) * xF32);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_float(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_float(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_float(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dxF32 = (dyF32 * quickGeluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rms_norm_forward_f32 = R"kernel(
__kernel void navatala_transformer_rms_norm_forward_f32(__global const float* _input, __global const float* gamma, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_float(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + (uint)(128u));
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + (uint)(64u));
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + (uint)(32u));
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + (uint)(16u));
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + (uint)(8u));
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + (uint)(4u));
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + (uint)(2u));
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + (uint)(1u));
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSq[(uint)(0u)];
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
const char* k_opencl_navatala_transformer_rms_norm_forward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rms_norm_forward_f16(__global const half* _input, __global const half* gamma, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumSq[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = ((hiddenValid) ? (xSq) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + (uint)(128u));
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + (uint)(64u));
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + (uint)(32u));
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + (uint)(16u));
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + (uint)(8u));
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + (uint)(4u));
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + (uint)(2u));
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + (uint)(1u));
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSq[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  if ((batchValid && hiddenValid)) {
    half gF16 = gamma[lid];
    float g = ((float)(gF16));
    float xNorm = (x / rms);
    float resultF32 = (g * xNorm);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_layer_norm_forward_f32 = R"kernel(
__kernel void navatala_transformer_layer_norm_forward_f32(__global const float* _input, __global const float* gamma, __global const float* beta, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumBuf[256];
  __local float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_float(0x00000000u)));
  sumBuf[lid] = ((hiddenValid) ? (x) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSum = sumBuf[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + (uint)(128u));
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + (uint)(64u));
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + (uint)(32u));
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + (uint)(16u));
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + (uint)(8u));
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + (uint)(4u));
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + (uint)(2u));
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + (uint)(1u));
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSqBuf[(uint)(0u)];
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
const char* k_opencl_navatala_transformer_layer_norm_forward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_layer_norm_forward_f16(__global const half* _input, __global const half* gamma, __global const half* beta, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumBuf[256];
  __local float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((float)(xF16));
  sumBuf[lid] = ((hiddenValid) ? (x) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSum = sumBuf[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float mean = (totalSum / hsF32);
  float xMinusMean = (x - mean);
  float xMinusMeanSq = (xMinusMean * xMinusMean);
  sumSqBuf[lid] = ((hiddenValid) ? (xMinusMeanSq) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + (uint)(128u));
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + (uint)(64u));
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + (uint)(32u));
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + (uint)(16u));
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + (uint)(8u));
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + (uint)(4u));
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + (uint)(2u));
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + (uint)(1u));
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSqBuf[(uint)(0u)];
  float variance = (totalSumSq / hsF32);
  float varEps = (variance + eps);
  float std = sqrt(varEps);
  if ((batchValid && hiddenValid)) {
    half gF16 = gamma[lid];
    half bF16 = beta[lid];
    float g = ((float)(gF16));
    float b = ((float)(bF16));
    float xNorm = (xMinusMean / std);
    float scaled = (g * xNorm);
    float resultF32 = (scaled + b);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_layer_norm_backward_f32 = R"kernel(
__kernel void navatala_transformer_layer_norm_backward_f32(__global const float* _input, __global const float* gradOutput, __global const float* gamma, __global const float* mean, __global const float* rsigma, __global const uint* batchSize, __global const uint* hiddenSize, __global float* gradInput, __global float* gradGamma, __global float* gradBeta) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  __local float sumDyGamma[256];
  __local float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_float(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (as_float(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (as_float(0x00000000u)));
  float mu = ((batchValid) ? (mean[batchIdx]) : (as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_float(0x3f800000u)));
  float xMinusMean = (x - mu);
  float xHat = (xMinusMean * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGamma[lid] = ((hiddenValid) ? (dyGamma) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGamma_128) {
    uint neighborIdx_sumDyGamma_128 = (lid + (uint)(128u));
    float myVal_sumDyGamma_128 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_128 = sumDyGamma[neighborIdx_sumDyGamma_128];
    float sumVal_sumDyGamma_128 = (myVal_sumDyGamma_128 + neighborVal_sumDyGamma_128);
    sumDyGamma[lid] = sumVal_sumDyGamma_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGamma_64) {
    uint neighborIdx_sumDyGamma_64 = (lid + (uint)(64u));
    float myVal_sumDyGamma_64 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_64 = sumDyGamma[neighborIdx_sumDyGamma_64];
    float sumVal_sumDyGamma_64 = (myVal_sumDyGamma_64 + neighborVal_sumDyGamma_64);
    sumDyGamma[lid] = sumVal_sumDyGamma_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGamma_32) {
    uint neighborIdx_sumDyGamma_32 = (lid + (uint)(32u));
    float myVal_sumDyGamma_32 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_32 = sumDyGamma[neighborIdx_sumDyGamma_32];
    float sumVal_sumDyGamma_32 = (myVal_sumDyGamma_32 + neighborVal_sumDyGamma_32);
    sumDyGamma[lid] = sumVal_sumDyGamma_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGamma_16) {
    uint neighborIdx_sumDyGamma_16 = (lid + (uint)(16u));
    float myVal_sumDyGamma_16 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_16 = sumDyGamma[neighborIdx_sumDyGamma_16];
    float sumVal_sumDyGamma_16 = (myVal_sumDyGamma_16 + neighborVal_sumDyGamma_16);
    sumDyGamma[lid] = sumVal_sumDyGamma_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGamma_8) {
    uint neighborIdx_sumDyGamma_8 = (lid + (uint)(8u));
    float myVal_sumDyGamma_8 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_8 = sumDyGamma[neighborIdx_sumDyGamma_8];
    float sumVal_sumDyGamma_8 = (myVal_sumDyGamma_8 + neighborVal_sumDyGamma_8);
    sumDyGamma[lid] = sumVal_sumDyGamma_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGamma_4) {
    uint neighborIdx_sumDyGamma_4 = (lid + (uint)(4u));
    float myVal_sumDyGamma_4 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_4 = sumDyGamma[neighborIdx_sumDyGamma_4];
    float sumVal_sumDyGamma_4 = (myVal_sumDyGamma_4 + neighborVal_sumDyGamma_4);
    sumDyGamma[lid] = sumVal_sumDyGamma_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGamma_2) {
    uint neighborIdx_sumDyGamma_2 = (lid + (uint)(2u));
    float myVal_sumDyGamma_2 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_2 = sumDyGamma[neighborIdx_sumDyGamma_2];
    float sumVal_sumDyGamma_2 = (myVal_sumDyGamma_2 + neighborVal_sumDyGamma_2);
    sumDyGamma[lid] = sumVal_sumDyGamma_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGamma_1) {
    uint neighborIdx_sumDyGamma_1 = (lid + (uint)(1u));
    float myVal_sumDyGamma_1 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_1 = sumDyGamma[neighborIdx_sumDyGamma_1];
    float sumVal_sumDyGamma_1 = (myVal_sumDyGamma_1 + neighborVal_sumDyGamma_1);
    sumDyGamma[lid] = sumVal_sumDyGamma_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGamma = sumDyGamma[(uint)(0u)];
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + (uint)(128u));
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + (uint)(64u));
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + (uint)(32u));
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + (uint)(16u));
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + (uint)(8u));
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + (uint)(4u));
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + (uint)(2u));
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + (uint)(1u));
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGammaXhat = sumDyGammaXhat[(uint)(0u)];
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
    atomic_add(&gradGamma[lid], dyXhat);
    atomic_add(&gradBeta[lid], dy);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_layer_norm_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_layer_norm_backward_f16(__global const half* _input, __global const half* gradOutput, __global const half* gamma, __global const float* mean, __global const float* rsigma, __global const uint* batchSize, __global const uint* hiddenSize, __global half* gradInput, __global float* gradGamma, __global float* gradBeta) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  __local float sumDyGamma[256];
  __local float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((float)(xF16));
  half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : ((half)(0.000000f)));
  float dy = ((float)(dyF16));
  half gF16 = ((hiddenValid) ? (gamma[lid]) : ((half)(0.000000f)));
  float g = ((float)(gF16));
  float mu = ((batchValid) ? (mean[batchIdx]) : (as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_float(0x3f800000u)));
  float xMinusMean = (x - mu);
  float xHat = (xMinusMean * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGamma[lid] = ((hiddenValid) ? (dyGamma) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGamma_128) {
    uint neighborIdx_sumDyGamma_128 = (lid + (uint)(128u));
    float myVal_sumDyGamma_128 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_128 = sumDyGamma[neighborIdx_sumDyGamma_128];
    float sumVal_sumDyGamma_128 = (myVal_sumDyGamma_128 + neighborVal_sumDyGamma_128);
    sumDyGamma[lid] = sumVal_sumDyGamma_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGamma_64) {
    uint neighborIdx_sumDyGamma_64 = (lid + (uint)(64u));
    float myVal_sumDyGamma_64 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_64 = sumDyGamma[neighborIdx_sumDyGamma_64];
    float sumVal_sumDyGamma_64 = (myVal_sumDyGamma_64 + neighborVal_sumDyGamma_64);
    sumDyGamma[lid] = sumVal_sumDyGamma_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGamma_32) {
    uint neighborIdx_sumDyGamma_32 = (lid + (uint)(32u));
    float myVal_sumDyGamma_32 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_32 = sumDyGamma[neighborIdx_sumDyGamma_32];
    float sumVal_sumDyGamma_32 = (myVal_sumDyGamma_32 + neighborVal_sumDyGamma_32);
    sumDyGamma[lid] = sumVal_sumDyGamma_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGamma_16) {
    uint neighborIdx_sumDyGamma_16 = (lid + (uint)(16u));
    float myVal_sumDyGamma_16 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_16 = sumDyGamma[neighborIdx_sumDyGamma_16];
    float sumVal_sumDyGamma_16 = (myVal_sumDyGamma_16 + neighborVal_sumDyGamma_16);
    sumDyGamma[lid] = sumVal_sumDyGamma_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGamma_8) {
    uint neighborIdx_sumDyGamma_8 = (lid + (uint)(8u));
    float myVal_sumDyGamma_8 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_8 = sumDyGamma[neighborIdx_sumDyGamma_8];
    float sumVal_sumDyGamma_8 = (myVal_sumDyGamma_8 + neighborVal_sumDyGamma_8);
    sumDyGamma[lid] = sumVal_sumDyGamma_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGamma_4) {
    uint neighborIdx_sumDyGamma_4 = (lid + (uint)(4u));
    float myVal_sumDyGamma_4 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_4 = sumDyGamma[neighborIdx_sumDyGamma_4];
    float sumVal_sumDyGamma_4 = (myVal_sumDyGamma_4 + neighborVal_sumDyGamma_4);
    sumDyGamma[lid] = sumVal_sumDyGamma_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGamma_2) {
    uint neighborIdx_sumDyGamma_2 = (lid + (uint)(2u));
    float myVal_sumDyGamma_2 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_2 = sumDyGamma[neighborIdx_sumDyGamma_2];
    float sumVal_sumDyGamma_2 = (myVal_sumDyGamma_2 + neighborVal_sumDyGamma_2);
    sumDyGamma[lid] = sumVal_sumDyGamma_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGamma_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGamma_1) {
    uint neighborIdx_sumDyGamma_1 = (lid + (uint)(1u));
    float myVal_sumDyGamma_1 = sumDyGamma[lid];
    float neighborVal_sumDyGamma_1 = sumDyGamma[neighborIdx_sumDyGamma_1];
    float sumVal_sumDyGamma_1 = (myVal_sumDyGamma_1 + neighborVal_sumDyGamma_1);
    sumDyGamma[lid] = sumVal_sumDyGamma_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGamma = sumDyGamma[(uint)(0u)];
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + (uint)(128u));
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + (uint)(64u));
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + (uint)(32u));
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + (uint)(16u));
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + (uint)(8u));
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + (uint)(4u));
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + (uint)(2u));
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + (uint)(1u));
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGammaXhat = sumDyGammaXhat[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanDyGamma = (totalDyGamma / hsF32);
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = meanDyGamma;
    float term3 = (xHat * meanDyGammaXhat);
    float inner = ((term1 - term2) - term3);
    float dxF32 = (rs * inner);
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_add(&gradGamma[lid], dyXhat);
    atomic_add(&gradBeta[lid], dy);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rms_norm_backward_f32 = R"kernel(
__kernel void navatala_transformer_rms_norm_backward_f32(__global const float* _input, __global const float* gradOutput, __global const float* gamma, __global const float* rsigma, __global const uint* batchSize, __global const uint* hiddenSize, __global float* gradInput, __global float* gradGamma) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  __local float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  float x = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : (as_float(0x00000000u)));
  float dy = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : (as_float(0x00000000u)));
  float g = ((hiddenValid) ? (gamma[lid]) : (as_float(0x00000000u)));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_float(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + (uint)(128u));
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + (uint)(64u));
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + (uint)(32u));
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + (uint)(16u));
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + (uint)(8u));
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + (uint)(4u));
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + (uint)(2u));
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + (uint)(1u));
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGammaXhat = sumDyGammaXhat[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dx = (rs * inner);
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_add(&gradGamma[lid], dyXhat);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rms_norm_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rms_norm_backward_f16(__global const half* _input, __global const half* gradOutput, __global const half* gamma, __global const float* rsigma, __global const uint* batchSize, __global const uint* hiddenSize, __global half* gradInput, __global float* gradGamma) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  __local float sumDyGammaXhat[256];
  bool batchValid = (batchIdx < bs);
  bool hiddenValid = (lid < hs);
  uint globalIdx = ((batchIdx * hs) + lid);
  half xF16 = (((batchValid && hiddenValid)) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((float)(xF16));
  half dyF16 = (((batchValid && hiddenValid)) ? (gradOutput[globalIdx]) : ((half)(0.000000f)));
  float dy = ((float)(dyF16));
  half gF16 = ((hiddenValid) ? (gamma[lid]) : ((half)(0.000000f)));
  float g = ((float)(gF16));
  float rs = ((batchValid) ? (rsigma[batchIdx]) : (as_float(0x3f800000u)));
  float xHat = (x * rs);
  float dyGamma = (dy * g);
  float dyGammaXhat = (dyGamma * xHat);
  sumDyGammaXhat[lid] = ((hiddenValid) ? (dyGammaXhat) : (as_float(0x00000000u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_128 = (lid < (uint)(128u));
  if (shouldReduce_sumDyGammaXhat_128) {
    uint neighborIdx_sumDyGammaXhat_128 = (lid + (uint)(128u));
    float myVal_sumDyGammaXhat_128 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_128 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_128];
    float sumVal_sumDyGammaXhat_128 = (myVal_sumDyGammaXhat_128 + neighborVal_sumDyGammaXhat_128);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_64 = (lid < (uint)(64u));
  if (shouldReduce_sumDyGammaXhat_64) {
    uint neighborIdx_sumDyGammaXhat_64 = (lid + (uint)(64u));
    float myVal_sumDyGammaXhat_64 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_64 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_64];
    float sumVal_sumDyGammaXhat_64 = (myVal_sumDyGammaXhat_64 + neighborVal_sumDyGammaXhat_64);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_32 = (lid < (uint)(32u));
  if (shouldReduce_sumDyGammaXhat_32) {
    uint neighborIdx_sumDyGammaXhat_32 = (lid + (uint)(32u));
    float myVal_sumDyGammaXhat_32 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_32 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_32];
    float sumVal_sumDyGammaXhat_32 = (myVal_sumDyGammaXhat_32 + neighborVal_sumDyGammaXhat_32);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_16 = (lid < (uint)(16u));
  if (shouldReduce_sumDyGammaXhat_16) {
    uint neighborIdx_sumDyGammaXhat_16 = (lid + (uint)(16u));
    float myVal_sumDyGammaXhat_16 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_16 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_16];
    float sumVal_sumDyGammaXhat_16 = (myVal_sumDyGammaXhat_16 + neighborVal_sumDyGammaXhat_16);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_8 = (lid < (uint)(8u));
  if (shouldReduce_sumDyGammaXhat_8) {
    uint neighborIdx_sumDyGammaXhat_8 = (lid + (uint)(8u));
    float myVal_sumDyGammaXhat_8 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_8 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_8];
    float sumVal_sumDyGammaXhat_8 = (myVal_sumDyGammaXhat_8 + neighborVal_sumDyGammaXhat_8);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_4 = (lid < (uint)(4u));
  if (shouldReduce_sumDyGammaXhat_4) {
    uint neighborIdx_sumDyGammaXhat_4 = (lid + (uint)(4u));
    float myVal_sumDyGammaXhat_4 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_4 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_4];
    float sumVal_sumDyGammaXhat_4 = (myVal_sumDyGammaXhat_4 + neighborVal_sumDyGammaXhat_4);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_2 = (lid < (uint)(2u));
  if (shouldReduce_sumDyGammaXhat_2) {
    uint neighborIdx_sumDyGammaXhat_2 = (lid + (uint)(2u));
    float myVal_sumDyGammaXhat_2 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_2 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_2];
    float sumVal_sumDyGammaXhat_2 = (myVal_sumDyGammaXhat_2 + neighborVal_sumDyGammaXhat_2);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumDyGammaXhat_1 = (lid < (uint)(1u));
  if (shouldReduce_sumDyGammaXhat_1) {
    uint neighborIdx_sumDyGammaXhat_1 = (lid + (uint)(1u));
    float myVal_sumDyGammaXhat_1 = sumDyGammaXhat[lid];
    float neighborVal_sumDyGammaXhat_1 = sumDyGammaXhat[neighborIdx_sumDyGammaXhat_1];
    float sumVal_sumDyGammaXhat_1 = (myVal_sumDyGammaXhat_1 + neighborVal_sumDyGammaXhat_1);
    sumDyGammaXhat[lid] = sumVal_sumDyGammaXhat_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalDyGammaXhat = sumDyGammaXhat[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanDyGammaXhat = (totalDyGammaXhat / hsF32);
  if ((batchValid && hiddenValid)) {
    float term1 = dyGamma;
    float term2 = (xHat * meanDyGammaXhat);
    float inner = (term1 - term2);
    float dxF32 = (rs * inner);
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
    float dyXhat = (dy * xHat);
    atomic_add(&gradGamma[lid], dyXhat);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_layer_norm_multi_pass_f32 = R"kernel(
__kernel void navatala_transformer_layer_norm_multi_pass_f32(__global const float* _input, __global const float* gamma, __global const float* beta, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumBuf[256];
  __local float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSum = as_float(0x00000000u);
  float partialSumSq = as_float(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = (uint)(256u);
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_float(0x00000000u)));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSum = sumBuf[(uint)(0u)];
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + (uint)(128u));
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + (uint)(64u));
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + (uint)(32u));
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + (uint)(16u));
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + (uint)(8u));
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + (uint)(4u));
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + (uint)(2u));
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + (uint)(1u));
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSqBuf[(uint)(0u)];
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
      uint globalIdx2 = (baseIdx + iterIdx);
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
const char* k_opencl_navatala_transformer_layer_norm_multi_pass_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_layer_norm_multi_pass_f16(__global const half* _input, __global const half* gamma, __global const half* beta, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumBuf[256];
  __local float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSum = as_float(0x00000000u);
  float partialSumSq = as_float(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = (uint)(256u);
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    half xF16 = ((batchValid) ? (_input[globalIdx]) : ((half)(0.000000f)));
    float xVal = ((float)(xF16));
    partialSum = (partialSum + xVal);
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  sumSqBuf[lid] = partialSumSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumBuf_128) {
    uint neighborIdx_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sumBuf_128 = sumBuf[lid];
    float neighborVal_sumBuf_128 = sumBuf[neighborIdx_sumBuf_128];
    float sumVal_sumBuf_128 = (myVal_sumBuf_128 + neighborVal_sumBuf_128);
    sumBuf[lid] = sumVal_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumBuf_64) {
    uint neighborIdx_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sumBuf_64 = sumBuf[lid];
    float neighborVal_sumBuf_64 = sumBuf[neighborIdx_sumBuf_64];
    float sumVal_sumBuf_64 = (myVal_sumBuf_64 + neighborVal_sumBuf_64);
    sumBuf[lid] = sumVal_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumBuf_32) {
    uint neighborIdx_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sumBuf_32 = sumBuf[lid];
    float neighborVal_sumBuf_32 = sumBuf[neighborIdx_sumBuf_32];
    float sumVal_sumBuf_32 = (myVal_sumBuf_32 + neighborVal_sumBuf_32);
    sumBuf[lid] = sumVal_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumBuf_16) {
    uint neighborIdx_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sumBuf_16 = sumBuf[lid];
    float neighborVal_sumBuf_16 = sumBuf[neighborIdx_sumBuf_16];
    float sumVal_sumBuf_16 = (myVal_sumBuf_16 + neighborVal_sumBuf_16);
    sumBuf[lid] = sumVal_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumBuf_8) {
    uint neighborIdx_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sumBuf_8 = sumBuf[lid];
    float neighborVal_sumBuf_8 = sumBuf[neighborIdx_sumBuf_8];
    float sumVal_sumBuf_8 = (myVal_sumBuf_8 + neighborVal_sumBuf_8);
    sumBuf[lid] = sumVal_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumBuf_4) {
    uint neighborIdx_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sumBuf_4 = sumBuf[lid];
    float neighborVal_sumBuf_4 = sumBuf[neighborIdx_sumBuf_4];
    float sumVal_sumBuf_4 = (myVal_sumBuf_4 + neighborVal_sumBuf_4);
    sumBuf[lid] = sumVal_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumBuf_2) {
    uint neighborIdx_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sumBuf_2 = sumBuf[lid];
    float neighborVal_sumBuf_2 = sumBuf[neighborIdx_sumBuf_2];
    float sumVal_sumBuf_2 = (myVal_sumBuf_2 + neighborVal_sumBuf_2);
    sumBuf[lid] = sumVal_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumBuf_1) {
    uint neighborIdx_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sumBuf_1 = sumBuf[lid];
    float neighborVal_sumBuf_1 = sumBuf[neighborIdx_sumBuf_1];
    float sumVal_sumBuf_1 = (myVal_sumBuf_1 + neighborVal_sumBuf_1);
    sumBuf[lid] = sumVal_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSum = sumBuf[(uint)(0u)];
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + (uint)(128u));
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + (uint)(64u));
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + (uint)(32u));
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + (uint)(16u));
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + (uint)(8u));
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + (uint)(4u));
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + (uint)(2u));
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + (uint)(1u));
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSqBuf[(uint)(0u)];
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
      uint globalIdx2 = (baseIdx + iterIdx);
      half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      half gF16 = gamma[iterIdx];
      half bF16 = beta[iterIdx];
      float g = ((float)(gF16));
      float b = ((float)(bF16));
      float xNorm = ((xVal2 - mean) / std);
      float scaled = (g * xNorm);
      float resultF32 = (scaled + b);
      half result = ((half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rms_norm_multi_pass_f32 = R"kernel(
__kernel void navatala_transformer_rms_norm_multi_pass_f32(__global const float* _input, __global const float* gamma, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSumSq = as_float(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = (uint)(256u);
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_float(0x00000000u)));
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumSqBuf[lid] = partialSumSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + (uint)(128u));
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + (uint)(64u));
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + (uint)(32u));
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + (uint)(16u));
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + (uint)(8u));
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + (uint)(4u));
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + (uint)(2u));
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + (uint)(1u));
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSqBuf[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
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
const char* k_opencl_navatala_transformer_rms_norm_multi_pass_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rms_norm_multi_pass_f16(__global const half* _input, __global const half* gamma, __global const float* epsilon, __global const uint* batchSize, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  float eps = epsilon[(uint)(0u)];
  __local float sumSqBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * hs);
  float partialSumSq = as_float(0x00000000u);
  uint iterIdx = lid;
  uint workgroupSize = (uint)(256u);
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    uint globalIdx = (baseIdx + iterIdx);
    half xF16 = ((batchValid) ? (_input[globalIdx]) : ((half)(0.000000f)));
    float xVal = ((float)(xF16));
    float xSq = (xVal * xVal);
    partialSumSq = (partialSumSq + xSq);
    iterIdx = (iterIdx + workgroupSize);
  }
  sumSqBuf[lid] = partialSumSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSqBuf_128) {
    uint neighborIdx_sumSqBuf_128 = (lid + (uint)(128u));
    float myVal_sumSqBuf_128 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_128 = sumSqBuf[neighborIdx_sumSqBuf_128];
    float sumVal_sumSqBuf_128 = (myVal_sumSqBuf_128 + neighborVal_sumSqBuf_128);
    sumSqBuf[lid] = sumVal_sumSqBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSqBuf_64) {
    uint neighborIdx_sumSqBuf_64 = (lid + (uint)(64u));
    float myVal_sumSqBuf_64 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_64 = sumSqBuf[neighborIdx_sumSqBuf_64];
    float sumVal_sumSqBuf_64 = (myVal_sumSqBuf_64 + neighborVal_sumSqBuf_64);
    sumSqBuf[lid] = sumVal_sumSqBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSqBuf_32) {
    uint neighborIdx_sumSqBuf_32 = (lid + (uint)(32u));
    float myVal_sumSqBuf_32 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_32 = sumSqBuf[neighborIdx_sumSqBuf_32];
    float sumVal_sumSqBuf_32 = (myVal_sumSqBuf_32 + neighborVal_sumSqBuf_32);
    sumSqBuf[lid] = sumVal_sumSqBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSqBuf_16) {
    uint neighborIdx_sumSqBuf_16 = (lid + (uint)(16u));
    float myVal_sumSqBuf_16 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_16 = sumSqBuf[neighborIdx_sumSqBuf_16];
    float sumVal_sumSqBuf_16 = (myVal_sumSqBuf_16 + neighborVal_sumSqBuf_16);
    sumSqBuf[lid] = sumVal_sumSqBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSqBuf_8) {
    uint neighborIdx_sumSqBuf_8 = (lid + (uint)(8u));
    float myVal_sumSqBuf_8 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_8 = sumSqBuf[neighborIdx_sumSqBuf_8];
    float sumVal_sumSqBuf_8 = (myVal_sumSqBuf_8 + neighborVal_sumSqBuf_8);
    sumSqBuf[lid] = sumVal_sumSqBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSqBuf_4) {
    uint neighborIdx_sumSqBuf_4 = (lid + (uint)(4u));
    float myVal_sumSqBuf_4 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_4 = sumSqBuf[neighborIdx_sumSqBuf_4];
    float sumVal_sumSqBuf_4 = (myVal_sumSqBuf_4 + neighborVal_sumSqBuf_4);
    sumSqBuf[lid] = sumVal_sumSqBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSqBuf_2) {
    uint neighborIdx_sumSqBuf_2 = (lid + (uint)(2u));
    float myVal_sumSqBuf_2 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_2 = sumSqBuf[neighborIdx_sumSqBuf_2];
    float sumVal_sumSqBuf_2 = (myVal_sumSqBuf_2 + neighborVal_sumSqBuf_2);
    sumSqBuf[lid] = sumVal_sumSqBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSqBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSqBuf_1) {
    uint neighborIdx_sumSqBuf_1 = (lid + (uint)(1u));
    float myVal_sumSqBuf_1 = sumSqBuf[lid];
    float neighborVal_sumSqBuf_1 = sumSqBuf[neighborIdx_sumSqBuf_1];
    float sumVal_sumSqBuf_1 = (myVal_sumSqBuf_1 + neighborVal_sumSqBuf_1);
    sumSqBuf[lid] = sumVal_sumSqBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float totalSumSq = sumSqBuf[(uint)(0u)];
  float hsF32 = ((float)(hs));
  float meanSq = (totalSumSq / hsF32);
  float meanSqEps = (meanSq + eps);
  float rms = sqrt(meanSqEps);
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < hs))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      half gF16 = gamma[iterIdx];
      float g = ((float)(gF16));
      float xNorm = (xVal2 / rms);
      float resultF32 = (g * xNorm);
      half result = ((half)(resultF32));
      _output[globalIdx2] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_forward_f32 = R"kernel(
__kernel void navatala_transformer_softmax_forward_f32(__global const float* _input, __global const uint* batchSize, __global const uint* seqLen, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  float x = (((batchValid && seqValid)) ? (_input[globalIdx]) : (as_float(0xf149f2cau)));
  maxBuf[lid] = x;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxVal = maxBuf[(uint)(0u)];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if ((batchValid && seqValid)) {
    float result = (expVal / sumExp);
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_forward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_softmax_forward_f16(__global const half* _input, __global const uint* batchSize, __global const uint* seqLen, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  half xF16 = (((batchValid && seqValid)) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((seqValid) ? (((float)(xF16))) : (as_float(0xf149f2cau)));
  maxBuf[lid] = x;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxVal = maxBuf[(uint)(0u)];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = ((seqValid) ? (expVal) : (as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if ((batchValid && seqValid)) {
    float resultF32 = (expVal / sumExp);
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_backward_f32 = R"kernel(
__kernel void navatala_transformer_softmax_backward_f32(__global const float* _output, __global const float* gradOutput, __global const uint* batchSize, __global const uint* seqLen, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  float y = (((batchValid && seqValid)) ? (_output[globalIdx]) : (as_float(0x00000000u)));
  float dy = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : (as_float(0x00000000u)));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + (uint)(128u));
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + (uint)(64u));
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + (uint)(32u));
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + (uint)(16u));
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + (uint)(8u));
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + (uint)(4u));
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + (uint)(2u));
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + (uint)(1u));
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = dotBuf[(uint)(0u)];
  if ((batchValid && seqValid)) {
    float dyMinusDot = (dy - dot);
    float dx = (y * dyMinusDot);
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_softmax_backward_f16(__global const half* _output, __global const half* gradOutput, __global const uint* batchSize, __global const uint* seqLen, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float dotBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  uint globalIdx = ((batchIdx * sl) + lid);
  half yF16 = (((batchValid && seqValid)) ? (_output[globalIdx]) : ((half)(0.000000f)));
  half dyF16 = (((batchValid && seqValid)) ? (gradOutput[globalIdx]) : ((half)(0.000000f)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = (y * dy);
  dotBuf[lid] = yDy;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + (uint)(128u));
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + (uint)(64u));
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + (uint)(32u));
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + (uint)(16u));
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + (uint)(8u));
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + (uint)(4u));
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + (uint)(2u));
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + (uint)(1u));
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = dotBuf[(uint)(0u)];
  if ((batchValid && seqValid)) {
    float dyMinusDot = (dy - dot);
    float dxF32 = (y * dyMinusDot);
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_with_mask_f32 = R"kernel(
__kernel void navatala_transformer_softmax_with_mask_f32(__global const float* _input, __global const uint* batchSize, __global const uint* seqLen, __global const bool* isCausal, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  bool causal = isCausal[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  uint globalIdx = ((rowIdx * sl) + lid);
  float rawX = ((valid) ? (_input[globalIdx]) : (as_float(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (as_float(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxVal = maxBuf[(uint)(0u)];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float result = ((isMasked) ? (as_float(0x00000000u)) : ((expVal / sumExp)));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_with_mask_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_softmax_with_mask_f16(__global const half* _input, __global const uint* batchSize, __global const uint* seqLen, __global const bool* isCausal, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  bool causal = isCausal[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  uint globalIdx = ((rowIdx * sl) + lid);
  half rawXF16 = ((valid) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float rawX = ((colValid) ? (((float)(rawXF16))) : (as_float(0xf149f2cau)));
  bool isMasked = (causal && (col > row));
  float x = ((isMasked) ? (as_float(0xf149f2cau)) : (rawX));
  maxBuf[lid] = x;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxVal = maxBuf[(uint)(0u)];
  float xShifted = (x - maxVal);
  float expVal = exp(xShifted);
  float expValMasked = (((colValid && (!isMasked))) ? (expVal) : (as_float(0x00000000u)));
  sumBuf[lid] = expValMasked;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float resultF32 = ((isMasked) ? (as_float(0x00000000u)) : ((expVal / sumExp)));
    half result = ((half)(resultF32));
    _output[globalIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_with_mask_backward_f32 = R"kernel(
__kernel void navatala_transformer_softmax_with_mask_backward_f32(__global const float* _output, __global const float* gradOutput, __global const uint* batchSize, __global const uint* seqLen, __global const bool* isCausal, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  bool causal = isCausal[(uint)(0u)];
  __local float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  uint globalIdx = ((rowIdx * sl) + lid);
  float y = ((valid) ? (_output[globalIdx]) : (as_float(0x00000000u)));
  float dy = ((valid) ? (gradOutput[globalIdx]) : (as_float(0x00000000u)));
  float yDy = ((isMasked) ? (as_float(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + (uint)(128u));
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + (uint)(64u));
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + (uint)(32u));
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + (uint)(16u));
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + (uint)(8u));
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + (uint)(4u));
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + (uint)(2u));
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + (uint)(1u));
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = dotBuf[(uint)(0u)];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxUnmasked = (y * dyMinusDot);
    float dx = ((isMasked) ? (as_float(0x00000000u)) : (dxUnmasked));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_with_mask_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_softmax_with_mask_backward_f16(__global const half* _output, __global const half* gradOutput, __global const uint* batchSize, __global const uint* seqLen, __global const bool* isCausal, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint rowIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  bool causal = isCausal[(uint)(0u)];
  __local float dotBuf[256];
  bool rowValid = (rowIdx < bs);
  bool colValid = (lid < sl);
  bool valid = (rowValid && colValid);
  uint col = lid;
  uint row = (rowIdx % sl);
  bool isMasked = (causal && (col > row));
  uint globalIdx = ((rowIdx * sl) + lid);
  half yF16 = ((valid) ? (_output[globalIdx]) : ((half)(0.000000f)));
  half dyF16 = ((valid) ? (gradOutput[globalIdx]) : ((half)(0.000000f)));
  float y = ((float)(yF16));
  float dy = ((float)(dyF16));
  float yDy = ((isMasked) ? (as_float(0x00000000u)) : ((y * dy)));
  dotBuf[lid] = yDy;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_dotBuf_128) {
    uint neighborIdx_sum_dotBuf_128 = (lid + (uint)(128u));
    float myVal_sum_dotBuf_128 = dotBuf[lid];
    float neighborVal_sum_dotBuf_128 = dotBuf[neighborIdx_sum_dotBuf_128];
    float sumVal_sum_dotBuf_128 = (myVal_sum_dotBuf_128 + neighborVal_sum_dotBuf_128);
    dotBuf[lid] = sumVal_sum_dotBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_dotBuf_64) {
    uint neighborIdx_sum_dotBuf_64 = (lid + (uint)(64u));
    float myVal_sum_dotBuf_64 = dotBuf[lid];
    float neighborVal_sum_dotBuf_64 = dotBuf[neighborIdx_sum_dotBuf_64];
    float sumVal_sum_dotBuf_64 = (myVal_sum_dotBuf_64 + neighborVal_sum_dotBuf_64);
    dotBuf[lid] = sumVal_sum_dotBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_dotBuf_32) {
    uint neighborIdx_sum_dotBuf_32 = (lid + (uint)(32u));
    float myVal_sum_dotBuf_32 = dotBuf[lid];
    float neighborVal_sum_dotBuf_32 = dotBuf[neighborIdx_sum_dotBuf_32];
    float sumVal_sum_dotBuf_32 = (myVal_sum_dotBuf_32 + neighborVal_sum_dotBuf_32);
    dotBuf[lid] = sumVal_sum_dotBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_dotBuf_16) {
    uint neighborIdx_sum_dotBuf_16 = (lid + (uint)(16u));
    float myVal_sum_dotBuf_16 = dotBuf[lid];
    float neighborVal_sum_dotBuf_16 = dotBuf[neighborIdx_sum_dotBuf_16];
    float sumVal_sum_dotBuf_16 = (myVal_sum_dotBuf_16 + neighborVal_sum_dotBuf_16);
    dotBuf[lid] = sumVal_sum_dotBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_dotBuf_8) {
    uint neighborIdx_sum_dotBuf_8 = (lid + (uint)(8u));
    float myVal_sum_dotBuf_8 = dotBuf[lid];
    float neighborVal_sum_dotBuf_8 = dotBuf[neighborIdx_sum_dotBuf_8];
    float sumVal_sum_dotBuf_8 = (myVal_sum_dotBuf_8 + neighborVal_sum_dotBuf_8);
    dotBuf[lid] = sumVal_sum_dotBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_dotBuf_4) {
    uint neighborIdx_sum_dotBuf_4 = (lid + (uint)(4u));
    float myVal_sum_dotBuf_4 = dotBuf[lid];
    float neighborVal_sum_dotBuf_4 = dotBuf[neighborIdx_sum_dotBuf_4];
    float sumVal_sum_dotBuf_4 = (myVal_sum_dotBuf_4 + neighborVal_sum_dotBuf_4);
    dotBuf[lid] = sumVal_sum_dotBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_dotBuf_2) {
    uint neighborIdx_sum_dotBuf_2 = (lid + (uint)(2u));
    float myVal_sum_dotBuf_2 = dotBuf[lid];
    float neighborVal_sum_dotBuf_2 = dotBuf[neighborIdx_sum_dotBuf_2];
    float sumVal_sum_dotBuf_2 = (myVal_sum_dotBuf_2 + neighborVal_sum_dotBuf_2);
    dotBuf[lid] = sumVal_sum_dotBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_dotBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_dotBuf_1) {
    uint neighborIdx_sum_dotBuf_1 = (lid + (uint)(1u));
    float myVal_sum_dotBuf_1 = dotBuf[lid];
    float neighborVal_sum_dotBuf_1 = dotBuf[neighborIdx_sum_dotBuf_1];
    float sumVal_sum_dotBuf_1 = (myVal_sum_dotBuf_1 + neighborVal_sum_dotBuf_1);
    dotBuf[lid] = sumVal_sum_dotBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = dotBuf[(uint)(0u)];
  if (valid) {
    float dyMinusDot = (dy - dot);
    float dxF32Unmasked = (y * dyMinusDot);
    float dxF32 = ((isMasked) ? (as_float(0x00000000u)) : (dxF32Unmasked));
    half dx = ((half)(dxF32));
    gradInput[globalIdx] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_softmax_multi_pass_f32 = R"kernel(
__kernel void navatala_transformer_softmax_multi_pass_f32(__global const float* _input, __global const uint* batchSize, __global const uint* seqLen, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * sl);
  uint workgroupSize = (uint)(256u);
  float partialMax = as_float(0xf149f2cau);
  float partialSum = as_float(0x00000000u);
  uint iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    uint globalIdx = (baseIdx + iterIdx);
    float xVal = ((batchValid) ? (_input[globalIdx]) : (as_float(0xf149f2cau)));
    bool isGreater = (xVal > partialMax);
    partialMax = ((isGreater) ? (xVal) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float globalMax = maxBuf[(uint)(0u)];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      float xVal2 = _input[globalIdx2];
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float globalSum = sumBuf[(uint)(0u)];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx3 = (baseIdx + iterIdx);
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
const char* k_opencl_navatala_transformer_softmax_multi_pass_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_softmax_multi_pass_f16(__global const half* _input, __global const uint* batchSize, __global const uint* seqLen, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  __local float maxBuf[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  uint baseIdx = (batchIdx * sl);
  uint workgroupSize = (uint)(256u);
  float partialMax = as_float(0xf149f2cau);
  float partialSum = as_float(0x00000000u);
  uint iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    uint globalIdx = (baseIdx + iterIdx);
    half xF16 = ((batchValid) ? (_input[globalIdx]) : ((half)(0.000000f)));
    float xVal = ((float)(xF16));
    float xValSafe = ((batchValid) ? (xVal) : (as_float(0xf149f2cau)));
    bool isGreater = (xValSafe > partialMax);
    partialMax = ((isGreater) ? (xValSafe) : (partialMax));
    iterIdx = (iterIdx + workgroupSize);
  }
  maxBuf[lid] = partialMax;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_maxBuf_128) {
    uint neighborIdx_max_maxBuf_128 = (lid + (uint)(128u));
    float myVal_max_maxBuf_128 = maxBuf[lid];
    float neighborVal_max_maxBuf_128 = maxBuf[neighborIdx_max_maxBuf_128];
    bool isGreater_max_maxBuf_128 = (neighborVal_max_maxBuf_128 > myVal_max_maxBuf_128);
    float maxVal_max_maxBuf_128 = ((isGreater_max_maxBuf_128) ? (neighborVal_max_maxBuf_128) : (myVal_max_maxBuf_128));
    maxBuf[lid] = maxVal_max_maxBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_maxBuf_64) {
    uint neighborIdx_max_maxBuf_64 = (lid + (uint)(64u));
    float myVal_max_maxBuf_64 = maxBuf[lid];
    float neighborVal_max_maxBuf_64 = maxBuf[neighborIdx_max_maxBuf_64];
    bool isGreater_max_maxBuf_64 = (neighborVal_max_maxBuf_64 > myVal_max_maxBuf_64);
    float maxVal_max_maxBuf_64 = ((isGreater_max_maxBuf_64) ? (neighborVal_max_maxBuf_64) : (myVal_max_maxBuf_64));
    maxBuf[lid] = maxVal_max_maxBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_maxBuf_32) {
    uint neighborIdx_max_maxBuf_32 = (lid + (uint)(32u));
    float myVal_max_maxBuf_32 = maxBuf[lid];
    float neighborVal_max_maxBuf_32 = maxBuf[neighborIdx_max_maxBuf_32];
    bool isGreater_max_maxBuf_32 = (neighborVal_max_maxBuf_32 > myVal_max_maxBuf_32);
    float maxVal_max_maxBuf_32 = ((isGreater_max_maxBuf_32) ? (neighborVal_max_maxBuf_32) : (myVal_max_maxBuf_32));
    maxBuf[lid] = maxVal_max_maxBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_maxBuf_16) {
    uint neighborIdx_max_maxBuf_16 = (lid + (uint)(16u));
    float myVal_max_maxBuf_16 = maxBuf[lid];
    float neighborVal_max_maxBuf_16 = maxBuf[neighborIdx_max_maxBuf_16];
    bool isGreater_max_maxBuf_16 = (neighborVal_max_maxBuf_16 > myVal_max_maxBuf_16);
    float maxVal_max_maxBuf_16 = ((isGreater_max_maxBuf_16) ? (neighborVal_max_maxBuf_16) : (myVal_max_maxBuf_16));
    maxBuf[lid] = maxVal_max_maxBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_maxBuf_8) {
    uint neighborIdx_max_maxBuf_8 = (lid + (uint)(8u));
    float myVal_max_maxBuf_8 = maxBuf[lid];
    float neighborVal_max_maxBuf_8 = maxBuf[neighborIdx_max_maxBuf_8];
    bool isGreater_max_maxBuf_8 = (neighborVal_max_maxBuf_8 > myVal_max_maxBuf_8);
    float maxVal_max_maxBuf_8 = ((isGreater_max_maxBuf_8) ? (neighborVal_max_maxBuf_8) : (myVal_max_maxBuf_8));
    maxBuf[lid] = maxVal_max_maxBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_maxBuf_4) {
    uint neighborIdx_max_maxBuf_4 = (lid + (uint)(4u));
    float myVal_max_maxBuf_4 = maxBuf[lid];
    float neighborVal_max_maxBuf_4 = maxBuf[neighborIdx_max_maxBuf_4];
    bool isGreater_max_maxBuf_4 = (neighborVal_max_maxBuf_4 > myVal_max_maxBuf_4);
    float maxVal_max_maxBuf_4 = ((isGreater_max_maxBuf_4) ? (neighborVal_max_maxBuf_4) : (myVal_max_maxBuf_4));
    maxBuf[lid] = maxVal_max_maxBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_maxBuf_2) {
    uint neighborIdx_max_maxBuf_2 = (lid + (uint)(2u));
    float myVal_max_maxBuf_2 = maxBuf[lid];
    float neighborVal_max_maxBuf_2 = maxBuf[neighborIdx_max_maxBuf_2];
    bool isGreater_max_maxBuf_2 = (neighborVal_max_maxBuf_2 > myVal_max_maxBuf_2);
    float maxVal_max_maxBuf_2 = ((isGreater_max_maxBuf_2) ? (neighborVal_max_maxBuf_2) : (myVal_max_maxBuf_2));
    maxBuf[lid] = maxVal_max_maxBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_maxBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_maxBuf_1) {
    uint neighborIdx_max_maxBuf_1 = (lid + (uint)(1u));
    float myVal_max_maxBuf_1 = maxBuf[lid];
    float neighborVal_max_maxBuf_1 = maxBuf[neighborIdx_max_maxBuf_1];
    bool isGreater_max_maxBuf_1 = (neighborVal_max_maxBuf_1 > myVal_max_maxBuf_1);
    float maxVal_max_maxBuf_1 = ((isGreater_max_maxBuf_1) ? (neighborVal_max_maxBuf_1) : (myVal_max_maxBuf_1));
    maxBuf[lid] = maxVal_max_maxBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float globalMax = maxBuf[(uint)(0u)];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx2 = (baseIdx + iterIdx);
      half xF16_2 = _input[globalIdx2];
      float xVal2 = ((float)(xF16_2));
      float xShifted = (xVal2 - globalMax);
      float expVal = exp(xShifted);
      partialSum = (partialSum + expVal);
    }
    iterIdx = (iterIdx + workgroupSize);
  }
  sumBuf[lid] = partialSum;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_sumBuf_128) {
    uint neighborIdx_sum_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_sumBuf_128 = sumBuf[neighborIdx_sum_sumBuf_128];
    float sumVal_sum_sumBuf_128 = (myVal_sum_sumBuf_128 + neighborVal_sum_sumBuf_128);
    sumBuf[lid] = sumVal_sum_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_sumBuf_64) {
    uint neighborIdx_sum_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_sumBuf_64 = sumBuf[neighborIdx_sum_sumBuf_64];
    float sumVal_sum_sumBuf_64 = (myVal_sum_sumBuf_64 + neighborVal_sum_sumBuf_64);
    sumBuf[lid] = sumVal_sum_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_sumBuf_32) {
    uint neighborIdx_sum_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_sumBuf_32 = sumBuf[neighborIdx_sum_sumBuf_32];
    float sumVal_sum_sumBuf_32 = (myVal_sum_sumBuf_32 + neighborVal_sum_sumBuf_32);
    sumBuf[lid] = sumVal_sum_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_sumBuf_16) {
    uint neighborIdx_sum_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_sumBuf_16 = sumBuf[neighborIdx_sum_sumBuf_16];
    float sumVal_sum_sumBuf_16 = (myVal_sum_sumBuf_16 + neighborVal_sum_sumBuf_16);
    sumBuf[lid] = sumVal_sum_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_sumBuf_8) {
    uint neighborIdx_sum_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_sumBuf_8 = sumBuf[neighborIdx_sum_sumBuf_8];
    float sumVal_sum_sumBuf_8 = (myVal_sum_sumBuf_8 + neighborVal_sum_sumBuf_8);
    sumBuf[lid] = sumVal_sum_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_sumBuf_4) {
    uint neighborIdx_sum_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_sumBuf_4 = sumBuf[neighborIdx_sum_sumBuf_4];
    float sumVal_sum_sumBuf_4 = (myVal_sum_sumBuf_4 + neighborVal_sum_sumBuf_4);
    sumBuf[lid] = sumVal_sum_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_sumBuf_2) {
    uint neighborIdx_sum_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_sumBuf_2 = sumBuf[neighborIdx_sum_sumBuf_2];
    float sumVal_sum_sumBuf_2 = (myVal_sum_sumBuf_2 + neighborVal_sum_sumBuf_2);
    sumBuf[lid] = sumVal_sum_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_sumBuf_1) {
    uint neighborIdx_sum_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_sumBuf_1 = sumBuf[neighborIdx_sum_sumBuf_1];
    float sumVal_sum_sumBuf_1 = (myVal_sum_sumBuf_1 + neighborVal_sum_sumBuf_1);
    sumBuf[lid] = sumVal_sum_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float globalSum = sumBuf[(uint)(0u)];
  iterIdx = lid;
  for (int __iter = 0; __iter < 16384; ++__iter) {
    if (!((iterIdx < sl))) break;
    if (batchValid) {
      uint globalIdx3 = (baseIdx + iterIdx);
      half xF16_3 = _input[globalIdx3];
      float xVal3 = ((float)(xF16_3));
      float xShifted3 = (xVal3 - globalMax);
      float expVal3 = exp(xShifted3);
      float resultF32 = (expVal3 / globalSum);
      half result = ((half)(resultF32));
      _output[globalIdx3] = result;
    }
    iterIdx = (iterIdx + workgroupSize);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_cast_f32_to_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_cast_f32_to_f16(__global const float* _input, __global const uint* count, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    half result = ((half)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_cast_f16_to_f32 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_cast_f16_to_f32(__global const half* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    float result = ((float)(x));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_transpose_f32 = R"kernel(
__kernel void navatala_transformer_transpose_f32(__global const float* _input, __global const uint* rows, __global const uint* cols, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint r = rows[(uint)(0u)];
  uint c = cols[(uint)(0u)];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float x = _input[gid];
    uint outIdx = ((col * r) + row);
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_transpose_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_transpose_f16(__global const half* _input, __global const uint* rows, __global const uint* cols, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint r = rows[(uint)(0u)];
  uint c = cols[(uint)(0u)];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    half x = _input[gid];
    uint outIdx = ((col * r) + row);
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_fused_cast_transpose_f32_to_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_fused_cast_transpose_f32_to_f16(__global const float* _input, __global const uint* rows, __global const uint* cols, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint r = rows[(uint)(0u)];
  uint c = cols[(uint)(0u)];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float x = _input[gid];
    half xF16 = ((half)(x));
    uint outIdx = ((col * r) + row);
    _output[outIdx] = xF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_cast_f32_to_f_p8_e4_m3 = R"kernel(
__kernel void navatala_transformer_cast_f32_to_f_p8_e4_m3(__global const float* _input, __global const float* scale, __global const uint* count, __global uchar* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[(uint)(0u)];
    float scaled = (x * s);
    float minVal = as_float(0xc3700000u);
    float maxVal = as_float(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    _output[gid] = fp8Val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_cast_f_p8_e4_m3_to_f32 = R"kernel(
__kernel void navatala_transformer_cast_f_p8_e4_m3_to_f32(__global const uchar* _input, __global const float* scale, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uchar fp8Val = _input[gid];
    float s = scale[(uint)(0u)];
    float f32Val = ((float)(fp8Val));
    float result = (f32Val / s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_cast_f32_to_f_p8_e5_m2 = R"kernel(
__kernel void navatala_transformer_cast_f32_to_f_p8_e5_m2(__global const float* _input, __global const float* scale, __global const uint* count, __global uchar* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float s = scale[(uint)(0u)];
    float scaled = (x * s);
    float minVal = as_float(0xc7600000u);
    float maxVal = as_float(0x47600000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    _output[gid] = fp8Val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_cast_f_p8_e5_m2_to_f32 = R"kernel(
__kernel void navatala_transformer_cast_f_p8_e5_m2_to_f32(__global const uchar* _input, __global const float* scale, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uchar fp8Val = _input[gid];
    float s = scale[(uint)(0u)];
    float f32Val = ((float)(fp8Val));
    float result = (f32Val / s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_fused_cast_transpose_f_p8 = R"kernel(
__kernel void navatala_transformer_fused_cast_transpose_f_p8(__global const float* _input, __global const float* scale, __global const uint* rows, __global const uint* cols, __global uchar* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint r = rows[(uint)(0u)];
  uint c = cols[(uint)(0u)];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float x = _input[gid];
    float s = scale[(uint)(0u)];
    float scaled = (x * s);
    float minVal = as_float(0xc3700000u);
    float maxVal = as_float(0x43700000u);
    float clamped = ((((((scaled < maxVal)) ? (scaled) : (maxVal)) > minVal)) ? ((((scaled < maxVal)) ? (scaled) : (maxVal))) : (minVal));
    float rounded = round(clamped);
    uchar fp8Val = ((uchar)(rounded));
    uint outIdx = ((col * r) + row);
    _output[outIdx] = fp8Val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_transpose4_d_f32 = R"kernel(
__kernel void navatala_transformer_transpose4_d_f32(__global const float* _input, __global const uint* dim0, __global const uint* dim1, __global const uint* dim2, __global const uint* dim3, __global const uint* perm0, __global const uint* perm1, __global const uint* perm2, __global const uint* perm3, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d0 = dim0[(uint)(0u)];
  uint d1 = dim1[(uint)(0u)];
  uint d2 = dim2[(uint)(0u)];
  uint d3 = dim3[(uint)(0u)];
  uint p0 = perm0[(uint)(0u)];
  uint p1 = perm1[(uint)(0u)];
  uint p2 = perm2[(uint)(0u)];
  uint p3 = perm3[(uint)(0u)];
  uint total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride1 = ((d1 * d2) * d3);
    uint stride2 = (d2 * d3);
    uint stride3 = d3;
    uint i0 = (gid / stride1);
    uint rem0 = (gid % stride1);
    uint i1 = (rem0 / stride2);
    uint rem1 = (rem0 % stride2);
    uint i2 = (rem1 / stride3);
    uint i3 = (rem1 % stride3);
    uint outD0 = (((p0 == (uint)(0u))) ? (d0) : ((((p0 == (uint)(1u))) ? (d1) : ((((p0 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD1 = (((p1 == (uint)(0u))) ? (d0) : ((((p1 == (uint)(1u))) ? (d1) : ((((p1 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD2 = (((p2 == (uint)(0u))) ? (d0) : ((((p2 == (uint)(1u))) ? (d1) : ((((p2 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD3 = (((p3 == (uint)(0u))) ? (d0) : ((((p3 == (uint)(1u))) ? (d1) : ((((p3 == (uint)(2u))) ? (d2) : (d3))))));
    uint outI0 = (((p0 == (uint)(0u))) ? (i0) : ((((p0 == (uint)(1u))) ? (i1) : ((((p0 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI1 = (((p1 == (uint)(0u))) ? (i0) : ((((p1 == (uint)(1u))) ? (i1) : ((((p1 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI2 = (((p2 == (uint)(0u))) ? (i0) : ((((p2 == (uint)(1u))) ? (i1) : ((((p2 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI3 = (((p3 == (uint)(0u))) ? (i0) : ((((p3 == (uint)(1u))) ? (i1) : ((((p3 == (uint)(2u))) ? (i2) : (i3))))));
    uint outStride1 = ((outD1 * outD2) * outD3);
    uint outStride2 = (outD2 * outD3);
    uint outStride3 = outD3;
    uint outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    float x = _input[gid];
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_transpose4_d_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_transpose4_d_f16(__global const half* _input, __global const uint* dim0, __global const uint* dim1, __global const uint* dim2, __global const uint* dim3, __global const uint* perm0, __global const uint* perm1, __global const uint* perm2, __global const uint* perm3, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d0 = dim0[(uint)(0u)];
  uint d1 = dim1[(uint)(0u)];
  uint d2 = dim2[(uint)(0u)];
  uint d3 = dim3[(uint)(0u)];
  uint p0 = perm0[(uint)(0u)];
  uint p1 = perm1[(uint)(0u)];
  uint p2 = perm2[(uint)(0u)];
  uint p3 = perm3[(uint)(0u)];
  uint total = (((d0 * d1) * d2) * d3);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride1 = ((d1 * d2) * d3);
    uint stride2 = (d2 * d3);
    uint stride3 = d3;
    uint i0 = (gid / stride1);
    uint rem0 = (gid % stride1);
    uint i1 = (rem0 / stride2);
    uint rem1 = (rem0 % stride2);
    uint i2 = (rem1 / stride3);
    uint i3 = (rem1 % stride3);
    uint outD0 = (((p0 == (uint)(0u))) ? (d0) : ((((p0 == (uint)(1u))) ? (d1) : ((((p0 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD1 = (((p1 == (uint)(0u))) ? (d0) : ((((p1 == (uint)(1u))) ? (d1) : ((((p1 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD2 = (((p2 == (uint)(0u))) ? (d0) : ((((p2 == (uint)(1u))) ? (d1) : ((((p2 == (uint)(2u))) ? (d2) : (d3))))));
    uint outD3 = (((p3 == (uint)(0u))) ? (d0) : ((((p3 == (uint)(1u))) ? (d1) : ((((p3 == (uint)(2u))) ? (d2) : (d3))))));
    uint outI0 = (((p0 == (uint)(0u))) ? (i0) : ((((p0 == (uint)(1u))) ? (i1) : ((((p0 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI1 = (((p1 == (uint)(0u))) ? (i0) : ((((p1 == (uint)(1u))) ? (i1) : ((((p1 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI2 = (((p2 == (uint)(0u))) ? (i0) : ((((p2 == (uint)(1u))) ? (i1) : ((((p2 == (uint)(2u))) ? (i2) : (i3))))));
    uint outI3 = (((p3 == (uint)(0u))) ? (i0) : ((((p3 == (uint)(1u))) ? (i1) : ((((p3 == (uint)(2u))) ? (i2) : (i3))))));
    uint outStride1 = ((outD1 * outD2) * outD3);
    uint outStride2 = (outD2 * outD3);
    uint outStride3 = outD3;
    uint outIdx = ((((outI0 * outStride1) + (outI1 * outStride2)) + (outI2 * outStride3)) + outI3);
    half x = _input[gid];
    _output[outIdx] = x;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_dropout_forward_f32 = R"kernel(
__kernel void navatala_transformer_dropout_forward_f32(__global const float* _input, __global const uint* count, __global const float* prob, __global const uint* seed, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  float p = prob[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    uint h0 = (gid ^ s);
    uint h1 = (h0 * (uint)(3432918353u));
    uint h2 = (h1 ^ (h1 >> (uint)(15u)));
    uint h3 = (h2 * (uint)(461845907u));
    uint hashVal = (h3 ^ (h3 >> (uint)(13u)));
    float randF32 = (((float)(hashVal)) / as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_float(0x3f800000u) - p);
    float scale = (as_float(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float result = ((keep) ? (scaled) : (as_float(0x00000000u)));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_dropout_forward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_dropout_forward_f16(__global const half* _input, __global const uint* count, __global const float* prob, __global const uint* seed, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  float p = prob[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half xF16 = _input[gid];
    float x = ((float)(xF16));
    uint h0 = (gid ^ s);
    uint h1 = (h0 * (uint)(3432918353u));
    uint h2 = (h1 ^ (h1 >> (uint)(15u)));
    uint h3 = (h2 * (uint)(461845907u));
    uint hashVal = (h3 ^ (h3 >> (uint)(13u)));
    float randF32 = (((float)(hashVal)) / as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_float(0x3f800000u) - p);
    float scale = (as_float(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float resultF32 = ((keep) ? (scaled) : (as_float(0x00000000u)));
    half result = ((half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_dropout_backward_f32 = R"kernel(
__kernel void navatala_transformer_dropout_backward_f32(__global const float* gradOutput, __global const uint* count, __global const float* prob, __global const uint* seed, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  float p = prob[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float dy = gradOutput[gid];
    uint h0 = (gid ^ s);
    uint h1 = (h0 * (uint)(3432918353u));
    uint h2 = (h1 ^ (h1 >> (uint)(15u)));
    uint h3 = (h2 * (uint)(461845907u));
    uint hashVal = (h3 ^ (h3 >> (uint)(13u)));
    float randF32 = (((float)(hashVal)) / as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_float(0x3f800000u) - p);
    float scale = (as_float(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dx = ((keep) ? (scaled) : (as_float(0x00000000u)));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_dropout_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_dropout_backward_f16(__global const half* gradOutput, __global const uint* count, __global const float* prob, __global const uint* seed, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  float p = prob[(uint)(0u)];
  uint s = seed[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    half dyF16 = gradOutput[gid];
    float dy = ((float)(dyF16));
    uint h0 = (gid ^ s);
    uint h1 = (h0 * (uint)(3432918353u));
    uint h2 = (h1 ^ (h1 >> (uint)(15u)));
    uint h3 = (h2 * (uint)(461845907u));
    uint hashVal = (h3 ^ (h3 >> (uint)(13u)));
    float randF32 = (((float)(hashVal)) / as_float(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_float(0x3f800000u) - p);
    float scale = (as_float(0x3f800000u) / oneMinusP);
    float scaled = (dy * scale);
    float dxF32 = ((keep) ? (scaled) : (as_float(0x00000000u)));
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rope_forward_f32 = R"kernel(
__kernel void navatala_transformer_rope_forward_f32(__global const float* _input, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
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
const char* k_opencl_navatala_transformer_rope_forward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rope_forward_f16(__global const half* _input, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    half x0F16 = _input[idx0];
    half x1F16 = _input[idx1];
    float x0 = ((float)(x0F16));
    float x1 = ((float)(x1F16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    half y0 = ((half)(y0F32));
    half y1 = ((half)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rope_with_cache_f32 = R"kernel(
__kernel void navatala_transformer_rope_with_cache_f32(__global const float* _input, __global const float* cosCache, __global const float* sinCache, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    uint cacheIdx = ((pos * halfHd) + dimPair);
    float cosTheta = cosCache[cacheIdx];
    float sinTheta = sinCache[cacheIdx];
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float x0 = _input[idx0];
    float x1 = _input[idx1];
    float y0 = ((x0 * cosTheta) - (x1 * sinTheta));
    float y1 = ((x0 * sinTheta) + (x1 * cosTheta));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rope_backward_f32 = R"kernel(
__kernel void navatala_transformer_rope_backward_f32(__global const float* gradOutput, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global float* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    float dy0 = gradOutput[idx0];
    float dy1 = gradOutput[idx1];
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (as_float(0x00000000u) - dy0Sin);
    float dx1 = (negDy0Sin + dy1Cos);
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rope_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rope_backward_f16(__global const half* gradOutput, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global half* gradInput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimPair = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint dim0 = (dimPair * (uint)(2u));
    uint dim1 = (dim0 + (uint)(1u));
    uint idx0 = (baseOffset + (seqOffset + dim0));
    uint idx1 = (baseOffset + (seqOffset + dim1));
    half dy0F16 = gradOutput[idx0];
    half dy1F16 = gradOutput[idx1];
    float dy0 = ((float)(dy0F16));
    float dy1 = ((float)(dy1F16));
    float dy0Cos = (dy0 * cosTheta);
    float dy1Sin = (dy1 * sinTheta);
    float dx0F32 = (dy0Cos + dy1Sin);
    float dy0Sin = (dy0 * sinTheta);
    float dy1Cos = (dy1 * cosTheta);
    float negDy0Sin = (as_float(0x00000000u) - dy0Sin);
    float dx1F32 = (negDy0Sin + dy1Cos);
    half dx0 = ((half)(dx0F32));
    half dx1 = ((half)(dx1F32));
    gradInput[idx0] = dx0;
    gradInput[idx1] = dx1;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_rope_interleaved_f32 = R"kernel(
__kernel void navatala_transformer_rope_interleaved_f32(__global const float* _input, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimIdx = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint idx0 = (baseOffset + (seqOffset + dimIdx));
    uint idx1 = (idx0 + halfHd);
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
const char* k_opencl_navatala_transformer_rope_interleaved_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_rope_interleaved_f16(__global const half* _input, __global const uint* positions, __global const uint* batchSize, __global const uint* seqLen, __global const uint* headDim, __global const float* base, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint totalPairs = (bs * (sl * halfHd));
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    uint batchSeqPairs = (sl * halfHd);
    uint batchIdx = (gid / batchSeqPairs);
    uint remainder = (gid % batchSeqPairs);
    uint seqIdx = (remainder / halfHd);
    uint dimIdx = (remainder % halfHd);
    uint posIdx = ((batchIdx * sl) + seqIdx);
    uint pos = positions[posIdx];
    float posF32 = ((float)(pos));
    float dimIdxF32 = ((float)(dimIdx));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimIdxF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint baseOffset = (batchIdx * (sl * hd));
    uint seqOffset = (seqIdx * hd);
    uint idx0 = (baseOffset + (seqOffset + dimIdx));
    uint idx1 = (idx0 + halfHd);
    half x0F16 = _input[idx0];
    half x1F16 = _input[idx1];
    float x0 = ((float)(x0F16));
    float x1 = ((float)(x1F16));
    float x0Cos = (x0 * cosTheta);
    float x1Sin = (x1 * sinTheta);
    float y0F32 = (x0Cos - x1Sin);
    float x0Sin = (x0 * sinTheta);
    float x1Cos = (x1 * cosTheta);
    float y1F32 = (x0Sin + x1Cos);
    half y0 = ((half)(y0F32));
    half y1 = ((half)(y1F32));
    _output[idx0] = y0;
    _output[idx1] = y1;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_simple_attention_f16(__global const half* query, __global const half* key, __global const half* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    half outF16 = ((half)(outVal));
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_f32 = R"kernel(
__kernel void navatala_transformer_simple_attention_f32(__global const float* query, __global const float* key, __global const float* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_paged_attention_f32 = R"kernel(
__kernel void navatala_transformer_paged_attention_f32(__global const float* query, __global const float* kvCache, __global const uint* blockTables, __global const uint* contextLens, __global const uint* numHeads, __global const uint* headDim, __global const uint* pageSize, __global const uint* maxBlocks, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint ps = pageSize[(uint)(0u)];
  uint mb = maxBlocks[(uint)(0u)];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint ctxLen = contextLens[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool kvValid = (lid < ctxLen);
  uint qBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  uint blockIdx = (lid / ps);
  uint blockOffset = (lid % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = ((uint)(2u) * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  if (kvValid) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if (kvValid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  for (int outD = 0; outD < (int)(hd); ++outD) {
    if (kvValid) {
      float myExpScore = attnScores[lid];
      float shiftedExp = (myExpScore - maxScore);
      float expVal = exp(shiftedExp);
      float attnWeight = (expVal / sumExp);
      uint vIdx = (vBase + outD);
      float vVal = kvCache[vIdx];
      float weighted = (attnWeight * vVal);
      sumBuf[lid] = weighted;
    } else {
      sumBuf[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
    if (shouldReduce_sum_p2_sumBuf_128) {
      uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
      float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
      float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
    if (shouldReduce_sum_p2_sumBuf_64) {
      uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
      float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
      float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
    if (shouldReduce_sum_p2_sumBuf_32) {
      uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
      float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
      float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
    if (shouldReduce_sum_p2_sumBuf_16) {
      uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
      float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
      float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
    if (shouldReduce_sum_p2_sumBuf_8) {
      uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
      float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
      float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
    if (shouldReduce_sum_p2_sumBuf_4) {
      uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
      float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
      float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
    if (shouldReduce_sum_p2_sumBuf_2) {
      uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
      float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
      float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
    if (shouldReduce_sum_p2_sumBuf_1) {
      uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
      float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
      float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
      float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
      sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lid == (uint)(0u))) {
      float outVal = sumBuf[(uint)(0u)];
      uint outIdx = (qBase + outD);
      _output[outIdx] = outVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_with_mask_f32 = R"kernel(
__kernel void navatala_transformer_simple_attention_with_mask_f32(__global const float* query, __global const float* key, __global const float* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool causalMasked = (lid > queryPos);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!causalMasked))) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if ((valid && (!causalMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if ((valid && (!causalMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_with_mask_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_simple_attention_with_mask_f16(__global const half* query, __global const half* key, __global const half* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool causalMasked = (lid > queryPos);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!causalMasked))) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if ((valid && (!causalMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if ((valid && (!causalMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    half outF16 = ((half)(outVal));
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_with_padding_f32 = R"kernel(
__kernel void navatala_transformer_simple_attention_with_padding_f32(__global const float* query, __global const float* key, __global const float* value, __global const uint* seqLengths, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* numHeads, __global const uint* headDim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = maxSeqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  uint actualSeqLen = seqLengths[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool paddingMasked = (lid >= actualSeqLen);
  bool queryMasked = (queryPos >= actualSeqLen);
  bool anyMasked = (paddingMasked || queryMasked);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!anyMasked))) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if ((valid && (!anyMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if ((valid && (!anyMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float v0 = value[vBase];
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outVal;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_with_padding_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_simple_attention_with_padding_f16(__global const half* query, __global const half* key, __global const half* value, __global const uint* seqLengths, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* numHeads, __global const uint* headDim, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = maxSeqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  uint actualSeqLen = seqLengths[batchIdx];
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  bool paddingMasked = (lid >= actualSeqLen);
  bool queryMasked = (queryPos >= actualSeqLen);
  bool anyMasked = (paddingMasked || queryMasked);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if ((valid && (!anyMasked))) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if ((valid && (!anyMasked))) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if ((valid && (!anyMasked))) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    half v0F16 = value[vBase];
    float v0 = ((float)(v0F16));
    float weighted = (attnWeight * v0);
    sumBuf[lid] = weighted;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p2_sumBuf_128) {
    uint neighborIdx_sum_p2_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p2_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_128 = sumBuf[neighborIdx_sum_p2_sumBuf_128];
    float sumVal_sum_p2_sumBuf_128 = (myVal_sum_p2_sumBuf_128 + neighborVal_sum_p2_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p2_sumBuf_64) {
    uint neighborIdx_sum_p2_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p2_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_64 = sumBuf[neighborIdx_sum_p2_sumBuf_64];
    float sumVal_sum_p2_sumBuf_64 = (myVal_sum_p2_sumBuf_64 + neighborVal_sum_p2_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p2_sumBuf_32) {
    uint neighborIdx_sum_p2_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p2_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_32 = sumBuf[neighborIdx_sum_p2_sumBuf_32];
    float sumVal_sum_p2_sumBuf_32 = (myVal_sum_p2_sumBuf_32 + neighborVal_sum_p2_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p2_sumBuf_16) {
    uint neighborIdx_sum_p2_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p2_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_16 = sumBuf[neighborIdx_sum_p2_sumBuf_16];
    float sumVal_sum_p2_sumBuf_16 = (myVal_sum_p2_sumBuf_16 + neighborVal_sum_p2_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p2_sumBuf_8) {
    uint neighborIdx_sum_p2_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p2_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_8 = sumBuf[neighborIdx_sum_p2_sumBuf_8];
    float sumVal_sum_p2_sumBuf_8 = (myVal_sum_p2_sumBuf_8 + neighborVal_sum_p2_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p2_sumBuf_4) {
    uint neighborIdx_sum_p2_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p2_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_4 = sumBuf[neighborIdx_sum_p2_sumBuf_4];
    float sumVal_sum_p2_sumBuf_4 = (myVal_sum_p2_sumBuf_4 + neighborVal_sum_p2_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p2_sumBuf_2) {
    uint neighborIdx_sum_p2_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p2_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_2 = sumBuf[neighborIdx_sum_p2_sumBuf_2];
    float sumVal_sum_p2_sumBuf_2 = (myVal_sum_p2_sumBuf_2 + neighborVal_sum_p2_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p2_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p2_sumBuf_1) {
    uint neighborIdx_sum_p2_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p2_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p2_sumBuf_1 = sumBuf[neighborIdx_sum_p2_sumBuf_1];
    float sumVal_sum_p2_sumBuf_1 = (myVal_sum_p2_sumBuf_1 + neighborVal_sum_p2_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p2_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((batchValid && (lid == (uint)(0u)))) {
    float outVal = sumBuf[(uint)(0u)];
    half outF16 = ((half)(outVal));
    uint outIdx = (qBase + (uint)(0u));
    _output[outIdx] = outF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_simple_attention_backward_f32 = R"kernel(
__kernel void navatala_transformer_simple_attention_backward_f32(__global const float* gradOutput, __global const float* query, __global const float* key, __global const float* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global float* gradQuery, __global float* gradKey, __global float* gradValue) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  __local float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProdQK = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float dotProdGradAttn = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint gradOutIdx = (qBase + ((uint)(d)));
      uint vIdx = (vBase + ((uint)(d)));
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
    gradBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
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
const char* k_opencl_navatala_transformer_simple_attention_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_simple_attention_backward_f16(__global const half* gradOutput, __global const half* query, __global const half* key, __global const half* value, __global const uint* batchSize, __global const uint* seqLen, __global const uint* numHeads, __global const uint* headDim, __global half* gradQuery, __global half* gradKey, __global half* gradValue) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint seqHeads = (sl * nh);
  uint batchIdx = (groupId / seqHeads);
  uint remainder = (groupId % seqHeads);
  uint queryPos = (remainder / nh);
  uint headIdx = (remainder % nh);
  float hdF32 = ((float)(hd));
  float sqrtHd = sqrt(hdF32);
  float scale = (as_float(0x3f800000u) / sqrtHd);
  __local float attnScores[256];
  __local float sumBuf[256];
  __local float gradBuf[256];
  bool batchValid = (batchIdx < bs);
  bool seqValid = (lid < sl);
  bool valid = (batchValid && seqValid);
  uint headDimStride = hd;
  uint headStride = (nh * hd);
  uint seqStride = (sl * headStride);
  uint qBase = ((batchIdx * seqStride) + ((queryPos * headStride) + (headIdx * headDimStride)));
  uint kBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
  if (valid) {
    float dotProd = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half qValF16 = query[qIdx];
      half kValF16 = key[kIdx];
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
    attnScores[lid] = as_float(0xf149f2cau);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_128 = (lid < (uint)(128u));
  if (shouldReduce_max_p1_attnScores_128) {
    uint neighborIdx_max_p1_attnScores_128 = (lid + (uint)(128u));
    float myVal_max_p1_attnScores_128 = attnScores[lid];
    float neighborVal_max_p1_attnScores_128 = attnScores[neighborIdx_max_p1_attnScores_128];
    bool isGreater_max_p1_attnScores_128 = (neighborVal_max_p1_attnScores_128 > myVal_max_p1_attnScores_128);
    float maxVal_max_p1_attnScores_128 = ((isGreater_max_p1_attnScores_128) ? (neighborVal_max_p1_attnScores_128) : (myVal_max_p1_attnScores_128));
    attnScores[lid] = maxVal_max_p1_attnScores_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_64 = (lid < (uint)(64u));
  if (shouldReduce_max_p1_attnScores_64) {
    uint neighborIdx_max_p1_attnScores_64 = (lid + (uint)(64u));
    float myVal_max_p1_attnScores_64 = attnScores[lid];
    float neighborVal_max_p1_attnScores_64 = attnScores[neighborIdx_max_p1_attnScores_64];
    bool isGreater_max_p1_attnScores_64 = (neighborVal_max_p1_attnScores_64 > myVal_max_p1_attnScores_64);
    float maxVal_max_p1_attnScores_64 = ((isGreater_max_p1_attnScores_64) ? (neighborVal_max_p1_attnScores_64) : (myVal_max_p1_attnScores_64));
    attnScores[lid] = maxVal_max_p1_attnScores_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_32 = (lid < (uint)(32u));
  if (shouldReduce_max_p1_attnScores_32) {
    uint neighborIdx_max_p1_attnScores_32 = (lid + (uint)(32u));
    float myVal_max_p1_attnScores_32 = attnScores[lid];
    float neighborVal_max_p1_attnScores_32 = attnScores[neighborIdx_max_p1_attnScores_32];
    bool isGreater_max_p1_attnScores_32 = (neighborVal_max_p1_attnScores_32 > myVal_max_p1_attnScores_32);
    float maxVal_max_p1_attnScores_32 = ((isGreater_max_p1_attnScores_32) ? (neighborVal_max_p1_attnScores_32) : (myVal_max_p1_attnScores_32));
    attnScores[lid] = maxVal_max_p1_attnScores_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_16 = (lid < (uint)(16u));
  if (shouldReduce_max_p1_attnScores_16) {
    uint neighborIdx_max_p1_attnScores_16 = (lid + (uint)(16u));
    float myVal_max_p1_attnScores_16 = attnScores[lid];
    float neighborVal_max_p1_attnScores_16 = attnScores[neighborIdx_max_p1_attnScores_16];
    bool isGreater_max_p1_attnScores_16 = (neighborVal_max_p1_attnScores_16 > myVal_max_p1_attnScores_16);
    float maxVal_max_p1_attnScores_16 = ((isGreater_max_p1_attnScores_16) ? (neighborVal_max_p1_attnScores_16) : (myVal_max_p1_attnScores_16));
    attnScores[lid] = maxVal_max_p1_attnScores_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_8 = (lid < (uint)(8u));
  if (shouldReduce_max_p1_attnScores_8) {
    uint neighborIdx_max_p1_attnScores_8 = (lid + (uint)(8u));
    float myVal_max_p1_attnScores_8 = attnScores[lid];
    float neighborVal_max_p1_attnScores_8 = attnScores[neighborIdx_max_p1_attnScores_8];
    bool isGreater_max_p1_attnScores_8 = (neighborVal_max_p1_attnScores_8 > myVal_max_p1_attnScores_8);
    float maxVal_max_p1_attnScores_8 = ((isGreater_max_p1_attnScores_8) ? (neighborVal_max_p1_attnScores_8) : (myVal_max_p1_attnScores_8));
    attnScores[lid] = maxVal_max_p1_attnScores_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_4 = (lid < (uint)(4u));
  if (shouldReduce_max_p1_attnScores_4) {
    uint neighborIdx_max_p1_attnScores_4 = (lid + (uint)(4u));
    float myVal_max_p1_attnScores_4 = attnScores[lid];
    float neighborVal_max_p1_attnScores_4 = attnScores[neighborIdx_max_p1_attnScores_4];
    bool isGreater_max_p1_attnScores_4 = (neighborVal_max_p1_attnScores_4 > myVal_max_p1_attnScores_4);
    float maxVal_max_p1_attnScores_4 = ((isGreater_max_p1_attnScores_4) ? (neighborVal_max_p1_attnScores_4) : (myVal_max_p1_attnScores_4));
    attnScores[lid] = maxVal_max_p1_attnScores_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_2 = (lid < (uint)(2u));
  if (shouldReduce_max_p1_attnScores_2) {
    uint neighborIdx_max_p1_attnScores_2 = (lid + (uint)(2u));
    float myVal_max_p1_attnScores_2 = attnScores[lid];
    float neighborVal_max_p1_attnScores_2 = attnScores[neighborIdx_max_p1_attnScores_2];
    bool isGreater_max_p1_attnScores_2 = (neighborVal_max_p1_attnScores_2 > myVal_max_p1_attnScores_2);
    float maxVal_max_p1_attnScores_2 = ((isGreater_max_p1_attnScores_2) ? (neighborVal_max_p1_attnScores_2) : (myVal_max_p1_attnScores_2));
    attnScores[lid] = maxVal_max_p1_attnScores_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_p1_attnScores_1 = (lid < (uint)(1u));
  if (shouldReduce_max_p1_attnScores_1) {
    uint neighborIdx_max_p1_attnScores_1 = (lid + (uint)(1u));
    float myVal_max_p1_attnScores_1 = attnScores[lid];
    float neighborVal_max_p1_attnScores_1 = attnScores[neighborIdx_max_p1_attnScores_1];
    bool isGreater_max_p1_attnScores_1 = (neighborVal_max_p1_attnScores_1 > myVal_max_p1_attnScores_1);
    float maxVal_max_p1_attnScores_1 = ((isGreater_max_p1_attnScores_1) ? (neighborVal_max_p1_attnScores_1) : (myVal_max_p1_attnScores_1));
    attnScores[lid] = maxVal_max_p1_attnScores_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxScore = attnScores[(uint)(0u)];
  if (valid) {
    float myScore = attnScores[lid];
    float shiftedScore = (myScore - maxScore);
    float expScore = exp(shiftedScore);
    sumBuf[lid] = expScore;
  } else {
    sumBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_p1_sumBuf_128) {
    uint neighborIdx_sum_p1_sumBuf_128 = (lid + (uint)(128u));
    float myVal_sum_p1_sumBuf_128 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_128 = sumBuf[neighborIdx_sum_p1_sumBuf_128];
    float sumVal_sum_p1_sumBuf_128 = (myVal_sum_p1_sumBuf_128 + neighborVal_sum_p1_sumBuf_128);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_p1_sumBuf_64) {
    uint neighborIdx_sum_p1_sumBuf_64 = (lid + (uint)(64u));
    float myVal_sum_p1_sumBuf_64 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_64 = sumBuf[neighborIdx_sum_p1_sumBuf_64];
    float sumVal_sum_p1_sumBuf_64 = (myVal_sum_p1_sumBuf_64 + neighborVal_sum_p1_sumBuf_64);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_p1_sumBuf_32) {
    uint neighborIdx_sum_p1_sumBuf_32 = (lid + (uint)(32u));
    float myVal_sum_p1_sumBuf_32 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_32 = sumBuf[neighborIdx_sum_p1_sumBuf_32];
    float sumVal_sum_p1_sumBuf_32 = (myVal_sum_p1_sumBuf_32 + neighborVal_sum_p1_sumBuf_32);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_p1_sumBuf_16) {
    uint neighborIdx_sum_p1_sumBuf_16 = (lid + (uint)(16u));
    float myVal_sum_p1_sumBuf_16 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_16 = sumBuf[neighborIdx_sum_p1_sumBuf_16];
    float sumVal_sum_p1_sumBuf_16 = (myVal_sum_p1_sumBuf_16 + neighborVal_sum_p1_sumBuf_16);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_p1_sumBuf_8) {
    uint neighborIdx_sum_p1_sumBuf_8 = (lid + (uint)(8u));
    float myVal_sum_p1_sumBuf_8 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_8 = sumBuf[neighborIdx_sum_p1_sumBuf_8];
    float sumVal_sum_p1_sumBuf_8 = (myVal_sum_p1_sumBuf_8 + neighborVal_sum_p1_sumBuf_8);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_p1_sumBuf_4) {
    uint neighborIdx_sum_p1_sumBuf_4 = (lid + (uint)(4u));
    float myVal_sum_p1_sumBuf_4 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_4 = sumBuf[neighborIdx_sum_p1_sumBuf_4];
    float sumVal_sum_p1_sumBuf_4 = (myVal_sum_p1_sumBuf_4 + neighborVal_sum_p1_sumBuf_4);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_p1_sumBuf_2) {
    uint neighborIdx_sum_p1_sumBuf_2 = (lid + (uint)(2u));
    float myVal_sum_p1_sumBuf_2 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_2 = sumBuf[neighborIdx_sum_p1_sumBuf_2];
    float sumVal_sum_p1_sumBuf_2 = (myVal_sum_p1_sumBuf_2 + neighborVal_sum_p1_sumBuf_2);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_p1_sumBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_p1_sumBuf_1) {
    uint neighborIdx_sum_p1_sumBuf_1 = (lid + (uint)(1u));
    float myVal_sum_p1_sumBuf_1 = sumBuf[lid];
    float neighborVal_sum_p1_sumBuf_1 = sumBuf[neighborIdx_sum_p1_sumBuf_1];
    float sumVal_sum_p1_sumBuf_1 = (myVal_sum_p1_sumBuf_1 + neighborVal_sum_p1_sumBuf_1);
    sumBuf[lid] = sumVal_sum_p1_sumBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = sumBuf[(uint)(0u)];
  if (valid) {
    float myExpScore = attnScores[lid];
    float shiftedExp = (myExpScore - maxScore);
    float expVal = exp(shiftedExp);
    float attnWeight = (expVal / sumExp);
    uint vBase = ((batchIdx * seqStride) + ((lid * headStride) + (headIdx * headDimStride)));
    float gradAttnAccum = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      uint gradOutIdx = (qBase + ((uint)(d)));
      uint vIdx = (vBase + ((uint)(d)));
      half gradOutF16 = gradOutput[gradOutIdx];
      float gradOutVal = ((float)(gradOutF16));
      half vValF16 = value[vIdx];
      float vVal = ((float)(vValF16));
      float gradVContrib = (attnWeight * gradOutVal);
      half oldGradVF16 = gradValue[vIdx];
      float oldGradV = ((float)(oldGradVF16));
      float newGradV = (oldGradV + gradVContrib);
      half newGradVF16 = ((half)(newGradV));
      gradValue[vIdx] = newGradVF16;
      float gradAttnProd = (gradOutVal * vVal);
      float oldGradAttnAccum = gradAttnAccum;
      gradAttnAccum = (oldGradAttnAccum + gradAttnProd);
    }
    float gradAttn = gradAttnAccum;
    gradBuf[lid] = gradAttn;
  } else {
    gradBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (valid) {
    float myAttnWeight = (exp((attnScores[lid] - maxScore)) / sumExp);
    float myGradAttn = gradBuf[lid];
    float gradScore = (myAttnWeight * myGradAttn);
    float scaledGradScore = (gradScore * scale);
    for (int d = 0; d < (int)(hd); ++d) {
      uint qIdx = (qBase + ((uint)(d)));
      uint kIdx = (kBase + ((uint)(d)));
      half kValF16 = key[kIdx];
      float kVal = ((float)(kValF16));
      float gradQContrib = (scaledGradScore * kVal);
      half oldGradQF16 = gradQuery[qIdx];
      float oldGradQ = ((float)(oldGradQF16));
      float newGradQ = (oldGradQ + gradQContrib);
      half newGradQF16 = ((half)(newGradQ));
      gradQuery[qIdx] = newGradQF16;
      half qValF16 = query[qIdx];
      float qVal = ((float)(qValF16));
      float gradKContrib = (scaledGradScore * qVal);
      half oldGradKF16 = gradKey[kIdx];
      float oldGradK = ((float)(oldGradKF16));
      float newGradK = (oldGradK + gradKContrib);
      half newGradKF16 = ((half)(newGradK));
      gradKey[kIdx] = newGradKF16;
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_kv_cache_append_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_kv_cache_append_f16(__global const half* newK, __global const half* newV, __global const uint* blockTables, __global const uint* seqPositions, __global const uint* numHeads, __global const uint* headDim, __global const uint* pageSize, __global const uint* maxBlocks, __global half* kvCache) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint ps = pageSize[(uint)(0u)];
  uint mb = maxBlocks[(uint)(0u)];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint seqPos = seqPositions[batchIdx];
  uint blockIdx = (seqPos / ps);
  uint blockOffset = (seqPos % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = ((uint)(2u) * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  uint srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    uint srcIdx = (srcBase + lid);
    uint kDstIdx = (kCacheBase + lid);
    uint vDstIdx = (vCacheBase + lid);
    half kVal = newK[srcIdx];
    half vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_kv_cache_append_f32 = R"kernel(
__kernel void navatala_transformer_kv_cache_append_f32(__global const float* newK, __global const float* newV, __global const uint* blockTables, __global const uint* seqPositions, __global const uint* numHeads, __global const uint* headDim, __global const uint* pageSize, __global const uint* maxBlocks, __global float* kvCache) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint ps = pageSize[(uint)(0u)];
  uint mb = maxBlocks[(uint)(0u)];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint seqPos = seqPositions[batchIdx];
  uint blockIdx = (seqPos / ps);
  uint blockOffset = (seqPos % ps);
  uint blockTableBase = (batchIdx * mb);
  uint physicalBlock = blockTables[(blockTableBase + blockIdx)];
  uint kvBlockStride = ((uint)(2u) * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  uint kCacheBase = ((physicalBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (blockOffset * hd)));
  uint vCacheBase = ((physicalBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (blockOffset * hd))));
  uint srcBase = ((batchIdx * (nh * hd)) + (headIdx * hd));
  bool dimValid = (lid < hd);
  if (dimValid) {
    uint srcIdx = (srcBase + lid);
    uint kDstIdx = (kCacheBase + lid);
    uint vDstIdx = (vCacheBase + lid);
    float kVal = newK[srcIdx];
    float vVal = newV[srcIdx];
    kvCache[kDstIdx] = kVal;
    kvCache[vDstIdx] = vVal;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_kv_cache_rotate_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_kv_cache_rotate_f16(__global const uint* blockTables, __global const uint* contextLens, __global const uint* shiftAmount, __global const uint* numHeads, __global const uint* headDim, __global const uint* pageSize, __global const uint* maxBlocks, __global half* kvCache) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint groupId = ((uint)((int)(get_group_id(0))));
  uint nh = numHeads[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  uint ps = pageSize[(uint)(0u)];
  uint mb = maxBlocks[(uint)(0u)];
  uint shift = shiftAmount[(uint)(0u)];
  uint batchIdx = (groupId / nh);
  uint headIdx = (groupId % nh);
  uint ctxLen = contextLens[batchIdx];
  bool posValid = (lid < ctxLen);
  uint srcPos = (lid + shift);
  bool srcValid = (srcPos < ctxLen);
  bool valid = (posValid && srcValid);
  uint blockTableBase = (batchIdx * mb);
  uint kvBlockStride = ((uint)(2u) * (nh * (ps * hd)));
  uint kvSlotStride = (nh * (ps * hd));
  uint kvHeadStride = (ps * hd);
  if (valid) {
    uint srcBlockIdx = (srcPos / ps);
    uint srcBlockOffset = (srcPos % ps);
    uint srcPhysBlock = blockTables[(blockTableBase + srcBlockIdx)];
    uint dstBlockIdx = (lid / ps);
    uint dstBlockOffset = (lid % ps);
    uint dstPhysBlock = blockTables[(blockTableBase + dstBlockIdx)];
    uint kSrcBase = ((srcPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (srcBlockOffset * hd)));
    uint kDstBase = ((dstPhysBlock * kvBlockStride) + ((headIdx * kvHeadStride) + (dstBlockOffset * hd)));
    uint vSrcBase = ((srcPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (srcBlockOffset * hd))));
    uint vDstBase = ((dstPhysBlock * kvBlockStride) + (kvSlotStride + ((headIdx * kvHeadStride) + (dstBlockOffset * hd))));
    for (int d = 0; d < (int)(hd); ++d) {
      uint kSrcIdx = (kSrcBase + ((uint)(d)));
      uint kDstIdx = (kDstBase + ((uint)(d)));
      uint vSrcIdx = (vSrcBase + ((uint)(d)));
      uint vDstIdx = (vDstBase + ((uint)(d)));
      half kVal = kvCache[kSrcIdx];
      half vVal = kvCache[vSrcIdx];
      kvCache[kDstIdx] = kVal;
      kvCache[vDstIdx] = vVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_tiled_gemm_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_tiled_gemm_f16(__global const half* a, __global const half* b, __global const uint* m, __global const uint* n, __global const uint* k, __global const float* alpha, __global const float* beta, __global half* c) {
  int gid0 = (int)get_global_id(0);
  uint tidX = ((uint)((int)(get_local_id(0))));
  uint tidY = ((uint)((int)(get_local_id(1))));
  uint gidX = ((uint)((int)(get_group_id(0))));
  uint gidY = ((uint)((int)(get_group_id(1))));
  uint mDim = m[(uint)(0u)];
  uint nDim = n[(uint)(0u)];
  uint kDim = k[(uint)(0u)];
  float alphaVal = alpha[(uint)(0u)];
  float betaVal = beta[(uint)(0u)];
  uint tileSize = (uint)(16u);
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __local float tileA[256];
  __local float tileB[256];
  if (valid) {
    float acc = as_float(0x00000000u);
    for (int kIter = 0; kIter < (int)(kDim); ++kIter) {
      uint aIdx = ((row * kDim) + kIter);
      uint bIdx = ((kIter * nDim) + col);
      half aValF16 = a[aIdx];
      half bValF16 = b[bIdx];
      float aVal = ((float)(aValF16));
      float bVal = ((float)(bValF16));
      float prod = (aVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    uint cIdx = ((row * nDim) + col);
    half cOldF16 = c[cIdx];
    float cOld = ((float)(cOldF16));
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    half resultF16 = ((half)(result));
    c[cIdx] = resultF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_tiled_gemm_f32 = R"kernel(
__kernel void navatala_transformer_tiled_gemm_f32(__global const float* a, __global const float* b, __global const uint* m, __global const uint* n, __global const uint* k, __global const float* alpha, __global const float* beta, __global float* c) {
  int gid0 = (int)get_global_id(0);
  uint tidX = ((uint)((int)(get_local_id(0))));
  uint tidY = ((uint)((int)(get_local_id(1))));
  uint gidX = ((uint)((int)(get_group_id(0))));
  uint gidY = ((uint)((int)(get_group_id(1))));
  uint mDim = m[(uint)(0u)];
  uint nDim = n[(uint)(0u)];
  uint kDim = k[(uint)(0u)];
  float alphaVal = alpha[(uint)(0u)];
  float betaVal = beta[(uint)(0u)];
  uint tileSize = (uint)(16u);
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __local float tileA[256];
  __local float tileB[256];
  if (valid) {
    float acc = as_float(0x00000000u);
    for (int kIter = 0; kIter < (int)(kDim); ++kIter) {
      uint aIdx = ((row * kDim) + kIter);
      uint bIdx = ((kIter * nDim) + col);
      float aVal = a[aIdx];
      float bVal = b[bIdx];
      float prod = (aVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    uint cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_tiled_gemm_backward_f32 = R"kernel(
__kernel void navatala_transformer_tiled_gemm_backward_f32(__global const float* gradC, __global const float* b, __global const uint* m, __global const uint* n, __global const uint* k, __global float* gradA) {
  int gid0 = (int)get_global_id(0);
  uint tidX = ((uint)((int)(get_local_id(0))));
  uint tidY = ((uint)((int)(get_local_id(1))));
  uint gidX = ((uint)((int)(get_group_id(0))));
  uint gidY = ((uint)((int)(get_group_id(1))));
  uint mDim = m[(uint)(0u)];
  uint nDim = n[(uint)(0u)];
  uint kDim = k[(uint)(0u)];
  uint tileSize = (uint)(16u);
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  __local float tileGradC[256];
  __local float tileBT[256];
  if (valid) {
    float acc = as_float(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      uint gradCIdx = ((row * nDim) + jIter);
      uint bIdx = ((col * nDim) + jIter);
      float gradCVal = gradC[gradCIdx];
      float bVal = b[bIdx];
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    uint gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAcc;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_tiled_gemm_backward_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_tiled_gemm_backward_f16(__global const half* gradC, __global const half* b, __global const uint* m, __global const uint* n, __global const uint* k, __global half* gradA) {
  int gid0 = (int)get_global_id(0);
  uint tidX = ((uint)((int)(get_local_id(0))));
  uint tidY = ((uint)((int)(get_local_id(1))));
  uint gidX = ((uint)((int)(get_group_id(0))));
  uint gidY = ((uint)((int)(get_group_id(1))));
  uint mDim = m[(uint)(0u)];
  uint nDim = n[(uint)(0u)];
  uint kDim = k[(uint)(0u)];
  uint tileSize = (uint)(16u);
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  __local float tileGradC[256];
  __local float tileBT[256];
  if (valid) {
    float acc = as_float(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      uint gradCIdx = ((row * nDim) + jIter);
      uint bIdx = ((col * nDim) + jIter);
      half gradCValF16 = gradC[gradCIdx];
      half bValF16 = b[bIdx];
      float gradCVal = ((float)(gradCValF16));
      float bVal = ((float)(bValF16));
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    half finalAccF16 = ((half)(finalAcc));
    uint gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAccF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_scale_f32 = R"kernel(
__kernel void navatala_transformer_multi_tensor_scale_f32(__global const float* _input, __global const float* scale, __global const uint* totalElements, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float s = scale[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x * s);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_scale_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_multi_tensor_scale_f16(__global const half* _input, __global const float* scale, __global const uint* totalElements, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float s = scale[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    half xF16 = _input[gid];
    float x = ((float)(xF16));
    float resultF32 = (x * s);
    half result = ((half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_add_f32 = R"kernel(
__kernel void navatala_transformer_multi_tensor_add_f32(__global const float* _input, __global const float* value, __global const uint* totalElements, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float v = value[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x + v);
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_add_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_multi_tensor_add_f16(__global const half* _input, __global const float* value, __global const uint* totalElements, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float v = value[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    half xF16 = _input[gid];
    float x = ((float)(xF16));
    float resultF32 = (x + v);
    half result = ((half)(resultF32));
    _output[gid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_copy_f32 = R"kernel(
__kernel void navatala_transformer_multi_tensor_copy_f32(__global const float* _input, __global const uint* totalElements, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    _output[gid] = x;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_copy_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_multi_tensor_copy_f16(__global const half* _input, __global const uint* totalElements, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  bool valid = (gid < total);
  if (valid) {
    half x = _input[gid];
    _output[gid] = x;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_l2_norm_f32 = R"kernel(
__kernel void navatala_transformer_multi_tensor_l2_norm_f32(__global const float* _input, __global const uint* totalElements, __global float* partialSums) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint gid = ((uint)((int)(get_group_id(0))));
  uint globalIdx = ((gid * (uint)(256u)) + lid);
  uint total = totalElements[(uint)(0u)];
  __local float sumSq[256];
  bool valid = (globalIdx < total);
  float x = ((valid) ? (_input[globalIdx]) : (as_float(0x00000000u)));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + (uint)(128u));
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + (uint)(64u));
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + (uint)(32u));
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + (uint)(16u));
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + (uint)(8u));
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + (uint)(4u));
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + (uint)(2u));
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + (uint)(1u));
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid == (uint)(0u))) {
    float partialSum = sumSq[(uint)(0u)];
    partialSums[gid] = partialSum;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_l2_norm_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_multi_tensor_l2_norm_f16(__global const half* _input, __global const uint* totalElements, __global float* partialSums) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint gid = ((uint)((int)(get_group_id(0))));
  uint globalIdx = ((gid * (uint)(256u)) + lid);
  uint total = totalElements[(uint)(0u)];
  __local float sumSq[256];
  bool valid = (globalIdx < total);
  half xF16 = ((valid) ? (_input[globalIdx]) : ((half)(0.000000f)));
  float x = ((float)(xF16));
  float xSq = (x * x);
  sumSq[lid] = xSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_128 = (lid < (uint)(128u));
  if (shouldReduce_sumSq_128) {
    uint neighborIdx_sumSq_128 = (lid + (uint)(128u));
    float myVal_sumSq_128 = sumSq[lid];
    float neighborVal_sumSq_128 = sumSq[neighborIdx_sumSq_128];
    float sumVal_sumSq_128 = (myVal_sumSq_128 + neighborVal_sumSq_128);
    sumSq[lid] = sumVal_sumSq_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_64 = (lid < (uint)(64u));
  if (shouldReduce_sumSq_64) {
    uint neighborIdx_sumSq_64 = (lid + (uint)(64u));
    float myVal_sumSq_64 = sumSq[lid];
    float neighborVal_sumSq_64 = sumSq[neighborIdx_sumSq_64];
    float sumVal_sumSq_64 = (myVal_sumSq_64 + neighborVal_sumSq_64);
    sumSq[lid] = sumVal_sumSq_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_32 = (lid < (uint)(32u));
  if (shouldReduce_sumSq_32) {
    uint neighborIdx_sumSq_32 = (lid + (uint)(32u));
    float myVal_sumSq_32 = sumSq[lid];
    float neighborVal_sumSq_32 = sumSq[neighborIdx_sumSq_32];
    float sumVal_sumSq_32 = (myVal_sumSq_32 + neighborVal_sumSq_32);
    sumSq[lid] = sumVal_sumSq_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_16 = (lid < (uint)(16u));
  if (shouldReduce_sumSq_16) {
    uint neighborIdx_sumSq_16 = (lid + (uint)(16u));
    float myVal_sumSq_16 = sumSq[lid];
    float neighborVal_sumSq_16 = sumSq[neighborIdx_sumSq_16];
    float sumVal_sumSq_16 = (myVal_sumSq_16 + neighborVal_sumSq_16);
    sumSq[lid] = sumVal_sumSq_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_8 = (lid < (uint)(8u));
  if (shouldReduce_sumSq_8) {
    uint neighborIdx_sumSq_8 = (lid + (uint)(8u));
    float myVal_sumSq_8 = sumSq[lid];
    float neighborVal_sumSq_8 = sumSq[neighborIdx_sumSq_8];
    float sumVal_sumSq_8 = (myVal_sumSq_8 + neighborVal_sumSq_8);
    sumSq[lid] = sumVal_sumSq_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_4 = (lid < (uint)(4u));
  if (shouldReduce_sumSq_4) {
    uint neighborIdx_sumSq_4 = (lid + (uint)(4u));
    float myVal_sumSq_4 = sumSq[lid];
    float neighborVal_sumSq_4 = sumSq[neighborIdx_sumSq_4];
    float sumVal_sumSq_4 = (myVal_sumSq_4 + neighborVal_sumSq_4);
    sumSq[lid] = sumVal_sumSq_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_2 = (lid < (uint)(2u));
  if (shouldReduce_sumSq_2) {
    uint neighborIdx_sumSq_2 = (lid + (uint)(2u));
    float myVal_sumSq_2 = sumSq[lid];
    float neighborVal_sumSq_2 = sumSq[neighborIdx_sumSq_2];
    float sumVal_sumSq_2 = (myVal_sumSq_2 + neighborVal_sumSq_2);
    sumSq[lid] = sumVal_sumSq_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sumSq_1 = (lid < (uint)(1u));
  if (shouldReduce_sumSq_1) {
    uint neighborIdx_sumSq_1 = (lid + (uint)(1u));
    float myVal_sumSq_1 = sumSq[lid];
    float neighborVal_sumSq_1 = sumSq[neighborIdx_sumSq_1];
    float sumVal_sumSq_1 = (myVal_sumSq_1 + neighborVal_sumSq_1);
    sumSq[lid] = sumVal_sumSq_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid == (uint)(0u))) {
    float partialSum = sumSq[(uint)(0u)];
    partialSums[gid] = partialSum;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_clip_grad_f32 = R"kernel(
__kernel void navatala_transformer_multi_tensor_clip_grad_f32(__global const float* totalNorm, __global const float* maxNorm, __global const uint* totalElements, __global float* grad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float tNorm = totalNorm[(uint)(0u)];
  float mNorm = maxNorm[(uint)(0u)];
  float epsilon = as_float(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (as_float(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    float g = grad[gid];
    float clipped = (g * clipCoef);
    grad[gid] = clipped;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_multi_tensor_clip_grad_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_multi_tensor_clip_grad_f16(__global const float* totalNorm, __global const float* maxNorm, __global const uint* totalElements, __global half* grad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float tNorm = totalNorm[(uint)(0u)];
  float mNorm = maxNorm[(uint)(0u)];
  float epsilon = as_float(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (as_float(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    half gF16 = grad[gid];
    float g = ((float)(gF16));
    float clippedF32 = (g * clipCoef);
    half clipped = ((half)(clippedF32));
    grad[gid] = clipped;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_top_k_gating_f32 = R"kernel(
__kernel void navatala_transformer_top_k_gating_f32(__global const float* routerLogits, __global const uint* numTokens, __global const uint* numExperts, __global const uint* topK, __global int* expertIndices, __global float* expertWeights) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint tokenIdx = ((uint)((int)(get_group_id(0))));
  uint nt = numTokens[(uint)(0u)];
  uint ne = numExperts[(uint)(0u)];
  uint k = topK[(uint)(0u)];
  __local float logitsBuf[256];
  __local float expBuf[256];
  __local int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  uint globalIdx = ((tokenIdx * ne) + lid);
  float logit = ((valid) ? (routerLogits[globalIdx]) : (as_float(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_logitsBuf_128) {
    uint neighborIdx_max_logitsBuf_128 = (lid + (uint)(128u));
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_logitsBuf_64) {
    uint neighborIdx_max_logitsBuf_64 = (lid + (uint)(64u));
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_logitsBuf_32) {
    uint neighborIdx_max_logitsBuf_32 = (lid + (uint)(32u));
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_logitsBuf_16) {
    uint neighborIdx_max_logitsBuf_16 = (lid + (uint)(16u));
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_logitsBuf_8) {
    uint neighborIdx_max_logitsBuf_8 = (lid + (uint)(8u));
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_logitsBuf_4) {
    uint neighborIdx_max_logitsBuf_4 = (lid + (uint)(4u));
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_logitsBuf_2) {
    uint neighborIdx_max_logitsBuf_2 = (lid + (uint)(2u));
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_logitsBuf_1) {
    uint neighborIdx_max_logitsBuf_1 = (lid + (uint)(1u));
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxLogit = logitsBuf[(uint)(0u)];
  float originalLogit = ((valid) ? (routerLogits[globalIdx]) : (as_float(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (as_float(0x00000000u)));
  expBuf[lid] = expMasked;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_expBuf_128) {
    uint neighborIdx_sum_expBuf_128 = (lid + (uint)(128u));
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_expBuf_64) {
    uint neighborIdx_sum_expBuf_64 = (lid + (uint)(64u));
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_expBuf_32) {
    uint neighborIdx_sum_expBuf_32 = (lid + (uint)(32u));
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_expBuf_16) {
    uint neighborIdx_sum_expBuf_16 = (lid + (uint)(16u));
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_expBuf_8) {
    uint neighborIdx_sum_expBuf_8 = (lid + (uint)(8u));
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_expBuf_4) {
    uint neighborIdx_sum_expBuf_4 = (lid + (uint)(4u));
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_expBuf_2) {
    uint neighborIdx_sum_expBuf_2 = (lid + (uint)(2u));
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_expBuf_1) {
    uint neighborIdx_sum_expBuf_1 = (lid + (uint)(1u));
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = expBuf[(uint)(0u)];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  indexBuf[lid] = lidI32;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == (uint)(0u))) {
      int bestIdx = 0;
      float bestVal = as_float(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      uint kIterU32 = kIter;
      uint outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValFinal;
      uint bestIdxU32 = ((uint)(bestIdxFinal));
      logitsBuf[bestIdxU32] = as_float(0xf149f2cau);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_top_k_gating_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_top_k_gating_f16(__global const half* routerLogits, __global const uint* numTokens, __global const uint* numExperts, __global const uint* topK, __global int* expertIndices, __global half* expertWeights) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint tokenIdx = ((uint)((int)(get_group_id(0))));
  uint nt = numTokens[(uint)(0u)];
  uint ne = numExperts[(uint)(0u)];
  uint k = topK[(uint)(0u)];
  __local float logitsBuf[256];
  __local float expBuf[256];
  __local int indexBuf[256];
  bool tokenValid = (tokenIdx < nt);
  bool expertValid = (lid < ne);
  bool valid = (tokenValid && expertValid);
  uint globalIdx = ((tokenIdx * ne) + lid);
  half logitF16 = ((valid) ? (routerLogits[globalIdx]) : ((half)(0.000000f)));
  float logit = ((expertValid) ? (((float)(logitF16))) : (as_float(0xf149f2cau)));
  logitsBuf[lid] = logit;
  int lidI32 = ((int)(lid));
  indexBuf[lid] = lidI32;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_max_logitsBuf_128) {
    uint neighborIdx_max_logitsBuf_128 = (lid + (uint)(128u));
    float myVal_max_logitsBuf_128 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_128 = logitsBuf[neighborIdx_max_logitsBuf_128];
    bool isGreater_max_logitsBuf_128 = (neighborVal_max_logitsBuf_128 > myVal_max_logitsBuf_128);
    float maxVal_max_logitsBuf_128 = ((isGreater_max_logitsBuf_128) ? (neighborVal_max_logitsBuf_128) : (myVal_max_logitsBuf_128));
    logitsBuf[lid] = maxVal_max_logitsBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_max_logitsBuf_64) {
    uint neighborIdx_max_logitsBuf_64 = (lid + (uint)(64u));
    float myVal_max_logitsBuf_64 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_64 = logitsBuf[neighborIdx_max_logitsBuf_64];
    bool isGreater_max_logitsBuf_64 = (neighborVal_max_logitsBuf_64 > myVal_max_logitsBuf_64);
    float maxVal_max_logitsBuf_64 = ((isGreater_max_logitsBuf_64) ? (neighborVal_max_logitsBuf_64) : (myVal_max_logitsBuf_64));
    logitsBuf[lid] = maxVal_max_logitsBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_max_logitsBuf_32) {
    uint neighborIdx_max_logitsBuf_32 = (lid + (uint)(32u));
    float myVal_max_logitsBuf_32 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_32 = logitsBuf[neighborIdx_max_logitsBuf_32];
    bool isGreater_max_logitsBuf_32 = (neighborVal_max_logitsBuf_32 > myVal_max_logitsBuf_32);
    float maxVal_max_logitsBuf_32 = ((isGreater_max_logitsBuf_32) ? (neighborVal_max_logitsBuf_32) : (myVal_max_logitsBuf_32));
    logitsBuf[lid] = maxVal_max_logitsBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_max_logitsBuf_16) {
    uint neighborIdx_max_logitsBuf_16 = (lid + (uint)(16u));
    float myVal_max_logitsBuf_16 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_16 = logitsBuf[neighborIdx_max_logitsBuf_16];
    bool isGreater_max_logitsBuf_16 = (neighborVal_max_logitsBuf_16 > myVal_max_logitsBuf_16);
    float maxVal_max_logitsBuf_16 = ((isGreater_max_logitsBuf_16) ? (neighborVal_max_logitsBuf_16) : (myVal_max_logitsBuf_16));
    logitsBuf[lid] = maxVal_max_logitsBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_max_logitsBuf_8) {
    uint neighborIdx_max_logitsBuf_8 = (lid + (uint)(8u));
    float myVal_max_logitsBuf_8 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_8 = logitsBuf[neighborIdx_max_logitsBuf_8];
    bool isGreater_max_logitsBuf_8 = (neighborVal_max_logitsBuf_8 > myVal_max_logitsBuf_8);
    float maxVal_max_logitsBuf_8 = ((isGreater_max_logitsBuf_8) ? (neighborVal_max_logitsBuf_8) : (myVal_max_logitsBuf_8));
    logitsBuf[lid] = maxVal_max_logitsBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_max_logitsBuf_4) {
    uint neighborIdx_max_logitsBuf_4 = (lid + (uint)(4u));
    float myVal_max_logitsBuf_4 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_4 = logitsBuf[neighborIdx_max_logitsBuf_4];
    bool isGreater_max_logitsBuf_4 = (neighborVal_max_logitsBuf_4 > myVal_max_logitsBuf_4);
    float maxVal_max_logitsBuf_4 = ((isGreater_max_logitsBuf_4) ? (neighborVal_max_logitsBuf_4) : (myVal_max_logitsBuf_4));
    logitsBuf[lid] = maxVal_max_logitsBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_max_logitsBuf_2) {
    uint neighborIdx_max_logitsBuf_2 = (lid + (uint)(2u));
    float myVal_max_logitsBuf_2 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_2 = logitsBuf[neighborIdx_max_logitsBuf_2];
    bool isGreater_max_logitsBuf_2 = (neighborVal_max_logitsBuf_2 > myVal_max_logitsBuf_2);
    float maxVal_max_logitsBuf_2 = ((isGreater_max_logitsBuf_2) ? (neighborVal_max_logitsBuf_2) : (myVal_max_logitsBuf_2));
    logitsBuf[lid] = maxVal_max_logitsBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_max_logitsBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_max_logitsBuf_1) {
    uint neighborIdx_max_logitsBuf_1 = (lid + (uint)(1u));
    float myVal_max_logitsBuf_1 = logitsBuf[lid];
    float neighborVal_max_logitsBuf_1 = logitsBuf[neighborIdx_max_logitsBuf_1];
    bool isGreater_max_logitsBuf_1 = (neighborVal_max_logitsBuf_1 > myVal_max_logitsBuf_1);
    float maxVal_max_logitsBuf_1 = ((isGreater_max_logitsBuf_1) ? (neighborVal_max_logitsBuf_1) : (myVal_max_logitsBuf_1));
    logitsBuf[lid] = maxVal_max_logitsBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maxLogit = logitsBuf[(uint)(0u)];
  half originalLogitF16 = ((valid) ? (routerLogits[globalIdx]) : ((half)(0.000000f)));
  float originalLogit = ((expertValid) ? (((float)(originalLogitF16))) : (as_float(0xf149f2cau)));
  float shiftedLogit = (originalLogit - maxLogit);
  float expLogit = exp(shiftedLogit);
  float expMasked = ((expertValid) ? (expLogit) : (as_float(0x00000000u)));
  expBuf[lid] = expMasked;
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_128 = (lid < (uint)(128u));
  if (shouldReduce_sum_expBuf_128) {
    uint neighborIdx_sum_expBuf_128 = (lid + (uint)(128u));
    float myVal_sum_expBuf_128 = expBuf[lid];
    float neighborVal_sum_expBuf_128 = expBuf[neighborIdx_sum_expBuf_128];
    float sumVal_sum_expBuf_128 = (myVal_sum_expBuf_128 + neighborVal_sum_expBuf_128);
    expBuf[lid] = sumVal_sum_expBuf_128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_64 = (lid < (uint)(64u));
  if (shouldReduce_sum_expBuf_64) {
    uint neighborIdx_sum_expBuf_64 = (lid + (uint)(64u));
    float myVal_sum_expBuf_64 = expBuf[lid];
    float neighborVal_sum_expBuf_64 = expBuf[neighborIdx_sum_expBuf_64];
    float sumVal_sum_expBuf_64 = (myVal_sum_expBuf_64 + neighborVal_sum_expBuf_64);
    expBuf[lid] = sumVal_sum_expBuf_64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_32 = (lid < (uint)(32u));
  if (shouldReduce_sum_expBuf_32) {
    uint neighborIdx_sum_expBuf_32 = (lid + (uint)(32u));
    float myVal_sum_expBuf_32 = expBuf[lid];
    float neighborVal_sum_expBuf_32 = expBuf[neighborIdx_sum_expBuf_32];
    float sumVal_sum_expBuf_32 = (myVal_sum_expBuf_32 + neighborVal_sum_expBuf_32);
    expBuf[lid] = sumVal_sum_expBuf_32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_16 = (lid < (uint)(16u));
  if (shouldReduce_sum_expBuf_16) {
    uint neighborIdx_sum_expBuf_16 = (lid + (uint)(16u));
    float myVal_sum_expBuf_16 = expBuf[lid];
    float neighborVal_sum_expBuf_16 = expBuf[neighborIdx_sum_expBuf_16];
    float sumVal_sum_expBuf_16 = (myVal_sum_expBuf_16 + neighborVal_sum_expBuf_16);
    expBuf[lid] = sumVal_sum_expBuf_16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_8 = (lid < (uint)(8u));
  if (shouldReduce_sum_expBuf_8) {
    uint neighborIdx_sum_expBuf_8 = (lid + (uint)(8u));
    float myVal_sum_expBuf_8 = expBuf[lid];
    float neighborVal_sum_expBuf_8 = expBuf[neighborIdx_sum_expBuf_8];
    float sumVal_sum_expBuf_8 = (myVal_sum_expBuf_8 + neighborVal_sum_expBuf_8);
    expBuf[lid] = sumVal_sum_expBuf_8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_4 = (lid < (uint)(4u));
  if (shouldReduce_sum_expBuf_4) {
    uint neighborIdx_sum_expBuf_4 = (lid + (uint)(4u));
    float myVal_sum_expBuf_4 = expBuf[lid];
    float neighborVal_sum_expBuf_4 = expBuf[neighborIdx_sum_expBuf_4];
    float sumVal_sum_expBuf_4 = (myVal_sum_expBuf_4 + neighborVal_sum_expBuf_4);
    expBuf[lid] = sumVal_sum_expBuf_4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_2 = (lid < (uint)(2u));
  if (shouldReduce_sum_expBuf_2) {
    uint neighborIdx_sum_expBuf_2 = (lid + (uint)(2u));
    float myVal_sum_expBuf_2 = expBuf[lid];
    float neighborVal_sum_expBuf_2 = expBuf[neighborIdx_sum_expBuf_2];
    float sumVal_sum_expBuf_2 = (myVal_sum_expBuf_2 + neighborVal_sum_expBuf_2);
    expBuf[lid] = sumVal_sum_expBuf_2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  bool shouldReduce_sum_expBuf_1 = (lid < (uint)(1u));
  if (shouldReduce_sum_expBuf_1) {
    uint neighborIdx_sum_expBuf_1 = (lid + (uint)(1u));
    float myVal_sum_expBuf_1 = expBuf[lid];
    float neighborVal_sum_expBuf_1 = expBuf[neighborIdx_sum_expBuf_1];
    float sumVal_sum_expBuf_1 = (myVal_sum_expBuf_1 + neighborVal_sum_expBuf_1);
    expBuf[lid] = sumVal_sum_expBuf_1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sumExp = expBuf[(uint)(0u)];
  float prob = (expLogit / sumExp);
  logitsBuf[lid] = prob;
  int lidI32_2 = ((int)(lid));
  indexBuf[lid] = lidI32_2;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int kIter = 0; kIter < (int)(k); ++kIter) {
    if ((lid == (uint)(0u))) {
      int bestIdx = 0;
      float bestVal = as_float(0xf149f2cau);
      for (int scanIdx = 0; scanIdx < (int)(ne); ++scanIdx) {
        float scanVal = logitsBuf[scanIdx];
        bool isBetter = (scanVal > bestVal);
        if (isBetter) {
          bestVal = scanVal;
          int scanIdxI32 = ((int)(scanIdx));
          bestIdx = scanIdxI32;
        }
      }
      uint kIterU32 = kIter;
      uint outIdx = ((tokenIdx * k) + kIterU32);
      int bestIdxFinal = bestIdx;
      float bestValFinal = bestVal;
      half bestValF16 = ((half)(bestValFinal));
      expertIndices[outIdx] = bestIdxFinal;
      expertWeights[outIdx] = bestValF16;
      uint bestIdxU32 = ((uint)(bestIdxFinal));
      logitsBuf[bestIdxU32] = as_float(0xf149f2cau);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

)kernel";
const char* k_opencl_navatala_transformer_unpermute_tokens_f32 = R"kernel(
__kernel void navatala_transformer_unpermute_tokens_f32(__global const float* permutedStates, __global const int* tokenIndices, __global const uint* numTokens, __global const uint* hiddenSize, __global float* hiddenStates) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint permutedIdx = ((uint)((int)(get_group_id(0))));
  uint nt = numTokens[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    uint origIdx = ((uint)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + ((uint)(256u) * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((permutedIdx * hs) + hIdx);
          uint dstIdx = ((origIdx * hs) + hIdx);
          float val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_unpermute_tokens_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_unpermute_tokens_f16(__global const half* permutedStates, __global const int* tokenIndices, __global const uint* numTokens, __global const uint* hiddenSize, __global half* hiddenStates) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint permutedIdx = ((uint)((int)(get_group_id(0))));
  uint nt = numTokens[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    uint origIdx = ((uint)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + ((uint)(256u) * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((permutedIdx * hs) + hIdx);
          uint dstIdx = ((origIdx * hs) + hIdx);
          half val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_pad_sequence_f32 = R"kernel(
__kernel void navatala_transformer_pad_sequence_f32(__global const float* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_pad_sequence_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_pad_sequence_f16(__global const half* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = (half)(0.000000f);
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_unpad_sequence_f32 = R"kernel(
__kernel void navatala_transformer_unpad_sequence_f32(__global const float* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global const uint* totalTokens, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint tt = totalTokens[(uint)(0u)];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = (uint)(0u);
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + (uint)(1u));
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_unpad_sequence_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_unpad_sequence_f16(__global const half* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global const uint* totalTokens, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint tt = totalTokens[(uint)(0u)];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = (uint)(0u);
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + (uint)(1u));
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    half val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_swizzle_layout_f32 = R"kernel(
__kernel void navatala_transformer_swizzle_layout_f32(__global const float* _input, __global const uint* rows, __global const uint* cols, __global const uint* tileSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint r = rows[(uint)(0u)];
  uint c = cols[(uint)(0u)];
  uint ts = tileSize[(uint)(0u)];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    float val = _input[gid];
    uint tileRow = (row / ts);
    uint tileCol = (col / ts);
    uint localRow = (row % ts);
    uint localCol = (col % ts);
    uint tilesPerRow = ((c + (ts - (uint)(1u))) / ts);
    uint tileIdx = ((tileRow * tilesPerRow) + tileCol);
    uint tileElements = (ts * ts);
    uint localIdx = ((localRow * ts) + localCol);
    uint outIdx = ((tileIdx * tileElements) + localIdx);
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_swizzle_layout_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_swizzle_layout_f16(__global const half* _input, __global const uint* rows, __global const uint* cols, __global const uint* tileSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint r = rows[(uint)(0u)];
  uint c = cols[(uint)(0u)];
  uint ts = tileSize[(uint)(0u)];
  uint total = (r * c);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / c);
    uint col = (gid % c);
    half val = _input[gid];
    uint tileRow = (row / ts);
    uint tileCol = (col / ts);
    uint localRow = (row % ts);
    uint localCol = (col % ts);
    uint tilesPerRow = ((c + (ts - (uint)(1u))) / ts);
    uint tileIdx = ((tileRow * tilesPerRow) + tileCol);
    uint tileElements = (ts * ts);
    uint localIdx = ((localRow * ts) + localCol);
    uint outIdx = ((tileIdx * tileElements) + localIdx);
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_hadamard_transform_f32 = R"kernel(
__kernel void navatala_transformer_hadamard_transform_f32(__global const float* _input, __global const uint* batchSize, __global const uint* vectorSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint vs = vectorSize[(uint)(0u)];
  __local float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  uint globalIdx = ((batchIdx * vs) + lid);
  if ((batchValid && threadValid)) {
    float val = _input[globalIdx];
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride1 = (uint)(1u);
  uint partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == (uint)(0u));
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal1;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride2 = (uint)(2u);
  uint partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == (uint)(0u));
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal2;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride3 = (uint)(4u);
  uint partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == (uint)(0u));
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal3;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride4 = (uint)(8u);
  uint partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == (uint)(0u));
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal4;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride5 = (uint)(16u);
  uint partner5 = (lid ^ stride5);
  float myVal5 = sharedBuf[lid];
  float partnerVal5 = sharedBuf[partner5];
  bool lowBit5 = ((lid & stride5) == (uint)(0u));
  float newVal5 = ((lowBit5) ? ((myVal5 + partnerVal5)) : ((partnerVal5 - myVal5)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal5;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride6 = (uint)(32u);
  uint partner6 = (lid ^ stride6);
  float myVal6 = sharedBuf[lid];
  float partnerVal6 = sharedBuf[partner6];
  bool lowBit6 = ((lid & stride6) == (uint)(0u));
  float newVal6 = ((lowBit6) ? ((myVal6 + partnerVal6)) : ((partnerVal6 - myVal6)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal6;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride7 = (uint)(64u);
  uint partner7 = (lid ^ stride7);
  float myVal7 = sharedBuf[lid];
  float partnerVal7 = sharedBuf[partner7];
  bool lowBit7 = ((lid & stride7) == (uint)(0u));
  float newVal7 = ((lowBit7) ? ((myVal7 + partnerVal7)) : ((partnerVal7 - myVal7)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal7;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride8 = (uint)(128u);
  uint partner8 = (lid ^ stride8);
  float myVal8 = sharedBuf[lid];
  float partnerVal8 = sharedBuf[partner8];
  bool lowBit8 = ((lid & stride8) == (uint)(0u));
  float newVal8 = ((lowBit8) ? ((myVal8 + partnerVal8)) : ((partnerVal8 - myVal8)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal8 / scale);
  if ((batchValid && threadValid)) {
    _output[globalIdx] = normalized;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_hadamard_transform_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_hadamard_transform_f16(__global const half* _input, __global const uint* batchSize, __global const uint* vectorSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint vs = vectorSize[(uint)(0u)];
  __local float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  uint globalIdx = ((batchIdx * vs) + lid);
  if ((batchValid && threadValid)) {
    half valF16 = _input[globalIdx];
    float val = ((float)(valF16));
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride1 = (uint)(1u);
  uint partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == (uint)(0u));
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal1;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride2 = (uint)(2u);
  uint partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == (uint)(0u));
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal2;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride3 = (uint)(4u);
  uint partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == (uint)(0u));
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal3;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride4 = (uint)(8u);
  uint partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == (uint)(0u));
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal4 / scale);
  half resultF16 = ((half)(normalized));
  if ((batchValid && threadValid)) {
    _output[globalIdx] = resultF16;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_permute_axes_f32 = R"kernel(
__kernel void navatala_transformer_permute_axes_f32(__global const float* _input, __global const uint* dim0, __global const uint* dim1, __global const uint* dim2, __global const uint* dim3, __global const uint* perm0, __global const uint* perm1, __global const uint* perm2, __global const uint* perm3, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d0 = dim0[(uint)(0u)];
  uint d1 = dim1[(uint)(0u)];
  uint d2 = dim2[(uint)(0u)];
  uint d3 = dim3[(uint)(0u)];
  uint p0 = perm0[(uint)(0u)];
  uint p1 = perm1[(uint)(0u)];
  uint p2 = perm2[(uint)(0u)];
  uint p3 = perm3[(uint)(0u)];
  uint total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride0 = (d1 * (d2 * d3));
    uint stride1 = (d2 * d3);
    uint stride2 = d3;
    uint stride3 = (uint)(1u);
    uint i0 = (gid / stride0);
    uint rem0 = (gid % stride0);
    uint i1 = (rem0 / stride1);
    uint rem1 = (rem0 % stride1);
    uint i2 = (rem1 / stride2);
    uint i3 = (rem1 % stride2);
    float val = _input[gid];
    uint od0 = (((p0 == (uint)(0u))) ? (d0) : ((((p0 == (uint)(1u))) ? (d1) : ((((p0 == (uint)(2u))) ? (d2) : (d3))))));
    uint od1 = (((p1 == (uint)(0u))) ? (d0) : ((((p1 == (uint)(1u))) ? (d1) : ((((p1 == (uint)(2u))) ? (d2) : (d3))))));
    uint od2 = (((p2 == (uint)(0u))) ? (d0) : ((((p2 == (uint)(1u))) ? (d1) : ((((p2 == (uint)(2u))) ? (d2) : (d3))))));
    uint od3 = (((p3 == (uint)(0u))) ? (d0) : ((((p3 == (uint)(1u))) ? (d1) : ((((p3 == (uint)(2u))) ? (d2) : (d3))))));
    uint oi0 = (((p0 == (uint)(0u))) ? (i0) : ((((p0 == (uint)(1u))) ? (i1) : ((((p0 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi1 = (((p1 == (uint)(0u))) ? (i0) : ((((p1 == (uint)(1u))) ? (i1) : ((((p1 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi2 = (((p2 == (uint)(0u))) ? (i0) : ((((p2 == (uint)(1u))) ? (i1) : ((((p2 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi3 = (((p3 == (uint)(0u))) ? (i0) : ((((p3 == (uint)(1u))) ? (i1) : ((((p3 == (uint)(2u))) ? (i2) : (i3))))));
    uint ostride0 = (od1 * (od2 * od3));
    uint ostride1 = (od2 * od3);
    uint ostride2 = od3;
    uint outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_permute_axes_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_permute_axes_f16(__global const half* _input, __global const uint* dim0, __global const uint* dim1, __global const uint* dim2, __global const uint* dim3, __global const uint* perm0, __global const uint* perm1, __global const uint* perm2, __global const uint* perm3, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d0 = dim0[(uint)(0u)];
  uint d1 = dim1[(uint)(0u)];
  uint d2 = dim2[(uint)(0u)];
  uint d3 = dim3[(uint)(0u)];
  uint p0 = perm0[(uint)(0u)];
  uint p1 = perm1[(uint)(0u)];
  uint p2 = perm2[(uint)(0u)];
  uint p3 = perm3[(uint)(0u)];
  uint total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    uint stride0 = (d1 * (d2 * d3));
    uint stride1 = (d2 * d3);
    uint stride2 = d3;
    uint i0 = (gid / stride0);
    uint rem0 = (gid % stride0);
    uint i1 = (rem0 / stride1);
    uint rem1 = (rem0 % stride1);
    uint i2 = (rem1 / stride2);
    uint i3 = (rem1 % stride2);
    half val = _input[gid];
    uint od0 = (((p0 == (uint)(0u))) ? (d0) : ((((p0 == (uint)(1u))) ? (d1) : ((((p0 == (uint)(2u))) ? (d2) : (d3))))));
    uint od1 = (((p1 == (uint)(0u))) ? (d0) : ((((p1 == (uint)(1u))) ? (d1) : ((((p1 == (uint)(2u))) ? (d2) : (d3))))));
    uint od2 = (((p2 == (uint)(0u))) ? (d0) : ((((p2 == (uint)(1u))) ? (d1) : ((((p2 == (uint)(2u))) ? (d2) : (d3))))));
    uint od3 = (((p3 == (uint)(0u))) ? (d0) : ((((p3 == (uint)(1u))) ? (d1) : ((((p3 == (uint)(2u))) ? (d2) : (d3))))));
    uint oi0 = (((p0 == (uint)(0u))) ? (i0) : ((((p0 == (uint)(1u))) ? (i1) : ((((p0 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi1 = (((p1 == (uint)(0u))) ? (i0) : ((((p1 == (uint)(1u))) ? (i1) : ((((p1 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi2 = (((p2 == (uint)(0u))) ? (i0) : ((((p2 == (uint)(1u))) ? (i1) : ((((p2 == (uint)(2u))) ? (i2) : (i3))))));
    uint oi3 = (((p3 == (uint)(0u))) ? (i0) : ((((p3 == (uint)(1u))) ? (i1) : ((((p3 == (uint)(2u))) ? (i2) : (i3))))));
    uint ostride0 = (od1 * (od2 * od3));
    uint ostride1 = (od2 * od3);
    uint ostride2 = od3;
    uint outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_pack_sequences_f32 = R"kernel(
__kernel void navatala_transformer_pack_sequences_f32(__global const float* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global const uint* totalTokens, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint tt = totalTokens[(uint)(0u)];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = (uint)(0u);
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + (uint)(1u));
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    float val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_pack_sequences_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_pack_sequences_f16(__global const half* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global const uint* totalTokens, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint tt = totalTokens[(uint)(0u)];
  uint totalOut = (tt * hs);
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint tokenIdx = (gid / hs);
    uint hiddenIdx = (gid % hs);
    uint foundBatchIdx = (uint)(0u);
    for (int b = 0; b < (int)(bs); ++b) {
      uint currOffset = seqOffsets[b];
      uint nextB = (b + (uint)(1u));
      uint nextOffset = (((nextB < bs)) ? (seqOffsets[nextB]) : (tt));
      bool inRange = ((tokenIdx >= currOffset) && (tokenIdx < nextOffset));
      if (inRange) {
        foundBatchIdx = b;
      }
    }
    uint batchIdx = foundBatchIdx;
    uint batchOffset = seqOffsets[batchIdx];
    uint localTokenIdx = (tokenIdx - batchOffset);
    uint batchStride = (msl * hs);
    uint inputIdx = ((batchIdx * batchStride) + ((localTokenIdx * hs) + hiddenIdx));
    half val = _input[inputIdx];
    _output[gid] = val;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_unpack_sequences_f32 = R"kernel(
__kernel void navatala_transformer_unpack_sequences_f32(__global const float* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      float val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_unpack_sequences_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_unpack_sequences_f16(__global const half* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = (half)(0.000000f);
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_generate_position_ids_f32 = R"kernel(
__kernel void navatala_transformer_generate_position_ids_f32(__global const uint* batchSize, __global const uint* seqLen, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint total = (bs * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid % sl);
    float posF32 = ((float)(pos));
    _output[gid] = posF32;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_generate_position_ids_u32 = R"kernel(
__kernel void navatala_transformer_generate_position_ids_u32(__global const uint* batchSize, __global const uint* seqLen, __global uint* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint sl = seqLen[(uint)(0u)];
  uint total = (bs * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid % sl);
    _output[gid] = pos;
  }
}

)kernel";
const char* k_opencl_navatala_transformer_generate_causal_mask_f32 = R"kernel(
__kernel void navatala_transformer_generate_causal_mask_f32(__global const uint* seqLen, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint sl = seqLen[(uint)(0u)];
  uint total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / sl);
    uint col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = as_float(0x00000000u);
    } else {
      _output[gid] = as_float(0xce6e6b28u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_generate_causal_mask_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_generate_causal_mask_f16(__global const uint* seqLen, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint sl = seqLen[(uint)(0u)];
  uint total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / sl);
    uint col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = (half)(0.000000f);
    } else {
      _output[gid] = (half)(-65000.000000f);
    }
  }
}

)kernel";
const char* k_opencl_navatala_transformer_precompute_rotary_emb_f32 = R"kernel(
__kernel void navatala_transformer_precompute_rotary_emb_f32(__global const uint* maxSeqLen, __global const uint* headDim, __global const float* base, __global float* cosOutput, __global float* sinOutput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint msl = maxSeqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid / halfHd);
    uint dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
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
const char* k_opencl_navatala_transformer_precompute_rotary_emb_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_precompute_rotary_emb_f16(__global const uint* maxSeqLen, __global const uint* headDim, __global const float* base, __global half* cosOutput, __global half* sinOutput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint msl = maxSeqLen[(uint)(0u)];
  uint hd = headDim[(uint)(0u)];
  float ropeBase = base[(uint)(0u)];
  uint halfHd = (hd / (uint)(2u));
  uint total = (msl * halfHd);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint pos = (gid / halfHd);
    uint dimPair = (gid % halfHd);
    float posF32 = ((float)(pos));
    float dimPairF32 = ((float)(dimPair));
    float hdF32 = ((float)(hd));
    float twoD = (as_float(0x40000000u) * dimPairF32);
    float exponent = (twoD / hdF32);
    float logBase = log(ropeBase);
    float expArg = (exponent * logBase);
    float freqScale = exp(expArg);
    float theta = (posF32 / freqScale);
    float cosValF32 = cos(theta);
    float sinValF32 = sin(theta);
    half cosVal = ((half)(cosValF32));
    half sinVal = ((half)(sinValF32));
    cosOutput[gid] = cosVal;
    sinOutput[gid] = sinVal;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_gelu_f32 = {
  1,
  "navatala_transformer_gelu_f32",
  "opencl",
  "navatala_transformer_gelu_f32",
  "kernel:opencl:navatala_transformer_gelu_f32",
  "abi-r1:opencl:navatala_transformer_gelu_f32",
  "abi-r1:opencl:navatala_transformer_gelu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_gelu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_gelu_tanh_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_gelu_tanh_f32 = {
  1,
  "navatala_transformer_gelu_tanh_f32",
  "opencl",
  "navatala_transformer_gelu_tanh_f32",
  "kernel:opencl:navatala_transformer_gelu_tanh_f32",
  "abi-r1:opencl:navatala_transformer_gelu_tanh_f32",
  "abi-r1:opencl:navatala_transformer_gelu_tanh_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_gelu_tanh_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_silu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_silu_f32 = {
  1,
  "navatala_transformer_silu_f32",
  "opencl",
  "navatala_transformer_silu_f32",
  "kernel:opencl:navatala_transformer_silu_f32",
  "abi-r1:opencl:navatala_transformer_silu_f32",
  "abi-r1:opencl:navatala_transformer_silu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_silu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_relu_f32 = {
  1,
  "navatala_transformer_relu_f32",
  "opencl",
  "navatala_transformer_relu_f32",
  "kernel:opencl:navatala_transformer_relu_f32",
  "abi-r1:opencl:navatala_transformer_relu_f32",
  "abi-r1:opencl:navatala_transformer_relu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_relu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_quick_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_quick_gelu_f32 = {
  1,
  "navatala_transformer_quick_gelu_f32",
  "opencl",
  "navatala_transformer_quick_gelu_f32",
  "kernel:opencl:navatala_transformer_quick_gelu_f32",
  "abi-r1:opencl:navatala_transformer_quick_gelu_f32",
  "abi-r1:opencl:navatala_transformer_quick_gelu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_quick_gelu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_squared_relu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_squared_relu_f32 = {
  1,
  "navatala_transformer_squared_relu_f32",
  "opencl",
  "navatala_transformer_squared_relu_f32",
  "kernel:opencl:navatala_transformer_squared_relu_f32",
  "abi-r1:opencl:navatala_transformer_squared_relu_f32",
  "abi-r1:opencl:navatala_transformer_squared_relu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_squared_relu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_bias_gelu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bias", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "biasSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_bias_gelu_f32 = {
  1,
  "navatala_transformer_bias_gelu_f32",
  "opencl",
  "navatala_transformer_bias_gelu_f32",
  "kernel:opencl:navatala_transformer_bias_gelu_f32",
  "abi-r1:opencl:navatala_transformer_bias_gelu_f32",
  "abi-r1:opencl:navatala_transformer_bias_gelu_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_bias_gelu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_gelu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_gelu_f16 = {
  1,
  "navatala_transformer_gelu_f16",
  "opencl",
  "navatala_transformer_gelu_f16",
  "kernel:opencl:navatala_transformer_gelu_f16",
  "abi-r1:opencl:navatala_transformer_gelu_f16",
  "abi-r1:opencl:navatala_transformer_gelu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_gelu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_silu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_silu_f16 = {
  1,
  "navatala_transformer_silu_f16",
  "opencl",
  "navatala_transformer_silu_f16",
  "kernel:opencl:navatala_transformer_silu_f16",
  "abi-r1:opencl:navatala_transformer_silu_f16",
  "abi-r1:opencl:navatala_transformer_silu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_silu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_relu_f16 = {
  1,
  "navatala_transformer_relu_f16",
  "opencl",
  "navatala_transformer_relu_f16",
  "kernel:opencl:navatala_transformer_relu_f16",
  "abi-r1:opencl:navatala_transformer_relu_f16",
  "abi-r1:opencl:navatala_transformer_relu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_relu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_quick_gelu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_quick_gelu_f16 = {
  1,
  "navatala_transformer_quick_gelu_f16",
  "opencl",
  "navatala_transformer_quick_gelu_f16",
  "kernel:opencl:navatala_transformer_quick_gelu_f16",
  "abi-r1:opencl:navatala_transformer_quick_gelu_f16",
  "abi-r1:opencl:navatala_transformer_quick_gelu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_quick_gelu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_squared_relu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_squared_relu_f16 = {
  1,
  "navatala_transformer_squared_relu_f16",
  "opencl",
  "navatala_transformer_squared_relu_f16",
  "kernel:opencl:navatala_transformer_squared_relu_f16",
  "abi-r1:opencl:navatala_transformer_squared_relu_f16",
  "abi-r1:opencl:navatala_transformer_squared_relu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_squared_relu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_geglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_geglu_f32 = {
  1,
  "navatala_transformer_geglu_f32",
  "opencl",
  "navatala_transformer_geglu_f32",
  "kernel:opencl:navatala_transformer_geglu_f32",
  "abi-r1:opencl:navatala_transformer_geglu_f32",
  "abi-r1:opencl:navatala_transformer_geglu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_geglu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_swiglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_swiglu_f32 = {
  1,
  "navatala_transformer_swiglu_f32",
  "opencl",
  "navatala_transformer_swiglu_f32",
  "kernel:opencl:navatala_transformer_swiglu_f32",
  "abi-r1:opencl:navatala_transformer_swiglu_f32",
  "abi-r1:opencl:navatala_transformer_swiglu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_swiglu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_reglu_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_reglu_f32 = {
  1,
  "navatala_transformer_reglu_f32",
  "opencl",
  "navatala_transformer_reglu_f32",
  "kernel:opencl:navatala_transformer_reglu_f32",
  "abi-r1:opencl:navatala_transformer_reglu_f32",
  "abi-r1:opencl:navatala_transformer_reglu_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_reglu_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_geglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_geglu_f16 = {
  1,
  "navatala_transformer_geglu_f16",
  "opencl",
  "navatala_transformer_geglu_f16",
  "kernel:opencl:navatala_transformer_geglu_f16",
  "abi-r1:opencl:navatala_transformer_geglu_f16",
  "abi-r1:opencl:navatala_transformer_geglu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_geglu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_swiglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_swiglu_f16 = {
  1,
  "navatala_transformer_swiglu_f16",
  "opencl",
  "navatala_transformer_swiglu_f16",
  "kernel:opencl:navatala_transformer_swiglu_f16",
  "abi-r1:opencl:navatala_transformer_swiglu_f16",
  "abi-r1:opencl:navatala_transformer_swiglu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_swiglu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_reglu_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_reglu_f16 = {
  1,
  "navatala_transformer_reglu_f16",
  "opencl",
  "navatala_transformer_reglu_f16",
  "kernel:opencl:navatala_transformer_reglu_f16",
  "abi-r1:opencl:navatala_transformer_reglu_f16",
  "abi-r1:opencl:navatala_transformer_reglu_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_reglu_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_gelu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_gelu_backward_f32 = {
  1,
  "navatala_transformer_gelu_backward_f32",
  "opencl",
  "navatala_transformer_gelu_backward_f32",
  "kernel:opencl:navatala_transformer_gelu_backward_f32",
  "abi-r1:opencl:navatala_transformer_gelu_backward_f32",
  "abi-r1:opencl:navatala_transformer_gelu_backward_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_gelu_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_silu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_silu_backward_f32 = {
  1,
  "navatala_transformer_silu_backward_f32",
  "opencl",
  "navatala_transformer_silu_backward_f32",
  "kernel:opencl:navatala_transformer_silu_backward_f32",
  "abi-r1:opencl:navatala_transformer_silu_backward_f32",
  "abi-r1:opencl:navatala_transformer_silu_backward_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_silu_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_relu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_relu_backward_f32 = {
  1,
  "navatala_transformer_relu_backward_f32",
  "opencl",
  "navatala_transformer_relu_backward_f32",
  "kernel:opencl:navatala_transformer_relu_backward_f32",
  "abi-r1:opencl:navatala_transformer_relu_backward_f32",
  "abi-r1:opencl:navatala_transformer_relu_backward_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_relu_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_quick_gelu_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_quick_gelu_backward_f32 = {
  1,
  "navatala_transformer_quick_gelu_backward_f32",
  "opencl",
  "navatala_transformer_quick_gelu_backward_f32",
  "kernel:opencl:navatala_transformer_quick_gelu_backward_f32",
  "abi-r1:opencl:navatala_transformer_quick_gelu_backward_f32",
  "abi-r1:opencl:navatala_transformer_quick_gelu_backward_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_quick_gelu_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_gelu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_gelu_backward_f16 = {
  1,
  "navatala_transformer_gelu_backward_f16",
  "opencl",
  "navatala_transformer_gelu_backward_f16",
  "kernel:opencl:navatala_transformer_gelu_backward_f16",
  "abi-r1:opencl:navatala_transformer_gelu_backward_f16",
  "abi-r1:opencl:navatala_transformer_gelu_backward_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_gelu_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_silu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_silu_backward_f16 = {
  1,
  "navatala_transformer_silu_backward_f16",
  "opencl",
  "navatala_transformer_silu_backward_f16",
  "kernel:opencl:navatala_transformer_silu_backward_f16",
  "abi-r1:opencl:navatala_transformer_silu_backward_f16",
  "abi-r1:opencl:navatala_transformer_silu_backward_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_silu_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_relu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_relu_backward_f16 = {
  1,
  "navatala_transformer_relu_backward_f16",
  "opencl",
  "navatala_transformer_relu_backward_f16",
  "kernel:opencl:navatala_transformer_relu_backward_f16",
  "abi-r1:opencl:navatala_transformer_relu_backward_f16",
  "abi-r1:opencl:navatala_transformer_relu_backward_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_relu_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_quick_gelu_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_quick_gelu_backward_f16 = {
  1,
  "navatala_transformer_quick_gelu_backward_f16",
  "opencl",
  "navatala_transformer_quick_gelu_backward_f16",
  "kernel:opencl:navatala_transformer_quick_gelu_backward_f16",
  "abi-r1:opencl:navatala_transformer_quick_gelu_backward_f16",
  "abi-r1:opencl:navatala_transformer_quick_gelu_backward_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_quick_gelu_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rms_norm_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rms_norm_forward_f32 = {
  1,
  "navatala_transformer_rms_norm_forward_f32",
  "opencl",
  "navatala_transformer_rms_norm_forward_f32",
  "kernel:opencl:navatala_transformer_rms_norm_forward_f32",
  "abi-r1:opencl:navatala_transformer_rms_norm_forward_f32",
  "abi-r1:opencl:navatala_transformer_rms_norm_forward_f32",
  6,
  kAbiArgs_opencl_navatala_transformer_rms_norm_forward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rms_norm_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rms_norm_forward_f16 = {
  1,
  "navatala_transformer_rms_norm_forward_f16",
  "opencl",
  "navatala_transformer_rms_norm_forward_f16",
  "kernel:opencl:navatala_transformer_rms_norm_forward_f16",
  "abi-r1:opencl:navatala_transformer_rms_norm_forward_f16",
  "abi-r1:opencl:navatala_transformer_rms_norm_forward_f16",
  6,
  kAbiArgs_opencl_navatala_transformer_rms_norm_forward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_layer_norm_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_layer_norm_forward_f32 = {
  1,
  "navatala_transformer_layer_norm_forward_f32",
  "opencl",
  "navatala_transformer_layer_norm_forward_f32",
  "kernel:opencl:navatala_transformer_layer_norm_forward_f32",
  "abi-r1:opencl:navatala_transformer_layer_norm_forward_f32",
  "abi-r1:opencl:navatala_transformer_layer_norm_forward_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_layer_norm_forward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_layer_norm_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_layer_norm_forward_f16 = {
  1,
  "navatala_transformer_layer_norm_forward_f16",
  "opencl",
  "navatala_transformer_layer_norm_forward_f16",
  "kernel:opencl:navatala_transformer_layer_norm_forward_f16",
  "abi-r1:opencl:navatala_transformer_layer_norm_forward_f16",
  "abi-r1:opencl:navatala_transformer_layer_norm_forward_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_layer_norm_forward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_layer_norm_backward_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_layer_norm_backward_f32 = {
  1,
  "navatala_transformer_layer_norm_backward_f32",
  "opencl",
  "navatala_transformer_layer_norm_backward_f32",
  "kernel:opencl:navatala_transformer_layer_norm_backward_f32",
  "abi-r1:opencl:navatala_transformer_layer_norm_backward_f32",
  "abi-r1:opencl:navatala_transformer_layer_norm_backward_f32",
  10,
  kAbiArgs_opencl_navatala_transformer_layer_norm_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_layer_norm_backward_f16[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_layer_norm_backward_f16 = {
  1,
  "navatala_transformer_layer_norm_backward_f16",
  "opencl",
  "navatala_transformer_layer_norm_backward_f16",
  "kernel:opencl:navatala_transformer_layer_norm_backward_f16",
  "abi-r1:opencl:navatala_transformer_layer_norm_backward_f16",
  "abi-r1:opencl:navatala_transformer_layer_norm_backward_f16",
  10,
  kAbiArgs_opencl_navatala_transformer_layer_norm_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rms_norm_backward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rms_norm_backward_f32 = {
  1,
  "navatala_transformer_rms_norm_backward_f32",
  "opencl",
  "navatala_transformer_rms_norm_backward_f32",
  "kernel:opencl:navatala_transformer_rms_norm_backward_f32",
  "abi-r1:opencl:navatala_transformer_rms_norm_backward_f32",
  "abi-r1:opencl:navatala_transformer_rms_norm_backward_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_rms_norm_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rms_norm_backward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rsigma", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradGamma", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rms_norm_backward_f16 = {
  1,
  "navatala_transformer_rms_norm_backward_f16",
  "opencl",
  "navatala_transformer_rms_norm_backward_f16",
  "kernel:opencl:navatala_transformer_rms_norm_backward_f16",
  "abi-r1:opencl:navatala_transformer_rms_norm_backward_f16",
  "abi-r1:opencl:navatala_transformer_rms_norm_backward_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_rms_norm_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_layer_norm_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_layer_norm_multi_pass_f32 = {
  1,
  "navatala_transformer_layer_norm_multi_pass_f32",
  "opencl",
  "navatala_transformer_layer_norm_multi_pass_f32",
  "kernel:opencl:navatala_transformer_layer_norm_multi_pass_f32",
  "abi-r1:opencl:navatala_transformer_layer_norm_multi_pass_f32",
  "abi-r1:opencl:navatala_transformer_layer_norm_multi_pass_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_layer_norm_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_layer_norm_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_layer_norm_multi_pass_f16 = {
  1,
  "navatala_transformer_layer_norm_multi_pass_f16",
  "opencl",
  "navatala_transformer_layer_norm_multi_pass_f16",
  "kernel:opencl:navatala_transformer_layer_norm_multi_pass_f16",
  "abi-r1:opencl:navatala_transformer_layer_norm_multi_pass_f16",
  "abi-r1:opencl:navatala_transformer_layer_norm_multi_pass_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_layer_norm_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rms_norm_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rms_norm_multi_pass_f32 = {
  1,
  "navatala_transformer_rms_norm_multi_pass_f32",
  "opencl",
  "navatala_transformer_rms_norm_multi_pass_f32",
  "kernel:opencl:navatala_transformer_rms_norm_multi_pass_f32",
  "abi-r1:opencl:navatala_transformer_rms_norm_multi_pass_f32",
  "abi-r1:opencl:navatala_transformer_rms_norm_multi_pass_f32",
  6,
  kAbiArgs_opencl_navatala_transformer_rms_norm_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rms_norm_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gamma", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "epsilon", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rms_norm_multi_pass_f16 = {
  1,
  "navatala_transformer_rms_norm_multi_pass_f16",
  "opencl",
  "navatala_transformer_rms_norm_multi_pass_f16",
  "kernel:opencl:navatala_transformer_rms_norm_multi_pass_f16",
  "abi-r1:opencl:navatala_transformer_rms_norm_multi_pass_f16",
  "abi-r1:opencl:navatala_transformer_rms_norm_multi_pass_f16",
  6,
  kAbiArgs_opencl_navatala_transformer_rms_norm_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_forward_f32 = {
  1,
  "navatala_transformer_softmax_forward_f32",
  "opencl",
  "navatala_transformer_softmax_forward_f32",
  "kernel:opencl:navatala_transformer_softmax_forward_f32",
  "abi-r1:opencl:navatala_transformer_softmax_forward_f32",
  "abi-r1:opencl:navatala_transformer_softmax_forward_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_softmax_forward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_forward_f16 = {
  1,
  "navatala_transformer_softmax_forward_f16",
  "opencl",
  "navatala_transformer_softmax_forward_f16",
  "kernel:opencl:navatala_transformer_softmax_forward_f16",
  "abi-r1:opencl:navatala_transformer_softmax_forward_f16",
  "abi-r1:opencl:navatala_transformer_softmax_forward_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_softmax_forward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_backward_f32[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_backward_f32 = {
  1,
  "navatala_transformer_softmax_backward_f32",
  "opencl",
  "navatala_transformer_softmax_backward_f32",
  "kernel:opencl:navatala_transformer_softmax_backward_f32",
  "abi-r1:opencl:navatala_transformer_softmax_backward_f32",
  "abi-r1:opencl:navatala_transformer_softmax_backward_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_softmax_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_backward_f16[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_backward_f16 = {
  1,
  "navatala_transformer_softmax_backward_f16",
  "opencl",
  "navatala_transformer_softmax_backward_f16",
  "kernel:opencl:navatala_transformer_softmax_backward_f16",
  "abi-r1:opencl:navatala_transformer_softmax_backward_f16",
  "abi-r1:opencl:navatala_transformer_softmax_backward_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_softmax_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_with_mask_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_with_mask_f32 = {
  1,
  "navatala_transformer_softmax_with_mask_f32",
  "opencl",
  "navatala_transformer_softmax_with_mask_f32",
  "kernel:opencl:navatala_transformer_softmax_with_mask_f32",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_f32",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_softmax_with_mask_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_with_mask_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_with_mask_f16 = {
  1,
  "navatala_transformer_softmax_with_mask_f16",
  "opencl",
  "navatala_transformer_softmax_with_mask_f16",
  "kernel:opencl:navatala_transformer_softmax_with_mask_f16",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_f16",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_softmax_with_mask_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_with_mask_backward_f32[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_with_mask_backward_f32 = {
  1,
  "navatala_transformer_softmax_with_mask_backward_f32",
  "opencl",
  "navatala_transformer_softmax_with_mask_backward_f32",
  "kernel:opencl:navatala_transformer_softmax_with_mask_backward_f32",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_backward_f32",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_backward_f32",
  6,
  kAbiArgs_opencl_navatala_transformer_softmax_with_mask_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_with_mask_backward_f16[] = {
  { "output", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOutput", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "isCausal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_with_mask_backward_f16 = {
  1,
  "navatala_transformer_softmax_with_mask_backward_f16",
  "opencl",
  "navatala_transformer_softmax_with_mask_backward_f16",
  "kernel:opencl:navatala_transformer_softmax_with_mask_backward_f16",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_backward_f16",
  "abi-r1:opencl:navatala_transformer_softmax_with_mask_backward_f16",
  6,
  kAbiArgs_opencl_navatala_transformer_softmax_with_mask_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_multi_pass_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_multi_pass_f32 = {
  1,
  "navatala_transformer_softmax_multi_pass_f32",
  "opencl",
  "navatala_transformer_softmax_multi_pass_f32",
  "kernel:opencl:navatala_transformer_softmax_multi_pass_f32",
  "abi-r1:opencl:navatala_transformer_softmax_multi_pass_f32",
  "abi-r1:opencl:navatala_transformer_softmax_multi_pass_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_softmax_multi_pass_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_softmax_multi_pass_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_softmax_multi_pass_f16 = {
  1,
  "navatala_transformer_softmax_multi_pass_f16",
  "opencl",
  "navatala_transformer_softmax_multi_pass_f16",
  "kernel:opencl:navatala_transformer_softmax_multi_pass_f16",
  "abi-r1:opencl:navatala_transformer_softmax_multi_pass_f16",
  "abi-r1:opencl:navatala_transformer_softmax_multi_pass_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_softmax_multi_pass_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_cast_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_cast_f32_to_f16 = {
  1,
  "navatala_transformer_cast_f32_to_f16",
  "opencl",
  "navatala_transformer_cast_f32_to_f16",
  "kernel:opencl:navatala_transformer_cast_f32_to_f16",
  "abi-r1:opencl:navatala_transformer_cast_f32_to_f16",
  "abi-r1:opencl:navatala_transformer_cast_f32_to_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_cast_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_cast_f16_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_cast_f16_to_f32 = {
  1,
  "navatala_transformer_cast_f16_to_f32",
  "opencl",
  "navatala_transformer_cast_f16_to_f32",
  "kernel:opencl:navatala_transformer_cast_f16_to_f32",
  "abi-r1:opencl:navatala_transformer_cast_f16_to_f32",
  "abi-r1:opencl:navatala_transformer_cast_f16_to_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_cast_f16_to_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_transpose_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_transpose_f32 = {
  1,
  "navatala_transformer_transpose_f32",
  "opencl",
  "navatala_transformer_transpose_f32",
  "kernel:opencl:navatala_transformer_transpose_f32",
  "abi-r1:opencl:navatala_transformer_transpose_f32",
  "abi-r1:opencl:navatala_transformer_transpose_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_transpose_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_transpose_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_transpose_f16 = {
  1,
  "navatala_transformer_transpose_f16",
  "opencl",
  "navatala_transformer_transpose_f16",
  "kernel:opencl:navatala_transformer_transpose_f16",
  "abi-r1:opencl:navatala_transformer_transpose_f16",
  "abi-r1:opencl:navatala_transformer_transpose_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_transpose_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_fused_cast_transpose_f32_to_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_fused_cast_transpose_f32_to_f16 = {
  1,
  "navatala_transformer_fused_cast_transpose_f32_to_f16",
  "opencl",
  "navatala_transformer_fused_cast_transpose_f32_to_f16",
  "kernel:opencl:navatala_transformer_fused_cast_transpose_f32_to_f16",
  "abi-r1:opencl:navatala_transformer_fused_cast_transpose_f32_to_f16",
  "abi-r1:opencl:navatala_transformer_fused_cast_transpose_f32_to_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_fused_cast_transpose_f32_to_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_cast_f32_to_f_p8_e4_m3[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_cast_f32_to_f_p8_e4_m3 = {
  1,
  "navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "opencl",
  "navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "kernel:opencl:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "abi-r1:opencl:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  "abi-r1:opencl:navatala_transformer_cast_f32_to_f_p8_e4_m3",
  4,
  kAbiArgs_opencl_navatala_transformer_cast_f32_to_f_p8_e4_m3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_cast_f_p8_e4_m3_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_cast_f_p8_e4_m3_to_f32 = {
  1,
  "navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "opencl",
  "navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "kernel:opencl:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "abi-r1:opencl:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  "abi-r1:opencl:navatala_transformer_cast_f_p8_e4_m3_to_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_cast_f_p8_e4_m3_to_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_cast_f32_to_f_p8_e5_m2[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_cast_f32_to_f_p8_e5_m2 = {
  1,
  "navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "opencl",
  "navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "kernel:opencl:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "abi-r1:opencl:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  "abi-r1:opencl:navatala_transformer_cast_f32_to_f_p8_e5_m2",
  4,
  kAbiArgs_opencl_navatala_transformer_cast_f32_to_f_p8_e5_m2
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_cast_f_p8_e5_m2_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_cast_f_p8_e5_m2_to_f32 = {
  1,
  "navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "opencl",
  "navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "kernel:opencl:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "abi-r1:opencl:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  "abi-r1:opencl:navatala_transformer_cast_f_p8_e5_m2_to_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_cast_f_p8_e5_m2_to_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_fused_cast_transpose_f_p8[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_fused_cast_transpose_f_p8 = {
  1,
  "navatala_transformer_fused_cast_transpose_f_p8",
  "opencl",
  "navatala_transformer_fused_cast_transpose_f_p8",
  "kernel:opencl:navatala_transformer_fused_cast_transpose_f_p8",
  "abi-r1:opencl:navatala_transformer_fused_cast_transpose_f_p8",
  "abi-r1:opencl:navatala_transformer_fused_cast_transpose_f_p8",
  5,
  kAbiArgs_opencl_navatala_transformer_fused_cast_transpose_f_p8
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_transpose4_d_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_transpose4_d_f32 = {
  1,
  "navatala_transformer_transpose4_d_f32",
  "opencl",
  "navatala_transformer_transpose4_d_f32",
  "kernel:opencl:navatala_transformer_transpose4_d_f32",
  "abi-r1:opencl:navatala_transformer_transpose4_d_f32",
  "abi-r1:opencl:navatala_transformer_transpose4_d_f32",
  10,
  kAbiArgs_opencl_navatala_transformer_transpose4_d_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_transpose4_d_f16[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_transpose4_d_f16 = {
  1,
  "navatala_transformer_transpose4_d_f16",
  "opencl",
  "navatala_transformer_transpose4_d_f16",
  "kernel:opencl:navatala_transformer_transpose4_d_f16",
  "abi-r1:opencl:navatala_transformer_transpose4_d_f16",
  "abi-r1:opencl:navatala_transformer_transpose4_d_f16",
  10,
  kAbiArgs_opencl_navatala_transformer_transpose4_d_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_dropout_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_dropout_forward_f32 = {
  1,
  "navatala_transformer_dropout_forward_f32",
  "opencl",
  "navatala_transformer_dropout_forward_f32",
  "kernel:opencl:navatala_transformer_dropout_forward_f32",
  "abi-r1:opencl:navatala_transformer_dropout_forward_f32",
  "abi-r1:opencl:navatala_transformer_dropout_forward_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_dropout_forward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_dropout_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_dropout_forward_f16 = {
  1,
  "navatala_transformer_dropout_forward_f16",
  "opencl",
  "navatala_transformer_dropout_forward_f16",
  "kernel:opencl:navatala_transformer_dropout_forward_f16",
  "abi-r1:opencl:navatala_transformer_dropout_forward_f16",
  "abi-r1:opencl:navatala_transformer_dropout_forward_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_dropout_forward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_dropout_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_dropout_backward_f32 = {
  1,
  "navatala_transformer_dropout_backward_f32",
  "opencl",
  "navatala_transformer_dropout_backward_f32",
  "kernel:opencl:navatala_transformer_dropout_backward_f32",
  "abi-r1:opencl:navatala_transformer_dropout_backward_f32",
  "abi-r1:opencl:navatala_transformer_dropout_backward_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_dropout_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_dropout_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "prob", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_dropout_backward_f16 = {
  1,
  "navatala_transformer_dropout_backward_f16",
  "opencl",
  "navatala_transformer_dropout_backward_f16",
  "kernel:opencl:navatala_transformer_dropout_backward_f16",
  "abi-r1:opencl:navatala_transformer_dropout_backward_f16",
  "abi-r1:opencl:navatala_transformer_dropout_backward_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_dropout_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_forward_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_forward_f32 = {
  1,
  "navatala_transformer_rope_forward_f32",
  "opencl",
  "navatala_transformer_rope_forward_f32",
  "kernel:opencl:navatala_transformer_rope_forward_f32",
  "abi-r1:opencl:navatala_transformer_rope_forward_f32",
  "abi-r1:opencl:navatala_transformer_rope_forward_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_rope_forward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_forward_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_forward_f16 = {
  1,
  "navatala_transformer_rope_forward_f16",
  "opencl",
  "navatala_transformer_rope_forward_f16",
  "kernel:opencl:navatala_transformer_rope_forward_f16",
  "abi-r1:opencl:navatala_transformer_rope_forward_f16",
  "abi-r1:opencl:navatala_transformer_rope_forward_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_rope_forward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_with_cache_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cosCache", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinCache", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_with_cache_f32 = {
  1,
  "navatala_transformer_rope_with_cache_f32",
  "opencl",
  "navatala_transformer_rope_with_cache_f32",
  "kernel:opencl:navatala_transformer_rope_with_cache_f32",
  "abi-r1:opencl:navatala_transformer_rope_with_cache_f32",
  "abi-r1:opencl:navatala_transformer_rope_with_cache_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_rope_with_cache_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_backward_f32[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_backward_f32 = {
  1,
  "navatala_transformer_rope_backward_f32",
  "opencl",
  "navatala_transformer_rope_backward_f32",
  "kernel:opencl:navatala_transformer_rope_backward_f32",
  "abi-r1:opencl:navatala_transformer_rope_backward_f32",
  "abi-r1:opencl:navatala_transformer_rope_backward_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_rope_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_backward_f16[] = {
  { "gradOutput", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradInput", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_backward_f16 = {
  1,
  "navatala_transformer_rope_backward_f16",
  "opencl",
  "navatala_transformer_rope_backward_f16",
  "kernel:opencl:navatala_transformer_rope_backward_f16",
  "abi-r1:opencl:navatala_transformer_rope_backward_f16",
  "abi-r1:opencl:navatala_transformer_rope_backward_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_rope_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_interleaved_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_interleaved_f32 = {
  1,
  "navatala_transformer_rope_interleaved_f32",
  "opencl",
  "navatala_transformer_rope_interleaved_f32",
  "kernel:opencl:navatala_transformer_rope_interleaved_f32",
  "abi-r1:opencl:navatala_transformer_rope_interleaved_f32",
  "abi-r1:opencl:navatala_transformer_rope_interleaved_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_rope_interleaved_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_rope_interleaved_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "positions", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_rope_interleaved_f16 = {
  1,
  "navatala_transformer_rope_interleaved_f16",
  "opencl",
  "navatala_transformer_rope_interleaved_f16",
  "kernel:opencl:navatala_transformer_rope_interleaved_f16",
  "abi-r1:opencl:navatala_transformer_rope_interleaved_f16",
  "abi-r1:opencl:navatala_transformer_rope_interleaved_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_rope_interleaved_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_f16 = {
  1,
  "navatala_transformer_simple_attention_f16",
  "opencl",
  "navatala_transformer_simple_attention_f16",
  "kernel:opencl:navatala_transformer_simple_attention_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_simple_attention_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_f32 = {
  1,
  "navatala_transformer_simple_attention_f32",
  "opencl",
  "navatala_transformer_simple_attention_f32",
  "kernel:opencl:navatala_transformer_simple_attention_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_simple_attention_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_paged_attention_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_paged_attention_f32 = {
  1,
  "navatala_transformer_paged_attention_f32",
  "opencl",
  "navatala_transformer_paged_attention_f32",
  "kernel:opencl:navatala_transformer_paged_attention_f32",
  "abi-r1:opencl:navatala_transformer_paged_attention_f32",
  "abi-r1:opencl:navatala_transformer_paged_attention_f32",
  9,
  kAbiArgs_opencl_navatala_transformer_paged_attention_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_with_mask_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_with_mask_f32 = {
  1,
  "navatala_transformer_simple_attention_with_mask_f32",
  "opencl",
  "navatala_transformer_simple_attention_with_mask_f32",
  "kernel:opencl:navatala_transformer_simple_attention_with_mask_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_mask_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_mask_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_simple_attention_with_mask_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_with_mask_f16[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "key", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_with_mask_f16 = {
  1,
  "navatala_transformer_simple_attention_with_mask_f16",
  "opencl",
  "navatala_transformer_simple_attention_with_mask_f16",
  "kernel:opencl:navatala_transformer_simple_attention_with_mask_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_mask_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_mask_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_simple_attention_with_mask_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_with_padding_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_with_padding_f32 = {
  1,
  "navatala_transformer_simple_attention_with_padding_f32",
  "opencl",
  "navatala_transformer_simple_attention_with_padding_f32",
  "kernel:opencl:navatala_transformer_simple_attention_with_padding_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_padding_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_padding_f32",
  9,
  kAbiArgs_opencl_navatala_transformer_simple_attention_with_padding_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_with_padding_f16[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_with_padding_f16 = {
  1,
  "navatala_transformer_simple_attention_with_padding_f16",
  "opencl",
  "navatala_transformer_simple_attention_with_padding_f16",
  "kernel:opencl:navatala_transformer_simple_attention_with_padding_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_padding_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_with_padding_f16",
  9,
  kAbiArgs_opencl_navatala_transformer_simple_attention_with_padding_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_backward_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_backward_f32 = {
  1,
  "navatala_transformer_simple_attention_backward_f32",
  "opencl",
  "navatala_transformer_simple_attention_backward_f32",
  "kernel:opencl:navatala_transformer_simple_attention_backward_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_backward_f32",
  "abi-r1:opencl:navatala_transformer_simple_attention_backward_f32",
  11,
  kAbiArgs_opencl_navatala_transformer_simple_attention_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_simple_attention_backward_f16[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_simple_attention_backward_f16 = {
  1,
  "navatala_transformer_simple_attention_backward_f16",
  "opencl",
  "navatala_transformer_simple_attention_backward_f16",
  "kernel:opencl:navatala_transformer_simple_attention_backward_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_backward_f16",
  "abi-r1:opencl:navatala_transformer_simple_attention_backward_f16",
  11,
  kAbiArgs_opencl_navatala_transformer_simple_attention_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_kv_cache_append_f16[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_kv_cache_append_f16 = {
  1,
  "navatala_transformer_kv_cache_append_f16",
  "opencl",
  "navatala_transformer_kv_cache_append_f16",
  "kernel:opencl:navatala_transformer_kv_cache_append_f16",
  "abi-r1:opencl:navatala_transformer_kv_cache_append_f16",
  "abi-r1:opencl:navatala_transformer_kv_cache_append_f16",
  9,
  kAbiArgs_opencl_navatala_transformer_kv_cache_append_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_kv_cache_append_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_kv_cache_append_f32 = {
  1,
  "navatala_transformer_kv_cache_append_f32",
  "opencl",
  "navatala_transformer_kv_cache_append_f32",
  "kernel:opencl:navatala_transformer_kv_cache_append_f32",
  "abi-r1:opencl:navatala_transformer_kv_cache_append_f32",
  "abi-r1:opencl:navatala_transformer_kv_cache_append_f32",
  9,
  kAbiArgs_opencl_navatala_transformer_kv_cache_append_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_kv_cache_rotate_f16[] = {
  { "blockTables", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contextLens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "shiftAmount", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numHeads", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pageSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxBlocks", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "kvCache", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_kv_cache_rotate_f16 = {
  1,
  "navatala_transformer_kv_cache_rotate_f16",
  "opencl",
  "navatala_transformer_kv_cache_rotate_f16",
  "kernel:opencl:navatala_transformer_kv_cache_rotate_f16",
  "abi-r1:opencl:navatala_transformer_kv_cache_rotate_f16",
  "abi-r1:opencl:navatala_transformer_kv_cache_rotate_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_kv_cache_rotate_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_tiled_gemm_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_tiled_gemm_f16 = {
  1,
  "navatala_transformer_tiled_gemm_f16",
  "opencl",
  "navatala_transformer_tiled_gemm_f16",
  "kernel:opencl:navatala_transformer_tiled_gemm_f16",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_f16",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_tiled_gemm_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_tiled_gemm_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "beta", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "c", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_tiled_gemm_f32 = {
  1,
  "navatala_transformer_tiled_gemm_f32",
  "opencl",
  "navatala_transformer_tiled_gemm_f32",
  "kernel:opencl:navatala_transformer_tiled_gemm_f32",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_f32",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_tiled_gemm_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_tiled_gemm_backward_f32[] = {
  { "gradC", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradA", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_tiled_gemm_backward_f32 = {
  1,
  "navatala_transformer_tiled_gemm_backward_f32",
  "opencl",
  "navatala_transformer_tiled_gemm_backward_f32",
  "kernel:opencl:navatala_transformer_tiled_gemm_backward_f32",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_backward_f32",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_backward_f32",
  6,
  kAbiArgs_opencl_navatala_transformer_tiled_gemm_backward_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_tiled_gemm_backward_f16[] = {
  { "gradC", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "m", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gradA", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_tiled_gemm_backward_f16 = {
  1,
  "navatala_transformer_tiled_gemm_backward_f16",
  "opencl",
  "navatala_transformer_tiled_gemm_backward_f16",
  "kernel:opencl:navatala_transformer_tiled_gemm_backward_f16",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_backward_f16",
  "abi-r1:opencl:navatala_transformer_tiled_gemm_backward_f16",
  6,
  kAbiArgs_opencl_navatala_transformer_tiled_gemm_backward_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_scale_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_scale_f32 = {
  1,
  "navatala_transformer_multi_tensor_scale_f32",
  "opencl",
  "navatala_transformer_multi_tensor_scale_f32",
  "kernel:opencl:navatala_transformer_multi_tensor_scale_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_scale_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_scale_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_scale_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_scale_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scale", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_scale_f16 = {
  1,
  "navatala_transformer_multi_tensor_scale_f16",
  "opencl",
  "navatala_transformer_multi_tensor_scale_f16",
  "kernel:opencl:navatala_transformer_multi_tensor_scale_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_scale_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_scale_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_scale_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_add_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_add_f32 = {
  1,
  "navatala_transformer_multi_tensor_add_f32",
  "opencl",
  "navatala_transformer_multi_tensor_add_f32",
  "kernel:opencl:navatala_transformer_multi_tensor_add_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_add_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_add_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_add_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_add_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "value", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_add_f16 = {
  1,
  "navatala_transformer_multi_tensor_add_f16",
  "opencl",
  "navatala_transformer_multi_tensor_add_f16",
  "kernel:opencl:navatala_transformer_multi_tensor_add_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_add_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_add_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_add_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_copy_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_copy_f32 = {
  1,
  "navatala_transformer_multi_tensor_copy_f32",
  "opencl",
  "navatala_transformer_multi_tensor_copy_f32",
  "kernel:opencl:navatala_transformer_multi_tensor_copy_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_copy_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_copy_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_copy_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_copy_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_copy_f16 = {
  1,
  "navatala_transformer_multi_tensor_copy_f16",
  "opencl",
  "navatala_transformer_multi_tensor_copy_f16",
  "kernel:opencl:navatala_transformer_multi_tensor_copy_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_copy_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_copy_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_copy_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_l2_norm_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partialSums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_l2_norm_f32 = {
  1,
  "navatala_transformer_multi_tensor_l2_norm_f32",
  "opencl",
  "navatala_transformer_multi_tensor_l2_norm_f32",
  "kernel:opencl:navatala_transformer_multi_tensor_l2_norm_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_l2_norm_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_l2_norm_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_l2_norm_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_l2_norm_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "totalElements", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partialSums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_l2_norm_f16 = {
  1,
  "navatala_transformer_multi_tensor_l2_norm_f16",
  "opencl",
  "navatala_transformer_multi_tensor_l2_norm_f16",
  "kernel:opencl:navatala_transformer_multi_tensor_l2_norm_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_l2_norm_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_l2_norm_f16",
  3,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_l2_norm_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_clip_grad_f32[] = {
  { "totalNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_clip_grad_f32 = {
  1,
  "navatala_transformer_multi_tensor_clip_grad_f32",
  "opencl",
  "navatala_transformer_multi_tensor_clip_grad_f32",
  "kernel:opencl:navatala_transformer_multi_tensor_clip_grad_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_clip_grad_f32",
  "abi-r1:opencl:navatala_transformer_multi_tensor_clip_grad_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_clip_grad_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_multi_tensor_clip_grad_f16[] = {
  { "totalNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalElements", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "grad", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_multi_tensor_clip_grad_f16 = {
  1,
  "navatala_transformer_multi_tensor_clip_grad_f16",
  "opencl",
  "navatala_transformer_multi_tensor_clip_grad_f16",
  "kernel:opencl:navatala_transformer_multi_tensor_clip_grad_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_clip_grad_f16",
  "abi-r1:opencl:navatala_transformer_multi_tensor_clip_grad_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_multi_tensor_clip_grad_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_top_k_gating_f32[] = {
  { "routerLogits", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "expertIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertWeights", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_top_k_gating_f32 = {
  1,
  "navatala_transformer_top_k_gating_f32",
  "opencl",
  "navatala_transformer_top_k_gating_f32",
  "kernel:opencl:navatala_transformer_top_k_gating_f32",
  "abi-r1:opencl:navatala_transformer_top_k_gating_f32",
  "abi-r1:opencl:navatala_transformer_top_k_gating_f32",
  6,
  kAbiArgs_opencl_navatala_transformer_top_k_gating_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_top_k_gating_f16[] = {
  { "routerLogits", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numExperts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "topK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "expertIndices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "expertWeights", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_top_k_gating_f16 = {
  1,
  "navatala_transformer_top_k_gating_f16",
  "opencl",
  "navatala_transformer_top_k_gating_f16",
  "kernel:opencl:navatala_transformer_top_k_gating_f16",
  "abi-r1:opencl:navatala_transformer_top_k_gating_f16",
  "abi-r1:opencl:navatala_transformer_top_k_gating_f16",
  6,
  kAbiArgs_opencl_navatala_transformer_top_k_gating_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_unpermute_tokens_f32[] = {
  { "permutedStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenStates", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_unpermute_tokens_f32 = {
  1,
  "navatala_transformer_unpermute_tokens_f32",
  "opencl",
  "navatala_transformer_unpermute_tokens_f32",
  "kernel:opencl:navatala_transformer_unpermute_tokens_f32",
  "abi-r1:opencl:navatala_transformer_unpermute_tokens_f32",
  "abi-r1:opencl:navatala_transformer_unpermute_tokens_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_unpermute_tokens_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_unpermute_tokens_f16[] = {
  { "permutedStates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tokenIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numTokens", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenStates", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_unpermute_tokens_f16 = {
  1,
  "navatala_transformer_unpermute_tokens_f16",
  "opencl",
  "navatala_transformer_unpermute_tokens_f16",
  "kernel:opencl:navatala_transformer_unpermute_tokens_f16",
  "abi-r1:opencl:navatala_transformer_unpermute_tokens_f16",
  "abi-r1:opencl:navatala_transformer_unpermute_tokens_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_unpermute_tokens_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_pad_sequence_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_pad_sequence_f32 = {
  1,
  "navatala_transformer_pad_sequence_f32",
  "opencl",
  "navatala_transformer_pad_sequence_f32",
  "kernel:opencl:navatala_transformer_pad_sequence_f32",
  "abi-r1:opencl:navatala_transformer_pad_sequence_f32",
  "abi-r1:opencl:navatala_transformer_pad_sequence_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_pad_sequence_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_pad_sequence_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_pad_sequence_f16 = {
  1,
  "navatala_transformer_pad_sequence_f16",
  "opencl",
  "navatala_transformer_pad_sequence_f16",
  "kernel:opencl:navatala_transformer_pad_sequence_f16",
  "abi-r1:opencl:navatala_transformer_pad_sequence_f16",
  "abi-r1:opencl:navatala_transformer_pad_sequence_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_pad_sequence_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_unpad_sequence_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_unpad_sequence_f32 = {
  1,
  "navatala_transformer_unpad_sequence_f32",
  "opencl",
  "navatala_transformer_unpad_sequence_f32",
  "kernel:opencl:navatala_transformer_unpad_sequence_f32",
  "abi-r1:opencl:navatala_transformer_unpad_sequence_f32",
  "abi-r1:opencl:navatala_transformer_unpad_sequence_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_unpad_sequence_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_unpad_sequence_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_unpad_sequence_f16 = {
  1,
  "navatala_transformer_unpad_sequence_f16",
  "opencl",
  "navatala_transformer_unpad_sequence_f16",
  "kernel:opencl:navatala_transformer_unpad_sequence_f16",
  "abi-r1:opencl:navatala_transformer_unpad_sequence_f16",
  "abi-r1:opencl:navatala_transformer_unpad_sequence_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_unpad_sequence_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_swizzle_layout_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tileSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_swizzle_layout_f32 = {
  1,
  "navatala_transformer_swizzle_layout_f32",
  "opencl",
  "navatala_transformer_swizzle_layout_f32",
  "kernel:opencl:navatala_transformer_swizzle_layout_f32",
  "abi-r1:opencl:navatala_transformer_swizzle_layout_f32",
  "abi-r1:opencl:navatala_transformer_swizzle_layout_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_swizzle_layout_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_swizzle_layout_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tileSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_swizzle_layout_f16 = {
  1,
  "navatala_transformer_swizzle_layout_f16",
  "opencl",
  "navatala_transformer_swizzle_layout_f16",
  "kernel:opencl:navatala_transformer_swizzle_layout_f16",
  "abi-r1:opencl:navatala_transformer_swizzle_layout_f16",
  "abi-r1:opencl:navatala_transformer_swizzle_layout_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_swizzle_layout_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_hadamard_transform_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vectorSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_hadamard_transform_f32 = {
  1,
  "navatala_transformer_hadamard_transform_f32",
  "opencl",
  "navatala_transformer_hadamard_transform_f32",
  "kernel:opencl:navatala_transformer_hadamard_transform_f32",
  "abi-r1:opencl:navatala_transformer_hadamard_transform_f32",
  "abi-r1:opencl:navatala_transformer_hadamard_transform_f32",
  4,
  kAbiArgs_opencl_navatala_transformer_hadamard_transform_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_hadamard_transform_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "vectorSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_hadamard_transform_f16 = {
  1,
  "navatala_transformer_hadamard_transform_f16",
  "opencl",
  "navatala_transformer_hadamard_transform_f16",
  "kernel:opencl:navatala_transformer_hadamard_transform_f16",
  "abi-r1:opencl:navatala_transformer_hadamard_transform_f16",
  "abi-r1:opencl:navatala_transformer_hadamard_transform_f16",
  4,
  kAbiArgs_opencl_navatala_transformer_hadamard_transform_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_permute_axes_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_permute_axes_f32 = {
  1,
  "navatala_transformer_permute_axes_f32",
  "opencl",
  "navatala_transformer_permute_axes_f32",
  "kernel:opencl:navatala_transformer_permute_axes_f32",
  "abi-r1:opencl:navatala_transformer_permute_axes_f32",
  "abi-r1:opencl:navatala_transformer_permute_axes_f32",
  10,
  kAbiArgs_opencl_navatala_transformer_permute_axes_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_permute_axes_f16[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_permute_axes_f16 = {
  1,
  "navatala_transformer_permute_axes_f16",
  "opencl",
  "navatala_transformer_permute_axes_f16",
  "kernel:opencl:navatala_transformer_permute_axes_f16",
  "abi-r1:opencl:navatala_transformer_permute_axes_f16",
  "abi-r1:opencl:navatala_transformer_permute_axes_f16",
  10,
  kAbiArgs_opencl_navatala_transformer_permute_axes_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_pack_sequences_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_pack_sequences_f32 = {
  1,
  "navatala_transformer_pack_sequences_f32",
  "opencl",
  "navatala_transformer_pack_sequences_f32",
  "kernel:opencl:navatala_transformer_pack_sequences_f32",
  "abi-r1:opencl:navatala_transformer_pack_sequences_f32",
  "abi-r1:opencl:navatala_transformer_pack_sequences_f32",
  8,
  kAbiArgs_opencl_navatala_transformer_pack_sequences_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_pack_sequences_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "totalTokens", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_pack_sequences_f16 = {
  1,
  "navatala_transformer_pack_sequences_f16",
  "opencl",
  "navatala_transformer_pack_sequences_f16",
  "kernel:opencl:navatala_transformer_pack_sequences_f16",
  "abi-r1:opencl:navatala_transformer_pack_sequences_f16",
  "abi-r1:opencl:navatala_transformer_pack_sequences_f16",
  8,
  kAbiArgs_opencl_navatala_transformer_pack_sequences_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_unpack_sequences_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_unpack_sequences_f32 = {
  1,
  "navatala_transformer_unpack_sequences_f32",
  "opencl",
  "navatala_transformer_unpack_sequences_f32",
  "kernel:opencl:navatala_transformer_unpack_sequences_f32",
  "abi-r1:opencl:navatala_transformer_unpack_sequences_f32",
  "abi-r1:opencl:navatala_transformer_unpack_sequences_f32",
  7,
  kAbiArgs_opencl_navatala_transformer_unpack_sequences_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_unpack_sequences_f16[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqLengths", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seqOffsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "batchSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxSeqLen", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hiddenSize", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_unpack_sequences_f16 = {
  1,
  "navatala_transformer_unpack_sequences_f16",
  "opencl",
  "navatala_transformer_unpack_sequences_f16",
  "kernel:opencl:navatala_transformer_unpack_sequences_f16",
  "abi-r1:opencl:navatala_transformer_unpack_sequences_f16",
  "abi-r1:opencl:navatala_transformer_unpack_sequences_f16",
  7,
  kAbiArgs_opencl_navatala_transformer_unpack_sequences_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_generate_position_ids_f32[] = {
  { "batchSize", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_generate_position_ids_f32 = {
  1,
  "navatala_transformer_generate_position_ids_f32",
  "opencl",
  "navatala_transformer_generate_position_ids_f32",
  "kernel:opencl:navatala_transformer_generate_position_ids_f32",
  "abi-r1:opencl:navatala_transformer_generate_position_ids_f32",
  "abi-r1:opencl:navatala_transformer_generate_position_ids_f32",
  3,
  kAbiArgs_opencl_navatala_transformer_generate_position_ids_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_generate_position_ids_u32[] = {
  { "batchSize", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seqLen", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_generate_position_ids_u32 = {
  1,
  "navatala_transformer_generate_position_ids_u32",
  "opencl",
  "navatala_transformer_generate_position_ids_u32",
  "kernel:opencl:navatala_transformer_generate_position_ids_u32",
  "abi-r1:opencl:navatala_transformer_generate_position_ids_u32",
  "abi-r1:opencl:navatala_transformer_generate_position_ids_u32",
  3,
  kAbiArgs_opencl_navatala_transformer_generate_position_ids_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_generate_causal_mask_f32[] = {
  { "seqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_generate_causal_mask_f32 = {
  1,
  "navatala_transformer_generate_causal_mask_f32",
  "opencl",
  "navatala_transformer_generate_causal_mask_f32",
  "kernel:opencl:navatala_transformer_generate_causal_mask_f32",
  "abi-r1:opencl:navatala_transformer_generate_causal_mask_f32",
  "abi-r1:opencl:navatala_transformer_generate_causal_mask_f32",
  2,
  kAbiArgs_opencl_navatala_transformer_generate_causal_mask_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_generate_causal_mask_f16[] = {
  { "seqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_generate_causal_mask_f16 = {
  1,
  "navatala_transformer_generate_causal_mask_f16",
  "opencl",
  "navatala_transformer_generate_causal_mask_f16",
  "kernel:opencl:navatala_transformer_generate_causal_mask_f16",
  "abi-r1:opencl:navatala_transformer_generate_causal_mask_f16",
  "abi-r1:opencl:navatala_transformer_generate_causal_mask_f16",
  2,
  kAbiArgs_opencl_navatala_transformer_generate_causal_mask_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_precompute_rotary_emb_f32[] = {
  { "maxSeqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cosOutput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinOutput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_precompute_rotary_emb_f32 = {
  1,
  "navatala_transformer_precompute_rotary_emb_f32",
  "opencl",
  "navatala_transformer_precompute_rotary_emb_f32",
  "kernel:opencl:navatala_transformer_precompute_rotary_emb_f32",
  "abi-r1:opencl:navatala_transformer_precompute_rotary_emb_f32",
  "abi-r1:opencl:navatala_transformer_precompute_rotary_emb_f32",
  5,
  kAbiArgs_opencl_navatala_transformer_precompute_rotary_emb_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_transformer_precompute_rotary_emb_f16[] = {
  { "maxSeqLen", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "base", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cosOutput", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sinOutput", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_transformer_precompute_rotary_emb_f16 = {
  1,
  "navatala_transformer_precompute_rotary_emb_f16",
  "opencl",
  "navatala_transformer_precompute_rotary_emb_f16",
  "kernel:opencl:navatala_transformer_precompute_rotary_emb_f16",
  "abi-r1:opencl:navatala_transformer_precompute_rotary_emb_f16",
  "abi-r1:opencl:navatala_transformer_precompute_rotary_emb_f16",
  5,
  kAbiArgs_opencl_navatala_transformer_precompute_rotary_emb_f16
};

bool tryGetKernelAbiManifest_opencl_transformer(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_gelu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_tanh_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_gelu_tanh_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_silu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_relu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_quick_gelu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_squared_relu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_squared_relu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_bias_gelu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_bias_gelu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_gelu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_silu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_relu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_quick_gelu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_squared_relu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_squared_relu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_geglu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_geglu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swiglu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_swiglu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_reglu_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_reglu_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_geglu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_geglu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swiglu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_swiglu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_reglu_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_reglu_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_gelu_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_silu_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_relu_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_quick_gelu_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_gelu_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_silu_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_relu_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_quick_gelu_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_forward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rms_norm_forward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_forward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_rms_norm_forward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_forward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_layer_norm_forward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_forward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_layer_norm_forward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_layer_norm_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_layer_norm_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rms_norm_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_rms_norm_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_multi_pass_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_layer_norm_multi_pass_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_multi_pass_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_layer_norm_multi_pass_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_multi_pass_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rms_norm_multi_pass_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_multi_pass_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_rms_norm_multi_pass_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_forward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_forward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_forward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_forward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_with_mask_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_with_mask_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_with_mask_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_with_mask_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_multi_pass_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_multi_pass_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_multi_pass_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_softmax_multi_pass_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f32_to_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_cast_f32_to_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f16_to_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_cast_f16_to_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_transpose_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_transpose_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_fused_cast_transpose_f32_to_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_fused_cast_transpose_f32_to_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e4_m3") {
    out = &kAbiManifest_opencl_navatala_transformer_cast_f32_to_f_p8_e4_m3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f_p8_e4_m3_to_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_cast_f_p8_e4_m3_to_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e5_m2") {
    out = &kAbiManifest_opencl_navatala_transformer_cast_f32_to_f_p8_e5_m2;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f_p8_e5_m2_to_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_cast_f_p8_e5_m2_to_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_fused_cast_transpose_f_p8") {
    out = &kAbiManifest_opencl_navatala_transformer_fused_cast_transpose_f_p8;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose4_d_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_transpose4_d_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose4_d_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_transpose4_d_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_forward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_dropout_forward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_forward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_dropout_forward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_dropout_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_dropout_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_forward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_forward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_forward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_forward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_with_cache_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_with_cache_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_interleaved_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_interleaved_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_interleaved_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_rope_interleaved_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_paged_attention_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_paged_attention_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_mask_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_with_mask_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_mask_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_with_mask_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_padding_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_with_padding_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_padding_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_with_padding_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_simple_attention_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_kv_cache_append_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_kv_cache_append_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_kv_cache_append_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_kv_cache_append_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_kv_cache_rotate_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_kv_cache_rotate_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_tiled_gemm_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_tiled_gemm_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_backward_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_tiled_gemm_backward_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_backward_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_tiled_gemm_backward_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_scale_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_scale_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_scale_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_scale_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_add_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_add_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_add_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_add_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_copy_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_copy_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_copy_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_copy_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_l2_norm_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_l2_norm_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_clip_grad_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_multi_tensor_clip_grad_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_top_k_gating_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_top_k_gating_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_top_k_gating_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_top_k_gating_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpermute_tokens_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_unpermute_tokens_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpermute_tokens_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_unpermute_tokens_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pad_sequence_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_pad_sequence_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pad_sequence_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_pad_sequence_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpad_sequence_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_unpad_sequence_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpad_sequence_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_unpad_sequence_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swizzle_layout_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_swizzle_layout_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swizzle_layout_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_swizzle_layout_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_hadamard_transform_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_hadamard_transform_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_hadamard_transform_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_hadamard_transform_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_permute_axes_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_permute_axes_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_permute_axes_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_permute_axes_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pack_sequences_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_pack_sequences_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pack_sequences_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_pack_sequences_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpack_sequences_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_unpack_sequences_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpack_sequences_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_unpack_sequences_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_position_ids_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_generate_position_ids_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_position_ids_u32") {
    out = &kAbiManifest_opencl_navatala_transformer_generate_position_ids_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_causal_mask_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_generate_causal_mask_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_causal_mask_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_generate_causal_mask_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_precompute_rotary_emb_f32") {
    out = &kAbiManifest_opencl_navatala_transformer_precompute_rotary_emb_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_precompute_rotary_emb_f16") {
    out = &kAbiManifest_opencl_navatala_transformer_precompute_rotary_emb_f16;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_transformer(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_gelu_f32";
    std::string_view sv(k_opencl_navatala_transformer_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_tanh_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_gelu_tanh_f32";
    std::string_view sv(k_opencl_navatala_transformer_gelu_tanh_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_silu_f32";
    std::string_view sv(k_opencl_navatala_transformer_silu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_relu_f32";
    std::string_view sv(k_opencl_navatala_transformer_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_quick_gelu_f32";
    std::string_view sv(k_opencl_navatala_transformer_quick_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_squared_relu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_squared_relu_f32";
    std::string_view sv(k_opencl_navatala_transformer_squared_relu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_bias_gelu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_bias_gelu_f32";
    std::string_view sv(k_opencl_navatala_transformer_bias_gelu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_gelu_f16";
    std::string_view sv(k_opencl_navatala_transformer_gelu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_silu_f16";
    std::string_view sv(k_opencl_navatala_transformer_silu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_relu_f16";
    std::string_view sv(k_opencl_navatala_transformer_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_quick_gelu_f16";
    std::string_view sv(k_opencl_navatala_transformer_quick_gelu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_squared_relu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_squared_relu_f16";
    std::string_view sv(k_opencl_navatala_transformer_squared_relu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_geglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_geglu_f32";
    std::string_view sv(k_opencl_navatala_transformer_geglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swiglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_swiglu_f32";
    std::string_view sv(k_opencl_navatala_transformer_swiglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_reglu_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_reglu_f32";
    std::string_view sv(k_opencl_navatala_transformer_reglu_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_geglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_geglu_f16";
    std::string_view sv(k_opencl_navatala_transformer_geglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swiglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_swiglu_f16";
    std::string_view sv(k_opencl_navatala_transformer_swiglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_reglu_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_reglu_f16";
    std::string_view sv(k_opencl_navatala_transformer_reglu_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_gelu_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_gelu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_silu_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_silu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_relu_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_relu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_quick_gelu_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_quick_gelu_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_gelu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_gelu_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_gelu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_silu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_silu_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_silu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_relu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_relu_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_relu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_quick_gelu_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_quick_gelu_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_quick_gelu_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rms_norm_forward_f32";
    std::string_view sv(k_opencl_navatala_transformer_rms_norm_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rms_norm_forward_f16";
    std::string_view sv(k_opencl_navatala_transformer_rms_norm_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_layer_norm_forward_f32";
    std::string_view sv(k_opencl_navatala_transformer_layer_norm_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_layer_norm_forward_f16";
    std::string_view sv(k_opencl_navatala_transformer_layer_norm_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_layer_norm_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_layer_norm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_layer_norm_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_layer_norm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rms_norm_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_rms_norm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rms_norm_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_rms_norm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_layer_norm_multi_pass_f32";
    std::string_view sv(k_opencl_navatala_transformer_layer_norm_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_layer_norm_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_layer_norm_multi_pass_f16";
    std::string_view sv(k_opencl_navatala_transformer_layer_norm_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rms_norm_multi_pass_f32";
    std::string_view sv(k_opencl_navatala_transformer_rms_norm_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rms_norm_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rms_norm_multi_pass_f16";
    std::string_view sv(k_opencl_navatala_transformer_rms_norm_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_forward_f32";
    std::string_view sv(k_opencl_navatala_transformer_softmax_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_forward_f16";
    std::string_view sv(k_opencl_navatala_transformer_softmax_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_softmax_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_softmax_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_with_mask_f32";
    std::string_view sv(k_opencl_navatala_transformer_softmax_with_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_with_mask_f16";
    std::string_view sv(k_opencl_navatala_transformer_softmax_with_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_with_mask_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_softmax_with_mask_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_with_mask_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_with_mask_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_softmax_with_mask_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_multi_pass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_multi_pass_f32";
    std::string_view sv(k_opencl_navatala_transformer_softmax_multi_pass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_softmax_multi_pass_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_softmax_multi_pass_f16";
    std::string_view sv(k_opencl_navatala_transformer_softmax_multi_pass_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_cast_f32_to_f16";
    std::string_view sv(k_opencl_navatala_transformer_cast_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f16_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_cast_f16_to_f32";
    std::string_view sv(k_opencl_navatala_transformer_cast_f16_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_transpose_f32";
    std::string_view sv(k_opencl_navatala_transformer_transpose_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_transpose_f16";
    std::string_view sv(k_opencl_navatala_transformer_transpose_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_fused_cast_transpose_f32_to_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_fused_cast_transpose_f32_to_f16";
    std::string_view sv(k_opencl_navatala_transformer_fused_cast_transpose_f32_to_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e4_m3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_cast_f32_to_f_p8_e4_m3";
    std::string_view sv(k_opencl_navatala_transformer_cast_f32_to_f_p8_e4_m3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f_p8_e4_m3_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_cast_f_p8_e4_m3_to_f32";
    std::string_view sv(k_opencl_navatala_transformer_cast_f_p8_e4_m3_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f32_to_f_p8_e5_m2") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_cast_f32_to_f_p8_e5_m2";
    std::string_view sv(k_opencl_navatala_transformer_cast_f32_to_f_p8_e5_m2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_cast_f_p8_e5_m2_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_cast_f_p8_e5_m2_to_f32";
    std::string_view sv(k_opencl_navatala_transformer_cast_f_p8_e5_m2_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_fused_cast_transpose_f_p8") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_fused_cast_transpose_f_p8";
    std::string_view sv(k_opencl_navatala_transformer_fused_cast_transpose_f_p8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose4_d_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_transpose4_d_f32";
    std::string_view sv(k_opencl_navatala_transformer_transpose4_d_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_transpose4_d_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_transpose4_d_f16";
    std::string_view sv(k_opencl_navatala_transformer_transpose4_d_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_dropout_forward_f32";
    std::string_view sv(k_opencl_navatala_transformer_dropout_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_dropout_forward_f16";
    std::string_view sv(k_opencl_navatala_transformer_dropout_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_dropout_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_dropout_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_dropout_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_dropout_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_dropout_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_forward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_forward_f32";
    std::string_view sv(k_opencl_navatala_transformer_rope_forward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_forward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_forward_f16";
    std::string_view sv(k_opencl_navatala_transformer_rope_forward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_with_cache_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_with_cache_f32";
    std::string_view sv(k_opencl_navatala_transformer_rope_with_cache_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_rope_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_rope_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_interleaved_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_interleaved_f32";
    std::string_view sv(k_opencl_navatala_transformer_rope_interleaved_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_rope_interleaved_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_rope_interleaved_f16";
    std::string_view sv(k_opencl_navatala_transformer_rope_interleaved_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_f16";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_f32";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_paged_attention_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_paged_attention_f32";
    std::string_view sv(k_opencl_navatala_transformer_paged_attention_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_with_mask_f32";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_with_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_with_mask_f16";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_with_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_padding_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_with_padding_f32";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_with_padding_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_with_padding_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_with_padding_f16";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_with_padding_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_simple_attention_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_simple_attention_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_simple_attention_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_kv_cache_append_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_kv_cache_append_f16";
    std::string_view sv(k_opencl_navatala_transformer_kv_cache_append_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_kv_cache_append_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_kv_cache_append_f32";
    std::string_view sv(k_opencl_navatala_transformer_kv_cache_append_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_kv_cache_rotate_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_kv_cache_rotate_f16";
    std::string_view sv(k_opencl_navatala_transformer_kv_cache_rotate_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_tiled_gemm_f16";
    std::string_view sv(k_opencl_navatala_transformer_tiled_gemm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_tiled_gemm_f32";
    std::string_view sv(k_opencl_navatala_transformer_tiled_gemm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_backward_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_tiled_gemm_backward_f32";
    std::string_view sv(k_opencl_navatala_transformer_tiled_gemm_backward_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_tiled_gemm_backward_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_tiled_gemm_backward_f16";
    std::string_view sv(k_opencl_navatala_transformer_tiled_gemm_backward_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_scale_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_scale_f32";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_scale_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_scale_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_scale_f16";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_scale_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_add_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_add_f32";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_add_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_add_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_add_f16";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_add_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_copy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_copy_f32";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_copy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_copy_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_copy_f16";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_copy_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_l2_norm_f32";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_l2_norm_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_l2_norm_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_l2_norm_f16";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_l2_norm_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_clip_grad_f32";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_clip_grad_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_multi_tensor_clip_grad_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_multi_tensor_clip_grad_f16";
    std::string_view sv(k_opencl_navatala_transformer_multi_tensor_clip_grad_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_top_k_gating_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_top_k_gating_f32";
    std::string_view sv(k_opencl_navatala_transformer_top_k_gating_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_top_k_gating_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_top_k_gating_f16";
    std::string_view sv(k_opencl_navatala_transformer_top_k_gating_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpermute_tokens_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_unpermute_tokens_f32";
    std::string_view sv(k_opencl_navatala_transformer_unpermute_tokens_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpermute_tokens_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_unpermute_tokens_f16";
    std::string_view sv(k_opencl_navatala_transformer_unpermute_tokens_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pad_sequence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_pad_sequence_f32";
    std::string_view sv(k_opencl_navatala_transformer_pad_sequence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pad_sequence_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_pad_sequence_f16";
    std::string_view sv(k_opencl_navatala_transformer_pad_sequence_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpad_sequence_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_unpad_sequence_f32";
    std::string_view sv(k_opencl_navatala_transformer_unpad_sequence_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpad_sequence_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_unpad_sequence_f16";
    std::string_view sv(k_opencl_navatala_transformer_unpad_sequence_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swizzle_layout_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_swizzle_layout_f32";
    std::string_view sv(k_opencl_navatala_transformer_swizzle_layout_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_swizzle_layout_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_swizzle_layout_f16";
    std::string_view sv(k_opencl_navatala_transformer_swizzle_layout_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_hadamard_transform_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_hadamard_transform_f32";
    std::string_view sv(k_opencl_navatala_transformer_hadamard_transform_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_hadamard_transform_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_hadamard_transform_f16";
    std::string_view sv(k_opencl_navatala_transformer_hadamard_transform_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_permute_axes_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_permute_axes_f32";
    std::string_view sv(k_opencl_navatala_transformer_permute_axes_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_permute_axes_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_permute_axes_f16";
    std::string_view sv(k_opencl_navatala_transformer_permute_axes_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pack_sequences_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_pack_sequences_f32";
    std::string_view sv(k_opencl_navatala_transformer_pack_sequences_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_pack_sequences_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_pack_sequences_f16";
    std::string_view sv(k_opencl_navatala_transformer_pack_sequences_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpack_sequences_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_unpack_sequences_f32";
    std::string_view sv(k_opencl_navatala_transformer_unpack_sequences_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_unpack_sequences_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_unpack_sequences_f16";
    std::string_view sv(k_opencl_navatala_transformer_unpack_sequences_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_position_ids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_generate_position_ids_f32";
    std::string_view sv(k_opencl_navatala_transformer_generate_position_ids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_position_ids_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_generate_position_ids_u32";
    std::string_view sv(k_opencl_navatala_transformer_generate_position_ids_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_causal_mask_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_generate_causal_mask_f32";
    std::string_view sv(k_opencl_navatala_transformer_generate_causal_mask_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_generate_causal_mask_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_generate_causal_mask_f16";
    std::string_view sv(k_opencl_navatala_transformer_generate_causal_mask_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_precompute_rotary_emb_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_precompute_rotary_emb_f32";
    std::string_view sv(k_opencl_navatala_transformer_precompute_rotary_emb_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_transformer_precompute_rotary_emb_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_transformer_precompute_rotary_emb_f16";
    std::string_view sv(k_opencl_navatala_transformer_precompute_rotary_emb_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

