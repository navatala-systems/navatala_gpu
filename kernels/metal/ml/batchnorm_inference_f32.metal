// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_batchnorm_inference_f32(device const float* x [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* var [[buffer(2)]], device const float* gamma [[buffer(3)]], device const float* beta [[buffer(4)]], device const uint* count [[buffer(5)]], device const float* eps [[buffer(6)]], device float* _output [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if ((gid < countVal)) {
    float epsVal = eps[0];
    float d = (x[gid] - mean[gid]);
    float denom = sqrt((var[gid] + epsVal));
    float xn = (d / denom);
    float outF = ((gamma[gid] * xn) + beta[gid]);
    _output[gid] = outF;
  }
}
