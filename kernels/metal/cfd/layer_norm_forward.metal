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

kernel void navatala_cfd_layer_norm_forward(device const float* x [[buffer(0)]], device const int* numRows [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]], device float* y [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int nRows = numRows[0];
  if (row < nRows) {
    int hd = 16;
    int rowBase = (row * hd);
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      sum = (sum + xv);
    }
    float hdF = ((float)(hd));
    float mean = (sum / hdF);
    float varSum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float diff = (xv - mean);
      varSum = (varSum + (diff * diff));
    }
    float var = (varSum / hdF);
    float invStd = sqrt((as_type<float>(0x3f800000u) / (var + as_type<float>(0x3727c5acu))));
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float xn = ((xv - mean) * invStd);
      float g = gamma[j];
      float b = beta[j];
      float yv = ((xn * g) + b);
      y[idx] = yv;
    }
  }
}
