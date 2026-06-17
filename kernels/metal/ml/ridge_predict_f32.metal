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

kernel void navatala_ml_ridge_predict_f32(device const float* X [[buffer(0)]], device const float* beta [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* p [[buffer(3)]], device float* yPred [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  uint pVal = p[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint colIdx = lid;
    bool colInBounds = (colIdx < pVal);
    if (colInBounds) {
      uint elemIdx = ((rowIdx * pVal) + colIdx);
      float xVal = X[elemIdx];
      float betaVal = beta[colIdx];
      float product = (xVal * betaVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pred32_reductionStride = 128u;
    for (int pred32_reductionStep = 0; pred32_reductionStep < (int)(8); ++pred32_reductionStep) {
      uint pred32_stride = pred32_reductionStride;
      if ((lid < pred32_stride)) {
        float pred32_other = sdata[(lid + pred32_stride)];
        float pred32_mine = sdata[lid];
        float pred32_sum = (pred32_mine + pred32_other);
        sdata[lid] = pred32_sum;
      }
      uint pred32_strideToHalve = pred32_reductionStride;
      uint pred32_nextStride = (pred32_strideToHalve >> 1u);
      pred32_reductionStride = pred32_nextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if ((lid == 0u)) {
      float dotProduct = sdata[0];
      yPred[rowIdx] = dotProduct;
    }
  }
}
