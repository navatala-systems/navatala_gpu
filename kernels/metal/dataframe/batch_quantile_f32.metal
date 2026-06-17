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

kernel void navatala_dataframe_batch_quantile_f32(device const float* sortedData [[buffer(0)]], device const uint* quantiles [[buffer(1)]], device const uint* dataCount [[buffer(2)]], device const uint* numQuantiles [[buffer(3)]], device float* results [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nQ = numQuantiles[0u];
  uint nData = dataCount[0u];
  bool inBounds = (gid < nQ);
  if (inBounds) {
    uint q = quantiles[gid];
    uint nMinus1 = (nData - 1u);
    uint posScaled = (q * nMinus1);
    uint floor = (posScaled / 1000u);
    uint ceilRaw = (floor + 1u);
    uint ceil = (((ceilRaw < nData)) ? (ceilRaw) : (nMinus1));
    uint frac = (posScaled % 1000u);
    float floorVal = sortedData[floor];
    float ceilVal = sortedData[ceil];
    float diff = (ceilVal - floorVal);
    float fracF = ((float)(frac));
    float scaledDiff = (diff * fracF);
    float interpVal = (scaledDiff / as_type<float>(0x447a0000u));
    float quantileVal = (floorVal + interpVal);
    results[gid] = quantileVal;
  }
}
