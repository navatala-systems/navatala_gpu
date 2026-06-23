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

kernel void navatala_dataframe_weighted_covariance_update_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* w [[buffer(2)]], device float* weightSum [[buffer(3)]], device float* meanX [[buffer(4)]], device float* meanY [[buffer(5)]], device float* Cxy [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float wOld = weightSum[0];
    float muXOld = meanX[0];
    float muYOld = meanY[0];
    float cxyOld = Cxy[0];
    float xVal = x[0];
    float yVal = y[0];
    float wVal = w[0];
    float wNew = (wOld + wVal);
    float wNewSafe = (((wNew == as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (wNew));
    float deltaX = (xVal - muXOld);
    float deltaY = (yVal - muYOld);
    float muXNew = (muXOld + ((wVal / wNewSafe) * deltaX));
    float muYNew = (muYOld + ((wVal / wNewSafe) * deltaY));
    float yMinusMuYNew = (yVal - muYNew);
    float cxyNew = (cxyOld + (wVal * (deltaX * yMinusMuYNew)));
    weightSum[0] = wNew;
    meanX[0] = muXNew;
    meanY[0] = muYNew;
    Cxy[0] = cxyNew;
  }
}
