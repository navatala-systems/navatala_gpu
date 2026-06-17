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

kernel void navatala_ml_ar_predict_f32(device const float* y [[buffer(0)]], device const float* phi [[buffer(1)]], device const float* intercept [[buffer(2)]], device const uint* p [[buffer(3)]], device const uint* batch_size [[buffer(4)]], device float* predictions [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint batchSize = batch_size[0];
  uint pVal = p[0];
  bool inBounds = (gid < batchSize);
  if (inBounds) {
    float pred = intercept[gid];
    float sumAR = pred;
    for (int i = 0; i < (int)(pVal); ++i) {
      uint iU32 = ((uint)(i));
      uint yIdx = ((gid * pVal) + iU32);
      float yVal = y[yIdx];
      uint phiIdx = yIdx;
      float phiVal = phi[phiIdx];
      float term = (phiVal * yVal);
      float currSum = sumAR;
      float newSum = (currSum + term);
      sumAR = newSum;
    }
    float finalPred = sumAR;
    predictions[gid] = finalPred;
  }
}
