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

kernel void navatala_ml_compute_repulsion_f32(device const float* embedding [[buffer(0)]], device const int* negative_samples [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device const uint* n_neg [[buffer(3)]], device const uint* n_components [[buffer(4)]], device const float* a_param [[buffer(5)]], device const float* b_param [[buffer(6)]], device const float* gamma [[buffer(7)]], device float* gradients [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nNeg = n_neg[0];
  uint nComp = n_components[0];
  float a = a_param[0];
  float b = b_param[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nComp);
    uint negBase = (gid * nNeg);
    for (int n = 0; n < (int)(nNeg); ++n) {
      uint nU32 = ((uint)(n));
      uint negSampleIdx = (negBase + nU32);
      int negIdx = negative_samples[negSampleIdx];
      bool validNeg = (negIdx >= 0);
      if (validNeg) {
        uint negIdxU32 = ((uint)(negIdx));
        uint negEmbBase = (negIdxU32 * nComp);
        float distSqNeg = as_type<float>(0x38d1b717u);
        for (int d = 0; d < (int)(nComp); ++d) {
          uint dU32 = ((uint)(d));
          uint iAddr = (sampleBase + dU32);
          uint jAddr = (negEmbBase + dU32);
          float iVal = embedding[iAddr];
          float jVal = embedding[jAddr];
          float diff = (iVal - jVal);
          float diffSq = (diff * diff);
          float currDSq = distSqNeg;
          float newDSq = (currDSq + diffSq);
          distSqNeg = newDSq;
        }
        float dSq = distSqNeg;
        float logDSqNeg = log(dSq);
        float dPow2b = exp((b * logDSqNeg));
        float denom = (as_type<float>(0x3f800000u) + (a * dPow2b));
        float gradScale = (((as_type<float>(0x40000000u) * gammaVal) * b) / (dSq * denom));
        for (int d = 0; d < (int)(nComp); ++d) {
          uint dU32 = ((uint)(d));
          uint iAddr = (sampleBase + dU32);
          uint jAddr = (negEmbBase + dU32);
          float iVal = embedding[iAddr];
          float jVal = embedding[jAddr];
          float diff = (iVal - jVal);
          float grad = (gradScale * diff);
          float gradClamped = (((grad > as_type<float>(0x40800000u))) ? (as_type<float>(0x40800000u)) : ((((grad < as_type<float>(0xc0800000u))) ? (as_type<float>(0xc0800000u)) : (grad))));
          float currGrad = gradients[iAddr];
          float newGrad = (currGrad + gradClamped);
          gradients[iAddr] = newGrad;
        }
      }
    }
  }
}
