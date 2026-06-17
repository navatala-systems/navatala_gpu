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

kernel void navatala_ml_svm_predict_classify_f32(device const float* support_vectors [[buffer(0)]], device const float* dual_coefs [[buffer(1)]], device const float* X [[buffer(2)]], device const float* bias [[buffer(3)]], device const float* gamma [[buffer(4)]], device const uint* n_samples [[buffer(5)]], device const uint* n_sv [[buffer(6)]], device const uint* n_features [[buffer(7)]], device int* predictions [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nSV = n_sv[0];
  uint nFeatures = n_features[0];
  float biasVal = bias[0];
  float gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nFeatures);
    float decisionSum = as_type<float>(0x00000000u);
    for (int sv = 0; sv < (int)(nSV); ++sv) {
      uint svU32 = ((uint)(sv));
      uint svBase = (svU32 * nFeatures);
      float sqDist = as_type<float>(0x00000000u);
      for (int f = 0; f < (int)(nFeatures); ++f) {
        uint fU32 = ((uint)(f));
        uint xIdx = (sampleBase + fU32);
        uint svIdx = (svBase + fU32);
        float xVal = X[xIdx];
        float svVal = support_vectors[svIdx];
        float diff = (xVal - svVal);
        float diffSq = (diff * diff);
        float currSqDist = sqDist;
        float newSqDist = (currSqDist + diffSq);
        sqDist = newSqDist;
      }
      float finalSqDist = sqDist;
      float negGammaDist = (as_type<float>(0xbf800000u) * (gammaVal * finalSqDist));
      float kernelVal = exp(negGammaDist);
      float alpha = dual_coefs[svU32];
      float contribution = (alpha * kernelVal);
      float currSum = decisionSum;
      float newSum = (currSum + contribution);
      decisionSum = newSum;
    }
    float finalSum = decisionSum;
    float decision = (finalSum + biasVal);
    bool isPositive = (decision >= as_type<float>(0x00000000u));
    if (isPositive) {
      predictions[gid] = 1;
    } else {
      predictions[gid] = -1;
    }
  }
}
