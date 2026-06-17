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

kernel void navatala_ml_compute_r_b_f_kernel_matrix_f32(device const float* X1 [[buffer(0)]], device const float* X2 [[buffer(1)]], device const float* gamma [[buffer(2)]], device const uint* n1 [[buffer(3)]], device const uint* n2 [[buffer(4)]], device const uint* n_features [[buffer(5)]], device float* K [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gidX = ((uint)(int(__gid.x)));
  uint gidY = ((uint)(int(__gid.y)));
  uint nRows = n1[0];
  uint nCols = n2[0];
  uint nFeatures = n_features[0];
  float gammaVal = gamma[0];
  bool inBoundsX = (gidX < nRows);
  bool inBoundsY = (gidY < nCols);
  bool inBounds = (inBoundsX && inBoundsY);
  if (inBounds) {
    uint x1Base = (gidX * nFeatures);
    uint x2Base = (gidY * nFeatures);
    float sqDist = as_type<float>(0x00000000u);
    for (int f = 0; f < (int)(nFeatures); ++f) {
      uint fU32 = ((uint)(f));
      uint x1Idx = (x1Base + fU32);
      uint x2Idx = (x2Base + fU32);
      float x1Val = X1[x1Idx];
      float x2Val = X2[x2Idx];
      float diff = (x1Val - x2Val);
      float diffSq = (diff * diff);
      float currSqDist = sqDist;
      float newSqDist = (currSqDist + diffSq);
      sqDist = newSqDist;
    }
    float finalSqDist = sqDist;
    float negGammaDist = (as_type<float>(0xbf800000u) * (gammaVal * finalSqDist));
    float kernelVal = exp(negGammaDist);
    uint outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}
