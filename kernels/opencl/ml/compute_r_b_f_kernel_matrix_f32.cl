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

__kernel void navatala_ml_compute_r_b_f_kernel_matrix_f32(__global const float* X1, __global const float* X2, __global const float* gamma, __global const uint* n1, __global const uint* n2, __global const uint* n_features, __global float* K) {
  int gid0 = (int)get_global_id(0);
  uint gidX = ((uint)((int)(get_global_id(0))));
  uint gidY = ((uint)((int)(get_global_id(1))));
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
    float sqDist = as_float(0x00000000u);
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
    float negGammaDist = (as_float(0xbf800000u) * (gammaVal * finalSqDist));
    float kernelVal = exp(negGammaDist);
    uint outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}
