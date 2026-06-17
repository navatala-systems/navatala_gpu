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

__kernel void navatala_ml_compute_covariance_matrix_f32(__global const float* Xc, __global const uint* nSamples, __global const uint* nFeatures, __global float* Cov) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint covSize = (d * d);
  bool inBounds = (gid < covSize);
  if (inBounds) {
    uint i = (gid / d);
    uint j = (gid % d);
    float sum = as_float(0x00000000u);
    for (int k = 0; k < (int)(n); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = ((kU32 * d) + i);
      uint idxJ = ((kU32 * d) + j);
      float xi = Xc[idxI];
      float xj = Xc[idxJ];
      float prod = (xi * xj);
      float currSum = sum;
      float newSum = (currSum + prod);
      sum = newSum;
    }
    float finalSum = sum;
    uint nMinus1 = (n - (uint)(1u));
    float divisor = ((float)(nMinus1));
    float covVal = (finalSum / divisor);
    Cov[gid] = covVal;
  }
}
