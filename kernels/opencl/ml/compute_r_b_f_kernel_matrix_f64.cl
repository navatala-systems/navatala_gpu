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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_compute_r_b_f_kernel_matrix_f64(__global const double* X1, __global const double* X2, __global const double* gamma, __global const uint* n1, __global const uint* n2, __global const uint* n_features, __global double* K) {
  int gid0 = (int)get_global_id(0);
  uint gidX = ((uint)((int)(get_global_id(0))));
  uint gidY = ((uint)((int)(get_global_id(1))));
  uint nRows = n1[0];
  uint nCols = n2[0];
  uint nFeatures = n_features[0];
  double gammaVal = gamma[0];
  bool inBoundsX = (gidX < nRows);
  bool inBoundsY = (gidY < nCols);
  bool inBounds = (inBoundsX && inBoundsY);
  if (inBounds) {
    uint x1Base = (gidX * nFeatures);
    uint x2Base = (gidY * nFeatures);
    double sqDist = as_double(0x0000000000000000ul);
    for (int f = 0; f < (int)(nFeatures); ++f) {
      uint fU32 = ((uint)(f));
      uint x1Idx = (x1Base + fU32);
      uint x2Idx = (x2Base + fU32);
      double x1Val = X1[x1Idx];
      double x2Val = X2[x2Idx];
      double diff = (x1Val - x2Val);
      double diffSq = (diff * diff);
      double currSqDist = sqDist;
      double newSqDist = (currSqDist + diffSq);
      sqDist = newSqDist;
    }
    double finalSqDist = sqDist;
    double negGammaDist = (as_double(0xbff0000000000000ul) * (gammaVal * finalSqDist));
    double kernelVal = exp(negGammaDist);
    uint outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}
