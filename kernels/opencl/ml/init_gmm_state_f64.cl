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
__kernel void navatala_ml_init_gmm_state_f64(__global const double* initialMeans, __global const uint* n, __global const uint* k, __global const uint* d, __global double* means, __global double* variances, __global double* weights) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numK = k[0];
  uint numD = d[0];
  uint kd = (numK * numD);
  bool inMatrixBounds = (gid < kd);
  if (inMatrixBounds) {
    double meanVal = initialMeans[gid];
    means[gid] = meanVal;
    variances[gid] = as_double(0x3ff0000000000000ul);
  }
  bool inWeightBounds = (gid < numK);
  if (inWeightBounds) {
    double kFloat = ((double)(numK));
    double invK = (as_double(0x3ff0000000000000ul) / kFloat);
    weights[gid] = invK;
  }
}
