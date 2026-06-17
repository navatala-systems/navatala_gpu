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
__kernel void navatala_ml_update_weights_f64(__global const double* responsibilities, __global const uint* n, __global const uint* k, __global double* weights) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  uint numK = k[0];
  uint j = gid;
  bool inBounds = (j < numK);
  if (inBounds) {
    double respSumAccum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      uint respIdx = ((iU32 * numK) + j);
      double r_ij = responsibilities[respIdx];
      double currRespSum = respSumAccum;
      double newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    double respSum = respSumAccum;
    double nFloat = ((double)(numN));
    double weightVal = (respSum / nFloat);
    weights[j] = weightVal;
  }
}
