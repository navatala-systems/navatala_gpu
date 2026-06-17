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
__kernel void navatala_linalg_batched_dot_f64(__global const double* a, __global const double* b, __global const uint* vecLen, __global const uint* batchSize, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint len = vecLen[(uint)(0u)];
  uint numBatches = batchSize[(uint)(0u)];
  uint k = gid;
  bool inBounds = (k < numBatches);
  if (inBounds) {
    uint baseOffset = (k * len);
    double sumAccum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(len); ++i) {
      uint iU32 = ((uint)(i));
      uint idx = (baseOffset + iU32);
      double aVal = a[idx];
      double bVal = b[idx];
      double prod = (aVal * bVal);
      double currentSum = sumAccum;
      double nextSum = (currentSum + prod);
      sumAccum = nextSum;
    }
    double finalSum = sumAccum;
    result[k] = finalSum;
  }
}
