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
__kernel void navatala_dataframe_reservoir_update256_f64(__global const double* x, __global double* reservoir, __global uint* count, __global uint* seed) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    double xVal = x[0];
    uint n = count[0];
    uint s = seed[0];
    uint s1 = (s ^ (s << (uint)(13u)));
    uint s2 = (s1 ^ (s1 >> (uint)(17u)));
    uint s3 = (s2 ^ (s2 << (uint)(5u)));
    uint nPlus1 = (n + (uint)(1u));
    uint j = (s3 % nPlus1);
    bool fillPhase = (n < (uint)(256u));
    bool replacePhase = ((n >= (uint)(256u)) && (j < (uint)(256u)));
    if (fillPhase) {
      reservoir[n] = xVal;
    }
    if (replacePhase) {
      reservoir[j] = xVal;
    }
    count[0] = nPlus1;
    seed[0] = s3;
  }
}
