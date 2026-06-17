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
__kernel void navatala_dataframe_normal_box_muller_f64(__global ulong* state, __global const uint* count, __global double* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[(uint)(0u)];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    ulong s = state[gid];
    ulong aU64 = ((ulong)((uint)(1103515245u)));
    ulong cU64 = ((ulong)((uint)(12345u)));
    ulong s1 = (((aU64 * s) + cU64) & ((ulong)((uint)(2147483647u))));
    double u1 = (((double)(s1)) / as_double(0x41e0000000000000ul));
    ulong s2 = (((aU64 * s1) + cU64) & ((ulong)((uint)(2147483647u))));
    double u2 = (((double)(s2)) / as_double(0x41e0000000000000ul));
    state[gid] = s2;
    double u1Clamped = (((u1 < as_double(0x3e7ad7f29abcaf48ul))) ? (as_double(0x3e7ad7f29abcaf48ul)) : (u1));
    double logU1 = log(u1Clamped);
    double negTwoLogU1 = (as_double(0xc000000000000000ul) * logU1);
    double sqrtTerm = sqrt(negTwoLogU1);
    double angle = (as_double(0x401921fb54442d18ul) * u2);
    double cosTerm = cos(angle);
    double z = (sqrtTerm * cosTerm);
    _output[gid] = z;
  }
}
