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

__kernel void navatala_sparse_jacobi_davidson_correction_f32(__global const float* r, __global const float* v, __global const float* theta, __global const uint* n, __global float* t) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (gid < N) {
    float ri = r[gid];
    float th = theta[0];
    float negR = (-ri);
    float absTh = fabs(th);
    float safeTh = (((absTh > as_float(0x0da24260u))) ? (th) : (as_float(0x3f800000u)));
    bool useDiag = (absTh > as_float(0x0da24260u));
    float ti = ((useDiag) ? ((negR / safeTh)) : (negR));
    t[gid] = ti;
  }
}
