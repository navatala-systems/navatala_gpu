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

__kernel void navatala_cfd_scaled_dot_product(__global const float* q, __global const float* k, __global const int* numQ, __global const int* numK, __global const int* headDim, __global float* scores) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * nk);
  if ((gid < total)) {
    int qi = (gid / nk);
    int kj = (gid % nk);
    float acc = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      int qFlat = ((qi * hd) + d);
      int kFlat = ((kj * hd) + d);
      float qv = q[qFlat];
      float kv = k[kFlat];
      acc = (acc + (qv * kv));
    }
    scores[gid] = (acc * as_float(0x3e800000u));
  }
}
