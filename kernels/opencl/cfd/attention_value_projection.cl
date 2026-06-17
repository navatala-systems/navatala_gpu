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

__kernel void navatala_cfd_attention_value_projection(__global const float* weights, __global const float* v, __global const int* numQ, __global const int* numK, __global const int* headDim, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * hd);
  if ((gid < total)) {
    int qi = (gid / hd);
    int d = (gid % hd);
    float acc = as_float(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int wIdx = ((qi * nk) + j);
      int vIdx = ((j * hd) + d);
      float w = weights[wIdx];
      float vv = v[vIdx];
      acc = (acc + (w * vv));
    }
    _out[gid] = acc;
  }
}
