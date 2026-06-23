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

__kernel void navatala_cfd_dic_apply(__global const float* r, __global const float* rD, __global const float* upper, __global const int* owner, __global const int* neighbour, __global const int* counts, __global float* z) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    z[c] = (rD[c] * r[c]);
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float zn = z[n];
    z[n] = (zn - ((rD[n] * upper[f]) * z[o]));
  }
  for (int t = 0; t < (int)(counts[2]); ++t) {
    int f2 = ((counts[2] - 1) - t);
    int o2 = owner[f2];
    int n2 = neighbour[f2];
    float zo = z[o2];
    z[o2] = (zo - ((rD[o2] * upper[f2]) * z[n2]));
  }
}
