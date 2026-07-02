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

__kernel void navatala_cfd_face_flux(__global const float* x, __global const int* owner, __global const int* neighbour, __global const float* cf, __global const float* bcVal, __global const int* bcMask, __global const int* counts, __global float* outFlux) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float xo = x[o];
    float other = xo;
    if (((int)((int)(get_global_id(0)))) < counts[2]) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      other = x[n];
    } else {
      int m = bcMask[((int)((int)(get_global_id(0))))];
      if ((m == 1) || (m == 3)) {
        other = bcVal[((int)((int)(get_global_id(0))))];
      } else {
        if (m == 2) {
          other = as_float(0x00000000u);
        } else {
          if (m == 4) {
            outFlux[((int)((int)(get_global_id(0))))] = bcVal[((int)((int)(get_global_id(0))))];
            return;
          }
        }
      }
    }
    outFlux[((int)((int)(get_global_id(0))))] = (cf[((int)((int)(get_global_id(0))))] * (other - xo));
  }
}
