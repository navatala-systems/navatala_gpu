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

__kernel void navatala_cfd_scalar_jacobi_update(__global const float* ax, __global const float* diag, __global const float* rhs, __global const float* params, __global const uint* counts, __global float* x) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[(int)(get_global_id(0))];
    float absD = fabs(d);
    if ((absD < smallDiag)) {
      if ((d >= as_float(0x00000000u))) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float xi = x[(int)(get_global_id(0))];
    float correction = (omega * ((rhs[(int)(get_global_id(0))] - ax[(int)(get_global_id(0))]) / d));
    x[(int)(get_global_id(0))] = (xi + correction);
  }
}
