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

__kernel void navatala_cfd_k_omega_s_s_t_nut_compute(__global const float* kVals, __global const float* omegaVals, __global const float* f23Vals, __global const float* s2Vals, __global const uint* counts, __global const float* params, __global float* outNut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    float a1 = params[0];
    float b1 = params[1];
    float k = kVals[(int)(get_global_id(0))];
    float om = omegaVals[(int)(get_global_id(0))];
    float f23 = f23Vals[(int)(get_global_id(0))];
    float s2raw = s2Vals[(int)(get_global_id(0))];
    float denomA = (a1 * om);
    float denomB = ((b1 * f23) * sqrt((s2raw * ((float)((s2raw > as_float(0x00000000u)))))));
    float num = (a1 * k);
    if ((denomA > denomB)) {
      if ((denomA > as_float(0x00000000u))) {
        outNut[(int)(get_global_id(0))] = (num / denomA);
      } else {
        outNut[(int)(get_global_id(0))] = as_float(0x00000000u);
      }
    } else {
      if ((denomB > as_float(0x00000000u))) {
        outNut[(int)(get_global_id(0))] = (num / denomB);
      } else {
        outNut[(int)(get_global_id(0))] = as_float(0x00000000u);
      }
    }
  }
}
