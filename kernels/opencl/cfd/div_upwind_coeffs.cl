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

__kernel void navatala_cfd_div_upwind_coeffs(__global const float* weights, __global const float* flux, __global const int* counts, __global float* outLower, __global float* outUpper) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float w = weights[((int)((int)(get_global_id(0))))];
    float phi = flux[((int)((int)(get_global_id(0))))];
    float low = ((as_float(0x00000000u) - w) * phi);
    float up = (low + phi);
    outLower[((int)((int)(get_global_id(0))))] = low;
    outUpper[((int)((int)(get_global_id(0))))] = up;
  }
}
