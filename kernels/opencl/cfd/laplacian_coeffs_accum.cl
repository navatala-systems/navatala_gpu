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

__kernel void navatala_cfd_laplacian_coeffs_accum(__global const float* gamma, __global const float* deltaCoeffs, __global const float* magSf, __global const int* counts, __global float* outUpper, __global float* outLower) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float coeff = (deltaCoeffs[((int)((int)(get_global_id(0))))] * (gamma[((int)((int)(get_global_id(0))))] * magSf[((int)((int)(get_global_id(0))))]));
    float prevUpper = outUpper[((int)((int)(get_global_id(0))))];
    float prevLower = outLower[((int)((int)(get_global_id(0))))];
    outUpper[((int)((int)(get_global_id(0))))] = (prevUpper + coeff);
    outLower[((int)((int)(get_global_id(0))))] = (prevLower + coeff);
  }
}
