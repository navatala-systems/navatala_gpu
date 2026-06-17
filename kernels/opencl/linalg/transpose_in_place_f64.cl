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
__kernel void navatala_linalg_transpose_in_place_f64(__global const uint* n, __global double* A) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint dim = n[0];
  uint numSwaps = ((dim * (dim - (uint)(1u))) / (uint)(2u));
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    double gidF = ((double)(gid));
    double dimF = ((double)(dim));
    double twoNm1 = ((as_double(0x4000000000000000ul) * dimF) - as_double(0x3ff0000000000000ul));
    double disc = ((twoNm1 * twoNm1) - (as_double(0x4020000000000000ul) * gidF));
    double sqrtDisc = sqrt(disc);
    double iF = floor(((twoNm1 - sqrtDisc) / as_double(0x4000000000000000ul)));
    uint i = ((uint)(iF));
    uint rowStart = ((i * ((((uint)(2u) * dim) - i) - (uint)(1u))) / (uint)(2u));
    uint offset = (gid - rowStart);
    uint j = ((i + (uint)(1u)) + offset);
    uint idx_ij = ((i * dim) + j);
    uint idx_ji = ((j * dim) + i);
    double val_ij = A[idx_ij];
    double val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}
