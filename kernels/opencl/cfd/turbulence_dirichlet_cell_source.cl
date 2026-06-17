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

__kernel void navatala_cfd_turbulence_dirichlet_cell_source(__global const int* constrainedMask, __global const float* constrainedValue, __global const float* diag, __global const int* nCells, __global float* source) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= nCells[0])) {
    return;
  } else {
    int mask = constrainedMask[((int)((int)(get_global_id(0))))];
    if ((mask == 0)) {
      return;
    } else {
      source[((int)((int)(get_global_id(0))))] = (diag[((int)((int)(get_global_id(0))))] * constrainedValue[((int)((int)(get_global_id(0))))]);
    }
  }
}
