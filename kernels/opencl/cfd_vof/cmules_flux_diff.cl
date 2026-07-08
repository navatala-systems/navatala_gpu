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

__kernel void navatala_cfd_vof_cmules_flux_diff(__global const float* alphaPhiUn, __global const float* alphaPhi10, __global const int* faceCounts, __global float* outPhiCorr) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= faceCounts[0]) {
    return;
  } else {
    outPhiCorr[((int)((int)(get_global_id(0))))] = (alphaPhiUn[((int)((int)(get_global_id(0))))] - alphaPhi10[((int)((int)(get_global_id(0))))]);
  }
}
