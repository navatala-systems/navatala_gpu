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

__kernel void navatala_cfd_add_vol_vector(__global const float* ax, __global const float* ay, __global const float* az, __global const float* bx, __global const float* by_, __global const float* bz, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    outX[((int)((int)(get_global_id(0))))] = (ax[((int)((int)(get_global_id(0))))] + bx[((int)((int)(get_global_id(0))))]);
    outY[((int)((int)(get_global_id(0))))] = (ay[((int)((int)(get_global_id(0))))] + by_[((int)((int)(get_global_id(0))))]);
    outZ[((int)((int)(get_global_id(0))))] = (az[((int)((int)(get_global_id(0))))] + bz[((int)((int)(get_global_id(0))))]);
  }
}
