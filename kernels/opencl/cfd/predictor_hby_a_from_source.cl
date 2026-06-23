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

__kernel void navatala_cfd_predictor_hby_a_from_source(__global const float* rAU, __global const float* sourceX, __global const float* sourceY, __global const float* sourceZ, __global const int* counts1, __global float* outHbyAX, __global float* outHbyAY, __global float* outHbyAZ) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= counts1[0]) {
    return;
  } else {
    float r = rAU[((int)((int)(get_global_id(0))))];
    outHbyAX[((int)((int)(get_global_id(0))))] = (r * sourceX[((int)((int)(get_global_id(0))))]);
    outHbyAY[((int)((int)(get_global_id(0))))] = (r * sourceY[((int)((int)(get_global_id(0))))]);
    outHbyAZ[((int)((int)(get_global_id(0))))] = (r * sourceZ[((int)((int)(get_global_id(0))))]);
  }
}
