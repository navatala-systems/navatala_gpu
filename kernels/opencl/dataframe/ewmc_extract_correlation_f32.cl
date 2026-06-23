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

__kernel void navatala_dataframe_ewmc_extract_correlation_f32(__global const float* Cxy, __global const float* varX, __global const float* varY, __global const float* nEff, __global float* result) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    float c = Cxy[0];
    float vx = varX[0];
    float vy = varY[0];
    float n = nEff[0];
    float vxvy = (vx * vy);
    float eps = as_float(0x2edbe6ffu);
    float denom = sqrt(vxvy);
    float safeDenom = (((vxvy < eps)) ? (as_float(0x3f800000u)) : (denom));
    float corr = (c / safeDenom);
    float res = (((vxvy < eps)) ? (as_float(0x00000000u)) : (corr));
    result[0] = res;
  }
}
