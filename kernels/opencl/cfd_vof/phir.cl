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

__kernel void navatala_cfd_vof_phir(__global const float* gx, __global const float* gy, __global const float* gz, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* phi, __global const float* magSf, __global const int* owner, __global const int* nei, __global const int* faceParams, __global const float* paramsF, __global float* outPhir) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= faceParams[0])) {
    return;
  } else {
    if ((paramsF[0] == as_float(0x00000000u))) {
      outPhir[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      return;
    }
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float gox = gx[o];
    float goy = gy[o];
    float goz = gz[o];
    float gnx = gx[n];
    float gny = gy[n];
    float gnz = gz[n];
    float gfx = (as_float(0x3f000000u) * (gox + gnx));
    float gfy = (as_float(0x3f000000u) * (goy + gny));
    float gfz = (as_float(0x3f000000u) * (goz + gnz));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float dotSf = (((gfx * sfX[((int)((int)(get_global_id(0))))]) + (gfy * sfY[((int)((int)(get_global_id(0))))])) + (gfz * sfZ[((int)((int)(get_global_id(0))))]));
    float nHatf = (dotSf / (magg + paramsF[1]));
    float phic = ((paramsF[0] * fabs(phi[((int)((int)(get_global_id(0))))])) / magSf[((int)((int)(get_global_id(0))))]);
    outPhir[((int)((int)(get_global_id(0))))] = (phic * nHatf);
  }
}
