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

__kernel void navatala_cfd_pressure_face_flux_correction_corrected(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* gradX, __global const float* gradY, __global const float* gradZ, __global const float* magSf, __global const float* dCorrX, __global const float* dCorrY, __global const float* dCorrZ, __global const float* rAUf, __global const uint* params, __global float* outFaceFlux) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    if (((int)(get_global_id(0)) >= ((int)(params[1])))) {
      outFaceFlux[(int)(get_global_id(0))] = as_float(0x00000000u);
    } else {
      int own = owner[(int)(get_global_id(0))];
      int nei = neighbour[(int)(get_global_id(0))];
      float w = weights[(int)(get_global_id(0))];
      float omw = (as_float(0x3f800000u) - w);
      float gx = ((w * gradX[own]) + (omw * gradX[nei]));
      float gy = ((w * gradY[own]) + (omw * gradY[nei]));
      float gz = ((w * gradZ[own]) + (omw * gradZ[nei]));
      float corr = (((dCorrX[(int)(get_global_id(0))] * gx) + (dCorrY[(int)(get_global_id(0))] * gy)) + (dCorrZ[(int)(get_global_id(0))] * gz));
      outFaceFlux[(int)(get_global_id(0))] = (rAUf[(int)(get_global_id(0))] * (magSf[(int)(get_global_id(0))] * corr));
    }
  }
}
