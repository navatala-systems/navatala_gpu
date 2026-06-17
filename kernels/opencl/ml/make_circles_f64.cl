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
__kernel void navatala_ml_make_circles_f64(__global const uint* labels, __global const double* randomUniform, __global const double* randomNormal, __global const double* noise, __global const double* factor, __global const uint* n, __global double* data) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    double sigma = noise[0];
    double innerFactor = factor[0];
    uint label = labels[gid];
    double u = randomUniform[gid];
    double theta = (as_double(0x401921fb54442d18ul) * u);
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);
    uint outIdxX = (gid * (uint)(2u));
    uint outIdxY = (outIdxX + (uint)(1u));
    double noiseX = randomNormal[outIdxX];
    double noiseY = randomNormal[outIdxY];
    bool isOuterCircle = (label == (uint)(1u));
    double radius = ((isOuterCircle) ? (as_double(0x3ff0000000000000ul)) : (innerFactor));
    double baseX = (radius * cosTheta);
    double baseY = (radius * sinTheta);
    double resultX = (baseX + (sigma * noiseX));
    double resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}
