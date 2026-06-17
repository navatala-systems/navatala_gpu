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

__kernel void navatala_ml_make_moons_f32(__global const uint* labels, __global const float* randomUniform, __global const float* randomNormal, __global const float* noise, __global const uint* n, __global float* data) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float sigma = noise[0];
    uint label = labels[gid];
    float u = randomUniform[gid];
    float theta = (as_float(0x40490fdbu) * u);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint outIdxX = (gid * (uint)(2u));
    uint outIdxY = (outIdxX + (uint)(1u));
    float noiseX = randomNormal[outIdxX];
    float noiseY = randomNormal[outIdxY];
    bool isLowerMoon = (label == (uint)(1u));
    float baseX_upper = cosTheta;
    float baseY_upper = sinTheta;
    float baseX_lower = (as_float(0x3f800000u) - cosTheta);
    float baseY_lower = (as_float(0x3f000000u) - sinTheta);
    float baseX = ((isLowerMoon) ? (baseX_lower) : (baseX_upper));
    float baseY = ((isLowerMoon) ? (baseY_lower) : (baseY_upper));
    float resultX = (baseX + (sigma * noiseX));
    float resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}
