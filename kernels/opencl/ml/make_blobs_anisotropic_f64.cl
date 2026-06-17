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
__kernel void navatala_ml_make_blobs_anisotropic_f64(__global const double* centers, __global const double* scales, __global const uint* labels, __global const double* randomNormal, __global const uint* n, __global const uint* d, __global double* data) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  uint numD = d[0];
  uint totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint sampleIdx = (gid / numD);
    uint dimIdx = (gid % numD);
    uint label = labels[sampleIdx];
    uint paramIdx = ((label * numD) + dimIdx);
    double centerVal = centers[paramIdx];
    double scaleVal = scales[paramIdx];
    double randVal = randomNormal[gid];
    double noise = (scaleVal * randVal);
    double result = (centerVal + noise);
    data[gid] = result;
  }
}
