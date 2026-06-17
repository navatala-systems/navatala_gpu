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

__kernel void navatala_ml_compute_k_l_divergence_f32(__global const float* P, __global const float* Q, __global const uint* nPoints, __global float* kl_divergence) {
  int gid0 = (int)get_global_id(0);
  __local float sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint n = nPoints[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float pij = P[gid];
    float qij = Q[gid];
    float minVal = as_float(0x33d6bf95u);
    bool pSafe = (pij > minVal);
    bool qSafe = (qij > minVal);
    bool bothSafe = (pSafe && qSafe);
    if (bothSafe) {
      float ratio = (pij / qij);
      float logRatio = log(ratio);
      float klTerm = (pij * logRatio);
      sdata[lid] = klTerm;
    } else {
      sdata[lid] = as_float(0x00000000u);
    }
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint klF32ReductionStride = (uint)(128u);
  for (int klF32ReductionStep = 0; klF32ReductionStep < (int)(8); ++klF32ReductionStep) {
    uint klF32Stride = klF32ReductionStride;
    if ((lid < klF32Stride)) {
      float klF32Other = sdata[(lid + klF32Stride)];
      float klF32Mine = sdata[lid];
      float klF32Sum = (klF32Mine + klF32Other);
      sdata[lid] = klF32Sum;
    }
    uint klF32StrideToHalve = klF32ReductionStride;
    uint klF32NextStride = (klF32StrideToHalve >> (uint)(1u));
    klF32ReductionStride = klF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float klVal = sdata[0];
    kl_divergence[0] = klVal;
  }
}
