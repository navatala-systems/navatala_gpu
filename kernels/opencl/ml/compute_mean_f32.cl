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

__kernel void navatala_ml_compute_mean_f32(__global const float* y, __global const uint* nSamples, __global float* mean) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint n = nSamples[0];
  bool inBounds = (gid < n);
  float val = y[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint meanF32Stride = (uint)(128u);
  for (int reductionStep5 = 0; reductionStep5 < (int)(8); ++reductionStep5) {
    uint stride5 = meanF32Stride;
    if ((lid < stride5)) {
      float other = sdata[(lid + stride5)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint strideToHalve5 = meanF32Stride;
    uint nextStride5 = (strideToHalve5 >> (uint)(1u));
    meanF32Stride = nextStride5;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float totalSum = sdata[0];
    float nFloat = ((float)(n));
    float meanVal = (totalSum / nFloat);
    mean[0] = meanVal;
  }
}
