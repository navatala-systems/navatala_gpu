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

__kernel void navatala_ml_compute_xt_y_f32(__global const float* X, __global const float* y, __global const uint* nSamples, __global const uint* nFeatures, __global float* Xty) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local float sdata[256];
  uint n = nSamples[0];
  uint p = nFeatures[0];
  uint featureIdx = wgid;
  bool featureInBounds = (featureIdx < p);
  if (featureInBounds) {
    uint sampleIdx = lid;
    bool sampleInBounds = (sampleIdx < n);
    uint xIdx = ((sampleIdx * p) + featureIdx);
    if (sampleInBounds) {
      float xVal = X[xIdx];
      float yVal = y[sampleIdx];
      float product = (xVal * yVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint xtyF32Stride = (uint)(128u);
    for (int reductionStep1 = 0; reductionStep1 < (int)(8); ++reductionStep1) {
      uint stride1 = xtyF32Stride;
      if (lid < stride1) {
        float other = sdata[(lid + stride1)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint strideToHalve1 = xtyF32Stride;
      uint nextStride1 = (strideToHalve1 >> (uint)(1u));
      xtyF32Stride = nextStride1;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == (uint)(0u)) {
      float totalSum = sdata[0];
      Xty[featureIdx] = totalSum;
    }
  }
}
