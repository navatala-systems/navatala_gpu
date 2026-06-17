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

__kernel void navatala_ml_explained_variance_ratio_f32(__global const float* eigenvalues, __global const uint* nFeatures, __global const uint* nComponents, __global float* ratios) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  bool inBoundsLoad = (gid < d);
  if (inBoundsLoad) {
    float eigVal = eigenvalues[gid];
    sdata[lid] = eigVal;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint varRatioF32RedStride = (uint)(128u);
  for (int varRatioF32RedStep = 0; varRatioF32RedStep < (int)(8); ++varRatioF32RedStep) {
    uint varRatioF32Stride = varRatioF32RedStride;
    if ((lid < varRatioF32Stride)) {
      float other = sdata[(lid + varRatioF32Stride)];
      float mine = sdata[lid];
      float sumVal = (mine + other);
      sdata[lid] = sumVal;
    }
    uint varRatioF32StrideToHalve = varRatioF32RedStride;
    uint varRatioF32NextStride = (varRatioF32StrideToHalve >> (uint)(1u));
    varRatioF32RedStride = varRatioF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float totalVariance = sdata[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  bool inBoundsRatio = (gid < kComp);
  if (inBoundsRatio) {
    float eigVal = eigenvalues[gid];
    float ratio = (eigVal / totalVariance);
    ratios[gid] = ratio;
  }
}
