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

__kernel void navatala_dataframe_column_means_f32(__global const float* _input, __global const uint* nSamples, __global const uint* nFeatures, __global float* means) {
  int gid0 = (int)get_global_id(0);
  __local float sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint colIdx = wgid;
  bool colInBounds = (colIdx < d);
  if (colInBounds) {
    float partialSum = as_float(0x00000000u);
    uint rowIdx = lid;
    bool rowInBounds = (rowIdx < n);
    uint elemIdx = ((rowIdx * d) + colIdx);
    if (rowInBounds) {
      float val = _input[elemIdx];
      sdata[lid] = val;
    } else {
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint colMeansF32_reductionStride = (uint)(128u);
    for (int colMeansF32_reductionStep = 0; colMeansF32_reductionStep < (int)(8); ++colMeansF32_reductionStep) {
      uint colMeansF32_stride = colMeansF32_reductionStride;
      if (lid < colMeansF32_stride) {
        float colMeansF32_other = sdata[(lid + colMeansF32_stride)];
        float colMeansF32_mine = sdata[lid];
        float colMeansF32_sum = (colMeansF32_mine + colMeansF32_other);
        sdata[lid] = colMeansF32_sum;
      }
      uint colMeansF32_strideToHalve = colMeansF32_reductionStride;
      uint colMeansF32_nextStride = (colMeansF32_strideToHalve >> (uint)(1u));
      colMeansF32_reductionStride = colMeansF32_nextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == (uint)(0u)) {
      float totalSum = sdata[(uint)(0u)];
      float nFloat = ((float)(n));
      float meanVal = (totalSum / nFloat);
      means[colIdx] = meanVal;
    }
  }
}
