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
__kernel void navatala_dataframe_column_means_f64(__global const double* _input, __global const uint* nSamples, __global const uint* nFeatures, __global double* means) {
  int gid0 = (int)get_global_id(0);
  __local double sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint colIdx = wgid;
  bool colInBounds = (colIdx < d);
  if (colInBounds) {
    double partialSum = as_double(0x0000000000000000ul);
    uint rowIdx = lid;
    bool rowInBounds = (rowIdx < n);
    uint elemIdx = ((rowIdx * d) + colIdx);
    if (rowInBounds) {
      double val = _input[elemIdx];
      sdata[lid] = val;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint colMeansF64_reductionStride = (uint)(128u);
    for (int colMeansF64_reductionStep = 0; colMeansF64_reductionStep < (int)(8); ++colMeansF64_reductionStep) {
      uint colMeansF64_stride = colMeansF64_reductionStride;
      if ((lid < colMeansF64_stride)) {
        double colMeansF64_other = sdata[(lid + colMeansF64_stride)];
        double colMeansF64_mine = sdata[lid];
        double colMeansF64_sum = (colMeansF64_mine + colMeansF64_other);
        sdata[lid] = colMeansF64_sum;
      }
      uint colMeansF64_strideToHalve = colMeansF64_reductionStride;
      uint colMeansF64_nextStride = (colMeansF64_strideToHalve >> (uint)(1u));
      colMeansF64_reductionStride = colMeansF64_nextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      double totalSum = sdata[(uint)(0u)];
      double nFloat = ((double)(n));
      double meanVal = (totalSum / nFloat);
      means[colIdx] = meanVal;
    }
  }
}
