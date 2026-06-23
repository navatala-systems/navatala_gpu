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
__kernel void navatala_sparse_sparse_l2_distance_f64(__global const double* a, __global const double* b, __global const uint* nnz, __global double* result) {
  int gid0 = (int)get_global_id(0);
  __local double sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lidU32 = ((uint)((int)(get_local_id(0))));
  uint nnzVal = nnz[(uint)(0u)];
  double localSumSqAccum = as_double(0x0000000000000000ul);
  uint idxAccum = gid;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      double aVal = a[currentIdx];
      double bVal = b[currentIdx];
      double diff = (aVal - bVal);
      double diffSq = (diff * diff);
      double currentSumSq = localSumSqAccum;
      double newSumSq = (currentSumSq + diffSq);
      localSumSqAccum = newSumSq;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  double finalLocalSumSq = localSumSqAccum;
  sdata[lidU32] = finalLocalSumSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint strideAccum = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if (lidU32 < currentStride) {
      uint otherIdx = (lidU32 + currentStride);
      double myVal = sdata[lidU32];
      double otherVal = sdata[otherIdx];
      sdata[lidU32] = (myVal + otherVal);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> (uint)(1u));
    strideAccum = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lidU32 == (uint)(0u)) {
    double sumSq = sdata[(uint)(0u)];
    double dist = sqrt(sumSq);
    result[(uint)(0u)] = dist;
  }
}
