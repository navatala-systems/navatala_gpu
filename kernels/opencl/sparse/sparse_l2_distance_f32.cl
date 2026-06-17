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

__kernel void navatala_sparse_sparse_l2_distance_f32(__global const float* a, __global const float* b, __global const uint* nnz, __global float* result) {
  int gid0 = (int)get_global_id(0);
  __local float sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lidU32 = ((uint)((int)(get_local_id(0))));
  uint nnzVal = nnz[(uint)(0u)];
  float localSumSqAccum = as_float(0x00000000u);
  uint idxAccum = gid;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if ((currentIdx < nnzVal)) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      float diff = (aVal - bVal);
      float diffSq = (diff * diff);
      float currentSumSq = localSumSqAccum;
      float newSumSq = (currentSumSq + diffSq);
      localSumSqAccum = newSumSq;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  float finalLocalSumSq = localSumSqAccum;
  sdata[lidU32] = finalLocalSumSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint strideAccum = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if ((lidU32 < currentStride)) {
      uint otherIdx = (lidU32 + currentStride);
      float myVal = sdata[lidU32];
      float otherVal = sdata[otherIdx];
      sdata[lidU32] = (myVal + otherVal);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> (uint)(1u));
    strideAccum = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lidU32 == (uint)(0u))) {
    float sumSq = sdata[(uint)(0u)];
    float dist = sqrt(sumSq);
    result[(uint)(0u)] = dist;
  }
}
