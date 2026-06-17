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

__kernel void navatala_dataframe_completeness_score_f32(__global const float* contingency, __global const float* rowSums, __global const float* colSums, __global const float* n, __global const uint* numTrue, __global const uint* numPred, __global float* completeness) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float hcySum[256];
  __local float hcSum[256];
  float nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  float localHCYAccum = as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      uint i = (idxU32 / numPredVal);
      float nij = contingency[idxU32];
      float ai = rowSums[i];
      bool nijGtZero = (nij > as_float(0x00000000u));
      bool aiGtZero = (ai > as_float(0x00000000u));
      bool bothGtZero = (nijGtZero && aiGtZero);
      if (bothGtZero) {
        float pij = (nij / nVal);
        float condProb = (nij / ai);
        float logCondProb = log(condProb);
        float termHCY = (pij * logCondProb);
        float curHCY = localHCYAccum;
        float newHCY = (curHCY - termHCY);
        localHCYAccum = newHCY;
      }
    }
  }
  float localHCY = localHCYAccum;
  float localHCAccumComp = as_float(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float bj = colSums[jU32];
      bool bjGtZero = (bj > as_float(0x00000000u));
      if (bjGtZero) {
        float pj = (bj / nVal);
        float logPj = log(pj);
        float termHC = (pj * logPj);
        float curHC = localHCAccumComp;
        float newHC = (curHC - termHC);
        localHCAccumComp = newHC;
      }
    }
  }
  float localHC = localHCAccumComp;
  hcySum[lid] = localHCY;
  hcSum[lid] = localHC;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      float otherHCY = hcySum[(lid + strideU32)];
      float mineHCY = hcySum[lid];
      hcySum[lid] = (mineHCY + otherHCY);
      float otherHC = hcSum[(lid + strideU32)];
      float mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float hcyFinal = hcySum[(uint)(0u)];
    float hcFinal = hcSum[(uint)(0u)];
    bool hcIsZero = (hcFinal == as_float(0x00000000u));
    float compVal = ((hcIsZero) ? (as_float(0x3f800000u)) : ((as_float(0x3f800000u) - (hcyFinal / hcFinal))));
    completeness[(uint)(0u)] = compVal;
  }
}
