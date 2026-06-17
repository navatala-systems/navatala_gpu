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

__kernel void navatala_dataframe_homogeneity_score_f32(__global const float* contingency, __global const float* rowSums, __global const float* colSums, __global const float* n, __global const uint* numTrue, __global const uint* numPred, __global float* homogeneity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float hycSum[256];
  __local float hySum[256];
  float nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  float localHYCAccum = as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      uint j = (idxU32 % numPredVal);
      float nij = contingency[idxU32];
      float bj = colSums[j];
      bool nijGtZero = (nij > as_float(0x00000000u));
      bool bjGtZero = (bj > as_float(0x00000000u));
      bool bothGtZero = (nijGtZero && bjGtZero);
      if (bothGtZero) {
        float pij = (nij / nVal);
        float condProb = (nij / bj);
        float logCondProb = log(condProb);
        float termHYC = (pij * logCondProb);
        float curHYC = localHYCAccum;
        float newHYC = (curHYC - termHYC);
        localHYCAccum = newHYC;
      }
    }
  }
  float localHYC = localHYCAccum;
  float localHYAccumHom = as_float(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float ai = rowSums[iU32];
      bool aiGtZero = (ai > as_float(0x00000000u));
      if (aiGtZero) {
        float pi = (ai / nVal);
        float logPi = log(pi);
        float termHY = (pi * logPi);
        float curHY = localHYAccumHom;
        float newHY = (curHY - termHY);
        localHYAccumHom = newHY;
      }
    }
  }
  float localHY = localHYAccumHom;
  hycSum[lid] = localHYC;
  hySum[lid] = localHY;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      float otherHYC = hycSum[(lid + strideU32)];
      float mineHYC = hycSum[lid];
      hycSum[lid] = (mineHYC + otherHYC);
      float otherHY = hySum[(lid + strideU32)];
      float mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float hycFinal = hycSum[(uint)(0u)];
    float hyFinal = hySum[(uint)(0u)];
    bool hyIsZero = (hyFinal == as_float(0x00000000u));
    float homoVal = ((hyIsZero) ? (as_float(0x3f800000u)) : ((as_float(0x3f800000u) - (hycFinal / hyFinal))));
    homogeneity[(uint)(0u)] = homoVal;
  }
}
