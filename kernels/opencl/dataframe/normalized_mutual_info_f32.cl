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

__kernel void navatala_dataframe_normalized_mutual_info_f32(__global const float* contingency, __global const float* rowSums, __global const float* colSums, __global const float* n, __global const uint* numTrue, __global const uint* numPred, __global float* nmi) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float miSum[256];
  __local float hySum[256];
  __local float hcSum[256];
  float nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  float localMIAccum = as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      uint i = (idxU32 / numPredVal);
      uint j = (idxU32 % numPredVal);
      float nij = contingency[idxU32];
      float ai = rowSums[i];
      float bj = colSums[j];
      bool nijGtZero = (nij > as_float(0x00000000u));
      if (nijGtZero) {
        float pij = (nij / nVal);
        float aibj = (ai * bj);
        float ratio = ((nVal * nij) / aibj);
        float logRatio = log(ratio);
        float termMI = (pij * logRatio);
        float curMI = localMIAccum;
        float newMI = (curMI + termMI);
        localMIAccum = newMI;
      }
    }
  }
  float localMI = localMIAccum;
  float localHYAccum = as_float(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float aiHY = rowSums[iU32];
      bool aiGtZero = (aiHY > as_float(0x00000000u));
      if (aiGtZero) {
        float pi = (aiHY / nVal);
        float logPi = log(pi);
        float termHY = (pi * logPi);
        float curHY = localHYAccum;
        float newHY = (curHY - termHY);
        localHYAccum = newHY;
      }
    }
  }
  float localHY = localHYAccum;
  float localHCAccum = as_float(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float bjHC = colSums[jU32];
      bool bjGtZero = (bjHC > as_float(0x00000000u));
      if (bjGtZero) {
        float pj = (bjHC / nVal);
        float logPj = log(pj);
        float termHC = (pj * logPj);
        float curHC = localHCAccum;
        float newHC = (curHC - termHC);
        localHCAccum = newHC;
      }
    }
  }
  float localHC = localHCAccum;
  miSum[lid] = localMI;
  hySum[lid] = localHY;
  hcSum[lid] = localHC;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      float otherMI = miSum[(lid + strideU32)];
      float mineMI = miSum[lid];
      miSum[lid] = (mineMI + otherMI);
      float otherHY = hySum[(lid + strideU32)];
      float mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
      float otherHC = hcSum[(lid + strideU32)];
      float mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float miFinal = miSum[(uint)(0u)];
    float hyFinal = hySum[(uint)(0u)];
    float hcFinal = hcSum[(uint)(0u)];
    float hSum = (hyFinal + hcFinal);
    bool hSumIsZero = (hSum == as_float(0x00000000u));
    float nmiVal = ((hSumIsZero) ? (as_float(0x3f800000u)) : (((as_float(0x40000000u) * miFinal) / hSum)));
    nmi[(uint)(0u)] = nmiVal;
  }
}
