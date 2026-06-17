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
__kernel void navatala_dataframe_completeness_score_f64(__global const double* contingency, __global const double* rowSums, __global const double* colSums, __global const double* n, __global const uint* numTrue, __global const uint* numPred, __global double* completeness) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double hcySum[256];
  __local double hcSum[256];
  double nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  double localHCYAccum64 = as_double(0x0000000000000000ul);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      uint i = (idxU32 / numPredVal);
      double nij = contingency[idxU32];
      double ai = rowSums[i];
      bool nijGtZero = (nij > as_double(0x0000000000000000ul));
      bool aiGtZero = (ai > as_double(0x0000000000000000ul));
      bool bothGtZero = (nijGtZero && aiGtZero);
      if (bothGtZero) {
        double pij = (nij / nVal);
        double condProb = (nij / ai);
        double logCondProb = log(condProb);
        double termHCY = (pij * logCondProb);
        double curHCY = localHCYAccum64;
        double newHCY = (curHCY - termHCY);
        localHCYAccum64 = newHCY;
      }
    }
  }
  double localHCY = localHCYAccum64;
  double localHCAccumComp64 = as_double(0x0000000000000000ul);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double bj = colSums[jU32];
      bool bjGtZero = (bj > as_double(0x0000000000000000ul));
      if (bjGtZero) {
        double pj = (bj / nVal);
        double logPj = log(pj);
        double termHC = (pj * logPj);
        double curHC = localHCAccumComp64;
        double newHC = (curHC - termHC);
        localHCAccumComp64 = newHC;
      }
    }
  }
  double localHC = localHCAccumComp64;
  hcySum[lid] = localHCY;
  hcSum[lid] = localHC;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      double otherHCY = hcySum[(lid + strideU32)];
      double mineHCY = hcySum[lid];
      hcySum[lid] = (mineHCY + otherHCY);
      double otherHC = hcSum[(lid + strideU32)];
      double mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double hcyFinal = hcySum[(uint)(0u)];
    double hcFinal = hcSum[(uint)(0u)];
    bool hcIsZero = (hcFinal == as_double(0x0000000000000000ul));
    double compVal = ((hcIsZero) ? (as_double(0x3ff0000000000000ul)) : ((as_double(0x3ff0000000000000ul) - (hcyFinal / hcFinal))));
    completeness[(uint)(0u)] = compVal;
  }
}
