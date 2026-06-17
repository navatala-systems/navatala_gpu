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
__kernel void navatala_dataframe_normalized_mutual_info_f64(__global const double* contingency, __global const double* rowSums, __global const double* colSums, __global const double* n, __global const uint* numTrue, __global const uint* numPred, __global double* nmi) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double miSum[256];
  __local double hySum[256];
  __local double hcSum[256];
  double nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  double localMIAccum64 = as_double(0x0000000000000000ul);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      uint i = (idxU32 / numPredVal);
      uint j = (idxU32 % numPredVal);
      double nij = contingency[idxU32];
      double ai = rowSums[i];
      double bj = colSums[j];
      bool nijGtZero = (nij > as_double(0x0000000000000000ul));
      if (nijGtZero) {
        double pij = (nij / nVal);
        double aibj = (ai * bj);
        double ratio = ((nVal * nij) / aibj);
        double logRatio = log(ratio);
        double termMI = (pij * logRatio);
        double curMI = localMIAccum64;
        double newMI = (curMI + termMI);
        localMIAccum64 = newMI;
      }
    }
  }
  double localMI = localMIAccum64;
  double localHYAccum64 = as_double(0x0000000000000000ul);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double aiHY = rowSums[iU32];
      bool aiGtZero = (aiHY > as_double(0x0000000000000000ul));
      if (aiGtZero) {
        double pi = (aiHY / nVal);
        double logPi = log(pi);
        double termHY = (pi * logPi);
        double curHY = localHYAccum64;
        double newHY = (curHY - termHY);
        localHYAccum64 = newHY;
      }
    }
  }
  double localHY = localHYAccum64;
  double localHCAccum64 = as_double(0x0000000000000000ul);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double bjHC = colSums[jU32];
      bool bjGtZero = (bjHC > as_double(0x0000000000000000ul));
      if (bjGtZero) {
        double pj = (bjHC / nVal);
        double logPj = log(pj);
        double termHC = (pj * logPj);
        double curHC = localHCAccum64;
        double newHC = (curHC - termHC);
        localHCAccum64 = newHC;
      }
    }
  }
  double localHC = localHCAccum64;
  miSum[lid] = localMI;
  hySum[lid] = localHY;
  hcSum[lid] = localHC;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      double otherMI = miSum[(lid + strideU32)];
      double mineMI = miSum[lid];
      miSum[lid] = (mineMI + otherMI);
      double otherHY = hySum[(lid + strideU32)];
      double mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
      double otherHC = hcSum[(lid + strideU32)];
      double mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double miFinal = miSum[(uint)(0u)];
    double hyFinal = hySum[(uint)(0u)];
    double hcFinal = hcSum[(uint)(0u)];
    double hSum = (hyFinal + hcFinal);
    bool hSumIsZero = (hSum == as_double(0x0000000000000000ul));
    double nmiVal = ((hSumIsZero) ? (as_double(0x3ff0000000000000ul)) : (((as_double(0x4000000000000000ul) * miFinal) / hSum)));
    nmi[(uint)(0u)] = nmiVal;
  }
}
