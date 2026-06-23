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
__kernel void navatala_dataframe_homogeneity_score_f64(__global const double* contingency, __global const double* rowSums, __global const double* colSums, __global const double* n, __global const uint* numTrue, __global const uint* numPred, __global double* homogeneity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double hycSum[256];
  __local double hySum[256];
  double nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  double localHYCAccum64 = as_double(0x0000000000000000ul);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      uint j = (idxU32 % numPredVal);
      double nij = contingency[idxU32];
      double bj = colSums[j];
      bool nijGtZero = (nij > as_double(0x0000000000000000ul));
      bool bjGtZero = (bj > as_double(0x0000000000000000ul));
      bool bothGtZero = (nijGtZero && bjGtZero);
      if (bothGtZero) {
        double pij = (nij / nVal);
        double condProb = (nij / bj);
        double logCondProb = log(condProb);
        double termHYC = (pij * logCondProb);
        double curHYC = localHYCAccum64;
        double newHYC = (curHYC - termHYC);
        localHYCAccum64 = newHYC;
      }
    }
  }
  double localHYC = localHYCAccum64;
  double localHYAccumHom64 = as_double(0x0000000000000000ul);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double ai = rowSums[iU32];
      bool aiGtZero = (ai > as_double(0x0000000000000000ul));
      if (aiGtZero) {
        double pi = (ai / nVal);
        double logPi = log(pi);
        double termHY = (pi * logPi);
        double curHY = localHYAccumHom64;
        double newHY = (curHY - termHY);
        localHYAccumHom64 = newHY;
      }
    }
  }
  double localHY = localHYAccumHom64;
  hycSum[lid] = localHYC;
  hySum[lid] = localHY;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if (lid < strideU32) {
      double otherHYC = hycSum[(lid + strideU32)];
      double mineHYC = hycSum[lid];
      hycSum[lid] = (mineHYC + otherHYC);
      double otherHY = hySum[(lid + strideU32)];
      double mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    double hycFinal = hycSum[(uint)(0u)];
    double hyFinal = hySum[(uint)(0u)];
    bool hyIsZero = (hyFinal == as_double(0x0000000000000000ul));
    double homoVal = ((hyIsZero) ? (as_double(0x3ff0000000000000ul)) : ((as_double(0x3ff0000000000000ul) - (hycFinal / hyFinal))));
    homogeneity[(uint)(0u)] = homoVal;
  }
}
