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
__kernel void navatala_dataframe_adjusted_rand_index_f64(__global const double* contingency, __global const double* rowSums, __global const double* colSums, __global const double* n, __global const uint* numTrue, __global const uint* numPred, __global double* ari) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sumNij[256];
  __local double sumAi[256];
  __local double sumBj[256];
  double nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  double localSumNijAccum64 = as_double(0x0000000000000000ul);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double nij = contingency[idxU32];
      double nijM1 = (nij - as_double(0x3ff0000000000000ul));
      double choose2 = ((nij * nijM1) / as_double(0x4000000000000000ul));
      double curSumNij = localSumNijAccum64;
      double newSumNij = (curSumNij + choose2);
      localSumNijAccum64 = newSumNij;
    }
  }
  double localSumNij = localSumNijAccum64;
  double localSumAiAccum64 = as_double(0x0000000000000000ul);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double ai = rowSums[iU32];
      double aiM1 = (ai - as_double(0x3ff0000000000000ul));
      double choose2Ai = ((ai * aiM1) / as_double(0x4000000000000000ul));
      double curSumAi = localSumAiAccum64;
      double newSumAi = (curSumAi + choose2Ai);
      localSumAiAccum64 = newSumAi;
    }
  }
  double localSumAi = localSumAiAccum64;
  double localSumBjAccum64 = as_double(0x0000000000000000ul);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      double bj = colSums[jU32];
      double bjM1 = (bj - as_double(0x3ff0000000000000ul));
      double choose2Bj = ((bj * bjM1) / as_double(0x4000000000000000ul));
      double curSumBj = localSumBjAccum64;
      double newSumBj = (curSumBj + choose2Bj);
      localSumBjAccum64 = newSumBj;
    }
  }
  double localSumBj = localSumBjAccum64;
  sumNij[lid] = localSumNij;
  sumAi[lid] = localSumAi;
  sumBj[lid] = localSumBj;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      double otherNij = sumNij[(lid + strideU32)];
      double mineNij = sumNij[lid];
      sumNij[lid] = (mineNij + otherNij);
      double otherAi = sumAi[(lid + strideU32)];
      double mineAi = sumAi[lid];
      sumAi[lid] = (mineAi + otherAi);
      double otherBj = sumBj[(lid + strideU32)];
      double mineBj = sumBj[lid];
      sumBj[lid] = (mineBj + otherBj);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double sumNijFinal = sumNij[(uint)(0u)];
    double sumAiFinal = sumAi[(uint)(0u)];
    double sumBjFinal = sumBj[(uint)(0u)];
    double nM1 = (nVal - as_double(0x3ff0000000000000ul));
    double nChoose2 = ((nVal * nM1) / as_double(0x4000000000000000ul));
    double expectedIdx = ((sumAiFinal * sumBjFinal) / nChoose2);
    double maxIdx = ((sumAiFinal + sumBjFinal) / as_double(0x4000000000000000ul));
    double numerator = (sumNijFinal - expectedIdx);
    double denominator = (maxIdx - expectedIdx);
    bool denomIsZero = (denominator == as_double(0x0000000000000000ul));
    double ariVal = ((denomIsZero) ? (as_double(0x3ff0000000000000ul)) : ((numerator / denominator)));
    ari[(uint)(0u)] = ariVal;
  }
}
