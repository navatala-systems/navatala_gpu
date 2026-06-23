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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_sparse_build_smoothed_prolongation_f64(const unsigned int* ArowPtr, const unsigned int* AcolIdx, const double* Avalues, const double* diagInv, const unsigned int* ProwPtr, const unsigned int* PcolIdx, const double* Pvalues, const double* omega, const unsigned int* nRows, double* PsmoothValues) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    unsigned int myAgg = PcolIdx[gid];
    double myPval = Pvalues[gid];
    int aStart = ((int)(ArowPtr[gid]));
    int aEnd = ((int)(ArowPtr[(gid + 1)]));
    double sumAP = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)((aEnd - aStart)); ++k) {
      int idx = (aStart + k);
      int col = ((int)(AcolIdx[idx]));
      double aVal = Avalues[idx];
      unsigned int colAgg = PcolIdx[col];
      if (colAgg == myAgg) {
        sumAP = (sumAP + aVal);
      }
    }
    double om = omega[0];
    double di = diagInv[gid];
    double correction = (om * (di * sumAP));
    PsmoothValues[gid] = (myPval - correction);
  }
}
