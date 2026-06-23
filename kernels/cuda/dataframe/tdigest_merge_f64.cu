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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_tdigest_merge_f64(const double* meansA, const double* weightsA, const unsigned int* countA, const double* totalWeightA, const double* minA, const double* maxA, const double* meansB, const double* weightsB, const unsigned int* countB, const double* totalWeightB, const double* minB, const double* maxB, const double* compressionIn, double* meansOut, double* weightsOut, unsigned int* countOut, double* totalWeightOut, double* minOut, double* maxOut, double* compressionOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    unsigned int cA = countA[0];
    unsigned int cB = countB[0];
    double twA = totalWeightA[0];
    double twB = totalWeightB[0];
    double minValA = minA[0];
    double maxValA = maxA[0];
    double minValB = minB[0];
    double maxValB = maxB[0];
    double delta = compressionIn[0];
    double newTw = (twA + twB);
    double newMin = (((minValA < minValB)) ? (minValA) : (minValB));
    double newMax = (((maxValA > maxValB)) ? (maxValA) : (maxValB));
    unsigned int newCount = (cA + cB);
    countOut[0] = newCount;
    totalWeightOut[0] = newTw;
    minOut[0] = newMin;
    maxOut[0] = newMax;
    compressionOut[0] = delta;
  }
}
