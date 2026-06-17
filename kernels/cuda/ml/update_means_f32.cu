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
extern "C" __global__ void navatala_ml_update_means_f32(const float* data, const float* responsibilities, const unsigned int* n, const unsigned int* k, const unsigned int* d, float* means) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numK = k[0];
  unsigned int numD = d[0];
  unsigned int kd = (numK * numD);
  bool inBounds = (gid < kd);
  if (inBounds) {
    unsigned int j = (gid / numD);
    unsigned int l = (gid % numD);
    float weightedSumAccum = __uint_as_float(0x00000000u);
    float respSumAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int respIdx = ((iU32 * numK) + j);
      float r_ij = responsibilities[respIdx];
      unsigned int dataIdx = ((iU32 * numD) + l);
      float x_il = data[dataIdx];
      float weighted = (r_ij * x_il);
      float currWeightedSum = weightedSumAccum;
      float newWeightedSum = (currWeightedSum + weighted);
      weightedSumAccum = newWeightedSum;
      float currRespSum = respSumAccum;
      float newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    float weightedSum = weightedSumAccum;
    float respSum = respSumAccum;
    float meanVal = (weightedSum / respSum);
    means[gid] = meanVal;
  }
}
