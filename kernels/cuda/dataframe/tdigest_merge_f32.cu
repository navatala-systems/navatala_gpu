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
extern "C" __global__ void navatala_dataframe_tdigest_merge_f32(const float* meansA, const float* weightsA, const unsigned int* countA, const float* totalWeightA, const float* minA, const float* maxA, const float* meansB, const float* weightsB, const unsigned int* countB, const float* totalWeightB, const float* minB, const float* maxB, const float* compressionIn, float* meansOut, float* weightsOut, unsigned int* countOut, float* totalWeightOut, float* minOut, float* maxOut, float* compressionOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gid == 0u)) {
    unsigned int cA = countA[0];
    unsigned int cB = countB[0];
    float twA = totalWeightA[0];
    float twB = totalWeightB[0];
    float minValA = minA[0];
    float maxValA = maxA[0];
    float minValB = minB[0];
    float maxValB = maxB[0];
    float delta = compressionIn[0];
    float newTw = (twA + twB);
    float newMin = (((minValA < minValB)) ? (minValA) : (minValB));
    float newMax = (((maxValA > maxValB)) ? (maxValA) : (maxValB));
    unsigned int newCount = (cA + cB);
    countOut[0] = newCount;
    totalWeightOut[0] = newTw;
    minOut[0] = newMin;
    maxOut[0] = newMax;
    compressionOut[0] = delta;
  }
}
