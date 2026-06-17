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
extern "C" __global__ void navatala_ml_knn_classify_weighted_f32(const unsigned int* neighborIndices, const float* neighborDistances, const unsigned int* labels, const unsigned int* q, const unsigned int* k, const unsigned int* numClasses, unsigned int* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int qVal = q[0];
  unsigned int kVal = k[0];
  unsigned int nClasses = numClasses[0];
  unsigned int queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    unsigned int baseIdx = (queryIdx * kVal);
    unsigned int bestClass = 0u;
    float bestWeight = __uint_as_float(0x00000000u);
    float epsilon = __uint_as_float(0x2edbe6ffu);
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      unsigned int classIdxU32 = ((unsigned int)(classIdx));
      float classWeight = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(kVal); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        unsigned int neighborIdx = (baseIdx + jU32);
        unsigned int neighborDataIdx = neighborIndices[neighborIdx];
        unsigned int neighborLabel = labels[neighborDataIdx];
        float dist = neighborDistances[neighborIdx];
        float weight = (__uint_as_float(0x3f800000u) / (dist + epsilon));
        bool isMatch = (neighborLabel == classIdxU32);
        float addWeight = ((isMatch) ? (weight) : (__uint_as_float(0x00000000u)));
        float currentWeight = classWeight;
        float newClassWeight = (currentWeight + addWeight);
        classWeight = newClassWeight;
      }
      float finalClassWeight = classWeight;
      float currentBestWeight = bestWeight;
      unsigned int currentBestClass = bestClass;
      bool isBetter = (finalClassWeight > currentBestWeight);
      unsigned int newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      float newBestWeight = ((isBetter) ? (finalClassWeight) : (currentBestWeight));
      bestClass = newBestClass;
      bestWeight = newBestWeight;
    }
    unsigned int finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}
