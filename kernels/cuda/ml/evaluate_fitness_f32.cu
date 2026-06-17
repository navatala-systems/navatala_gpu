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
extern "C" __global__ void navatala_ml_evaluate_fitness_f32(const float* predictions, const float* targets, const unsigned int* popSize, const unsigned int* nSamples, float* fitness) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int popSizeVal = popSize[0];
  unsigned int nSamplesVal = nSamples[0];
  bool inBounds = (gid < popSizeVal);
  if (inBounds) {
    unsigned int baseIdx = (gid * nSamplesVal);
    float mseAccum = __uint_as_float(0x00000000u);
    int nSamplesI32 = ((int)(nSamplesVal));
    for (int j = 0; j < (int)(nSamplesI32); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int predIdx = (baseIdx + jU32);
      float pred = predictions[predIdx];
      float target = targets[jU32];
      float diff = (pred - target);
      float sqErr = (diff * diff);
      float currAccum = mseAccum;
      float newAccum = (currAccum + sqErr);
      mseAccum = newAccum;
    }
    float finalSum = mseAccum;
    float nSamplesF32 = ((float)(nSamplesVal));
    float mse = (finalSum / nSamplesF32);
    fitness[gid] = mse;
  }
}
