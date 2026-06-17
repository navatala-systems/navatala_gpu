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
extern "C" __global__ void navatala_ml_evaluate_fitness_f64(const double* predictions, const double* targets, const unsigned int* popSize, const unsigned int* nSamples, double* fitness) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int popSizeVal = popSize[0];
  unsigned int nSamplesVal = nSamples[0];
  bool inBounds = (gid < popSizeVal);
  if (inBounds) {
    unsigned int baseIdx = (gid * nSamplesVal);
    double mseAccum = __longlong_as_double(0x0000000000000000ull);
    int nSamplesI32 = ((int)(nSamplesVal));
    for (int j = 0; j < (int)(nSamplesI32); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int predIdx = (baseIdx + jU32);
      double pred = predictions[predIdx];
      double target = targets[jU32];
      double diff = (pred - target);
      double sqErr = (diff * diff);
      double currAccum = mseAccum;
      double newAccum = (currAccum + sqErr);
      mseAccum = newAccum;
    }
    double finalSum = mseAccum;
    double nSamplesF64 = ((double)(nSamplesVal));
    double mse = (finalSum / nSamplesF64);
    fitness[gid] = mse;
  }
}
