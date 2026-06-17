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
extern "C" __global__ void navatala_ml_ar_predict_f32(const float* y, const float* phi, const float* intercept, const unsigned int* p, const unsigned int* batch_size, float* predictions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int batchSize = batch_size[0];
  unsigned int pVal = p[0];
  bool inBounds = (gid < batchSize);
  if (inBounds) {
    float pred = intercept[gid];
    float sumAR = pred;
    for (int i = 0; i < (int)(pVal); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int yIdx = ((gid * pVal) + iU32);
      float yVal = y[yIdx];
      unsigned int phiIdx = yIdx;
      float phiVal = phi[phiIdx];
      float term = (phiVal * yVal);
      float currSum = sumAR;
      float newSum = (currSum + term);
      sumAR = newSum;
    }
    float finalPred = sumAR;
    predictions[gid] = finalPred;
  }
}
