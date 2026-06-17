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
extern "C" __global__ void navatala_ml_differencing_f32(const float* y, const unsigned int* n_obs, const unsigned int* batch_size, float* y_diff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nObs = n_obs[0];
  unsigned int batchSize = batch_size[0];
  unsigned int nDiff = (nObs - 1u);
  unsigned int totalOutputs = (nDiff * batchSize);
  bool inBounds = (gid < totalOutputs);
  if (inBounds) {
    unsigned int batch = (gid / nDiff);
    unsigned int t = (gid % nDiff);
    unsigned int currIdx = ((batch * nObs) + (t + 1u));
    unsigned int prevIdx = ((batch * nObs) + t);
    float currVal = y[currIdx];
    float prevVal = y[prevIdx];
    float diff = (currVal - prevVal);
    y_diff[gid] = diff;
  }
}
