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
extern "C" __global__ void navatala_ml_optimize_layout_f32(const float* embedding, const float* gradients, const unsigned int* n_samples, const unsigned int* n_components, const float* alpha, float* embedding_out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nComp = n_components[0];
  unsigned int totalSize = (nSamples * nComp);
  float alphaVal = alpha[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    float embVal = embedding[gid];
    float gradVal = gradients[gid];
    float gradClipped = (((gradVal > __uint_as_float(0x40800000u))) ? (__uint_as_float(0x40800000u)) : ((((gradVal < __uint_as_float(0xc0800000u))) ? (__uint_as_float(0xc0800000u)) : (gradVal))));
    float update = (alphaVal * gradClipped);
    float newVal = (embVal + update);
    embedding_out[gid] = newVal;
  }
}
