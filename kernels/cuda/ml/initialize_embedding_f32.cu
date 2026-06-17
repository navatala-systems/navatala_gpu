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
extern "C" __global__ void navatala_ml_initialize_embedding_f32(const unsigned int* seed, const unsigned int* n_samples, const unsigned int* n_components, const float* spread, float* embedding) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nSamples = n_samples[0];
  unsigned int nComp = n_components[0];
  unsigned int totalSize = (nSamples * nComp);
  unsigned int seedVal = seed[0];
  float spreadVal = spread[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    unsigned int state = (seedVal + gid);
    unsigned int a = 1664525u;
    unsigned int c = 1013904223u;
    unsigned int newState = ((a * state) + c);
    float randBitsF = ((float)(newState));
    float maxU32 = __uint_as_float(0x4f800000u);
    float normalized = (randBitsF / maxU32);
    float centered = ((normalized * __uint_as_float(0x40000000u)) - __uint_as_float(0x3f800000u));
    float scaled = (centered * spreadVal);
    embedding[gid] = scaled;
  }
}
