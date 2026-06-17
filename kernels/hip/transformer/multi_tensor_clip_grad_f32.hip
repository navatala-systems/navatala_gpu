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
extern "C" __global__ void navatala_transformer_multi_tensor_clip_grad_f32(const float* totalNorm, const float* maxNorm, const unsigned int* totalElements, float* grad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float tNorm = totalNorm[0u];
  float mNorm = maxNorm[0u];
  float epsilon = __uint_as_float(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (__uint_as_float(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    float g = grad[gid];
    float clipped = (g * clipCoef);
    grad[gid] = clipped;
  }
}
