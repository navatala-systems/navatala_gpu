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
extern "C" __global__ void navatala_sparse_convergence_divergence_check_f32(const float* residualNorm, const float* prevNorm, const unsigned int* consecutiveIncreases, unsigned int* diverged, unsigned int* newConsecutiveIncreases) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  float norm = residualNorm[0];
  float prev = prevNorm[0];
  unsigned int ci = consecutiveIncreases[0];
  unsigned int increased = (((norm > prev)) ? (1u) : (0u));
  unsigned int newCi = (((increased == 1u)) ? ((ci + 1u)) : (0u));
  unsigned int div = (((newCi >= 3u)) ? (1u) : (0u));
  diverged[0] = div;
  newConsecutiveIncreases[0] = newCi;
}
