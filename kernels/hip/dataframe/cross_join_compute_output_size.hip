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
extern "C" __global__ void navatala_dataframe_cross_join_compute_output_size(const unsigned int* leftCount, const unsigned int* rightCount, unsigned long long* outputSize) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int left = leftCount[0u];
  unsigned int right = rightCount[0u];
  unsigned long long left64 = ((unsigned long long)(left));
  unsigned long long right64 = ((unsigned long long)(right));
  unsigned long long size = (left64 * right64);
  outputSize[0u] = size;
}
