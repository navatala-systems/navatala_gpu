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
extern "C" __global__ void navatala_transformer_multi_tensor_scale_f32(const float* _input, const float* scale, const unsigned int* totalElements, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((((unsigned int)((int)(blockIdx.x))) * 256u) + ((unsigned int)((int)(threadIdx.x))));
  unsigned int total = totalElements[0u];
  float s = scale[0u];
  bool valid = (gid < total);
  if (valid) {
    float x = _input[gid];
    float result = (x * s);
    _output[gid] = result;
  }
}
