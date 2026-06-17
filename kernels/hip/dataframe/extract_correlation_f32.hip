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
extern "C" __global__ void navatala_dataframe_extract_correlation_f32(const float* Cxy, const float* M2x, const float* M2y, float* correlation) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    float cxyVal = Cxy[0];
    float m2xVal = M2x[0];
    float m2yVal = M2y[0];
    float m2Product = (m2xVal * m2yVal);
    float denom = sqrt(m2Product);
    float cor = (cxyVal / denom);
    correlation[0] = cor;
  }
}
