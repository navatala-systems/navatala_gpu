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

__kernel void navatala_ml_batchnorm_inference_f32(__global const float* x, __global const float* mean, __global const float* var, __global const float* gamma, __global const float* beta, __global const uint* count, __global const float* eps, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    float epsVal = eps[0];
    float d = (x[gid] - mean[gid]);
    float denom = sqrt((var[gid] + epsVal));
    float xn = (d / denom);
    float outF = ((gamma[gid] * xn) + beta[gid]);
    _output[gid] = outF;
  }
}
