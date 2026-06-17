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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_ml_batchnorm_inference_f16(__global const half* x, __global const half* mean, __global const half* var, __global const half* gamma, __global const half* beta, __global const uint* count, __global const float* eps, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if ((gid < countVal)) {
    float epsVal = eps[0];
    float d = (((float)(x[gid])) - ((float)(mean[gid])));
    float denom = sqrt((((float)(var[gid])) + epsVal));
    float xn = (d / denom);
    float outF = ((((float)(gamma[gid])) * xn) + ((float)(beta[gid])));
    _output[gid] = ((half)(outF));
  }
}
