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

__kernel void navatala_sparse_chronopoulos_c_g_scalars_f32(__global const float* gamma, __global const float* delta, __global const float* gammaOld, __global float* alpha, __global float* beta) {
  int gid0 = (int)get_global_id(0);
  float g = gamma[0];
  float d = delta[0];
  float go = gammaOld[0];
  float a = (g / d);
  float b = (g / go);
  alpha[0] = a;
  beta[0] = b;
}
