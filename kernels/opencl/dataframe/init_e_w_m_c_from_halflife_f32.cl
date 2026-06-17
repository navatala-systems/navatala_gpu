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

__kernel void navatala_dataframe_init_e_w_m_c_from_halflife_f32(__global const float* halflife, __global float* alpha, __global float* nEff, __global float* meanX, __global float* meanY, __global float* Cxy, __global float* varX, __global float* varY) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    float h = halflife[0];
    float ln2 = as_float(0x3f317215u);
    float a = (ln2 / h);
    alpha[0] = a;
    nEff[0] = as_float(0x00000000u);
    meanX[0] = as_float(0x00000000u);
    meanY[0] = as_float(0x00000000u);
    Cxy[0] = as_float(0x00000000u);
    varX[0] = as_float(0x00000000u);
    varY[0] = as_float(0x00000000u);
  }
}
