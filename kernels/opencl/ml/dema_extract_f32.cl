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

__kernel void navatala_ml_dema_extract_f32(__global const float* ema1, __global const float* ema2, __global float* result) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    float e1 = ema1[0];
    float e2 = ema2[0];
    float t2 = (as_float(0x40000000u) * e1);
    float d = (t2 - e2);
    result[0] = d;
  }
}
