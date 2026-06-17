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

__kernel void navatala_dataframe_v_measure_f32(__global const float* homogeneity, __global const float* completeness, __global float* vMeasure) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    float h = homogeneity[(uint)(0u)];
    float c = completeness[(uint)(0u)];
    float hPlusC = (h + c);
    bool sumIsZero = (hPlusC == as_float(0x00000000u));
    float hTimesC = (h * c);
    float twoHC = (as_float(0x40000000u) * hTimesC);
    float vVal = ((sumIsZero) ? (as_float(0x00000000u)) : ((twoHC / hPlusC)));
    vMeasure[(uint)(0u)] = vVal;
  }
}
