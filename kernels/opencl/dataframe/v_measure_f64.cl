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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_dataframe_v_measure_f64(__global const double* homogeneity, __global const double* completeness, __global double* vMeasure) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    double h = homogeneity[(uint)(0u)];
    double c = completeness[(uint)(0u)];
    double hPlusC = (h + c);
    bool sumIsZero = (hPlusC == as_double(0x0000000000000000ul));
    double hTimesC = (h * c);
    double twoHC = (as_double(0x4000000000000000ul) * hTimesC);
    double vVal = ((sumIsZero) ? (as_double(0x0000000000000000ul)) : ((twoHC / hPlusC)));
    vMeasure[(uint)(0u)] = vVal;
  }
}
