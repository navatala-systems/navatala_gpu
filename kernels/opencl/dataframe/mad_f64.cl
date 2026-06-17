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
__kernel void navatala_dataframe_mad_f64(__global const double* sortedData, __global const uint* n, __global const double* median, __global double* mad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    uint len = n[0];
    double med = median[0];
    uint q1Idx = (len / (uint)(4u));
    uint q3Idx = (len - q1Idx);
    double q1Val = sortedData[q1Idx];
    double q3Val = sortedData[q3Idx];
    double dev1 = fabs((q1Val - med));
    double dev3 = fabs((q3Val - med));
    double madVal = ((dev1 + dev3) / as_double(0x4000000000000000ul));
    mad[0] = madVal;
  }
}
