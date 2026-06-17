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
__kernel void navatala_ml_soft_threshold_f64(__global const double* value, __global const double* threshold, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  bool inBounds = (gid == (uint)(0u));
  if (inBounds) {
    double v = value[0];
    double t = threshold[0];
    double absV = fabs(v);
    double diff = (absV - t);
    double maxDiff = (((diff > as_double(0x0000000000000000ul))) ? (diff) : (as_double(0x0000000000000000ul)));
    bool positive = (v > as_double(0x0000000000000000ul));
    bool negative = (v < as_double(0x0000000000000000ul));
    double signVal = ((positive) ? (as_double(0x3ff0000000000000ul)) : (((negative) ? (as_double(0xbff0000000000000ul)) : (as_double(0x0000000000000000ul)))));
    double res = (signVal * maxDiff);
    result[0] = res;
  }
}
