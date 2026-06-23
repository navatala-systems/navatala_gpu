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
__kernel void navatala_dataframe_median_f64(__global const double* sortedInput, __global const uint* count, __global double* median) {
  int gid0 = (int)get_global_id(0);
  uint gidU32 = ((uint)((int)(get_global_id(0))));
  if (gidU32 == (uint)(0u)) {
    uint n = count[(uint)(0u)];
    uint nMinus1 = (n - (uint)(1u));
    double nMinus1Float = ((double)(nMinus1));
    double idx = (as_double(0x3fe0000000000000ul) * nMinus1Float);
    double lowerFloat = floor(idx);
    uint lower = ((uint)(lowerFloat));
    uint upperCandidate = (lower + (uint)(1u));
    uint upper = (((upperCandidate < n)) ? (upperCandidate) : (nMinus1));
    double frac = (idx - lowerFloat);
    double valLower = sortedInput[lower];
    double valUpper = sortedInput[upper];
    double oneMinusFrac = (as_double(0x3ff0000000000000ul) - frac);
    double termLower = (oneMinusFrac * valLower);
    double termUpper = (frac * valUpper);
    double medianVal = (termLower + termUpper);
    median[(uint)(0u)] = medianVal;
  }
}
