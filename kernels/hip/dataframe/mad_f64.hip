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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_mad_f64(const double* sortedData, const unsigned int* n, const double* median, double* mad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    unsigned int len = n[0];
    double med = median[0];
    unsigned int q1Idx = (len / 4u);
    unsigned int q3Idx = (len - q1Idx);
    double q1Val = sortedData[q1Idx];
    double q3Val = sortedData[q3Idx];
    double dev1 = abs((q1Val - med));
    double dev3 = abs((q3Val - med));
    double madVal = ((dev1 + dev3) / __longlong_as_double(0x4000000000000000ull));
    mad[0] = madVal;
  }
}
