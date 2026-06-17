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
__kernel void navatala_sparse_sparse_threshold_f64(__global const double* valuesA, __global const double* threshold, __global const int* nnz, __global double* valuesB) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  double thresh = threshold[(uint)(0u)];
  if ((i < nnz[(uint)(0u)])) {
    double a = valuesA[i];
    double absA = fabs(a);
    double result = (((absA >= thresh)) ? (a) : (as_double(0x0000000000000000ul)));
    valuesB[i] = result;
  }
}
