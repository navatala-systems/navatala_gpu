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
__kernel void navatala_sparse_amg_level_metrics_f64(__global const uint* levelNRows, __global const uint* levelNnz, __global const uint* nLevels, __global double* gridComplexity, __global double* operatorComplexity) {
  int gid0 = (int)get_global_id(0);
  int nL = ((int)(nLevels[0]));
  double rows0 = ((double)(levelNRows[0]));
  double nnz0 = ((double)(levelNnz[0]));
  double totalRows = as_double(0x0000000000000000ul);
  double totalNnz = as_double(0x0000000000000000ul);
  for (int i = 0; i < (int)(nL); ++i) {
    double ri = ((double)(levelNRows[i]));
    double ni = ((double)(levelNnz[i]));
    totalRows = (totalRows + ri);
    totalNnz = (totalNnz + ni);
  }
  gridComplexity[0] = (totalRows / rows0);
  operatorComplexity[0] = (totalNnz / nnz0);
}
