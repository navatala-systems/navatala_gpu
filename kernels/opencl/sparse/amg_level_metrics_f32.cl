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

__kernel void navatala_sparse_amg_level_metrics_f32(__global const uint* levelNRows, __global const uint* levelNnz, __global const uint* nLevels, __global float* gridComplexity, __global float* operatorComplexity) {
  int gid0 = (int)get_global_id(0);
  int nL = ((int)(nLevels[0]));
  float rows0 = ((float)(levelNRows[0]));
  float nnz0 = ((float)(levelNnz[0]));
  float totalRows = as_float(0x00000000u);
  float totalNnz = as_float(0x00000000u);
  for (int i = 0; i < (int)(nL); ++i) {
    float ri = ((float)(levelNRows[i]));
    float ni = ((float)(levelNnz[i]));
    totalRows = (totalRows + ri);
    totalNnz = (totalNnz + ni);
  }
  gridComplexity[0] = (totalRows / rows0);
  operatorComplexity[0] = (totalNnz / nnz0);
}
