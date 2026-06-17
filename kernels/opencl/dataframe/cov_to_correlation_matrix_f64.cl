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
__kernel void navatala_dataframe_cov_to_correlation_matrix_f64(__global const double* covMatrix, __global const uint* nFeatures, __global double* corMatrix) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d = nFeatures[0];
  uint totalElems = (d * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / d);
    uint j = (gid % d);
    uint diagIdxI = ((i * d) + i);
    uint diagIdxJ = ((j * d) + j);
    double covIJ = covMatrix[gid];
    double covII = covMatrix[diagIdxI];
    double covJJ = covMatrix[diagIdxJ];
    double varProduct = (covII * covJJ);
    double denom = sqrt(varProduct);
    double corVal = (covIJ / denom);
    corMatrix[gid] = corVal;
  }
}
