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
__kernel void navatala_dataframe_compute_bin_indices_f64(__global const double* _input, __global const uint* count, __global const uint* numBins, __global const double* minVal, __global const double* maxVal, __global uint* binIndices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint nBins = numBins[(uint)(0u)];
  double vMin = minVal[(uint)(0u)];
  double vMax = maxVal[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    double val = _input[gid];
    double range = (vMax - vMin);
    double nBinsF = ((double)(nBins));
    double normalized = ((val - vMin) / range);
    double binIdxF = (normalized * nBinsF);
    int binIdxRaw = ((int)(floor(binIdxF)));
    int nBinsMinusOne = (((int)(nBins)) - 1);
    int binIdxClamped = (((binIdxRaw < 0)) ? (0) : ((((binIdxRaw > nBinsMinusOne)) ? (nBinsMinusOne) : (binIdxRaw))));
    uint binIdx = ((uint)(binIdxClamped));
    binIndices[gid] = binIdx;
  }
}
