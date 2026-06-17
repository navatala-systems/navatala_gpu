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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_compute_bin_indices_f64(const double* _input, const unsigned int* count, const unsigned int* numBins, const double* minVal, const double* maxVal, unsigned int* binIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int nBins = numBins[0u];
  double vMin = minVal[0u];
  double vMax = maxVal[0u];
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
    unsigned int binIdx = ((unsigned int)(binIdxClamped));
    binIndices[gid] = binIdx;
  }
}
