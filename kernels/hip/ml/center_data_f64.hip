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
extern "C" __global__ void navatala_ml_center_data_f64(const double* _input, const unsigned int* nSamples, const unsigned int* nFeatures, const double* means, double* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int colIdx = (gid % d);
    double x = _input[gid];
    double colMean = means[colIdx];
    double centered = (x - colMean);
    _output[gid] = centered;
  }
}
