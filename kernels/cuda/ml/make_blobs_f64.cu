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
extern "C" __global__ void navatala_ml_make_blobs_f64(const double* centers, const unsigned int* labels, const double* randomNormal, const double* stddev, const unsigned int* n, const unsigned int* d, double* data) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numD = d[0];
  unsigned int totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    double sigma = stddev[0];
    unsigned int sampleIdx = (gid / numD);
    unsigned int dimIdx = (gid % numD);
    unsigned int label = labels[sampleIdx];
    unsigned int centerIdx = ((label * numD) + dimIdx);
    double centerVal = centers[centerIdx];
    double randVal = randomNormal[gid];
    double noise = (sigma * randVal);
    double result = (centerVal + noise);
    data[gid] = result;
  }
}
