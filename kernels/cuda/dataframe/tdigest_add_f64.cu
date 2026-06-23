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
extern "C" __global__ void navatala_dataframe_tdigest_add_f64(const double* value, const double* weight, double* centroidMeans, double* centroidWeights, unsigned int* centroidCount, double* totalWeight, double* minVal, double* maxVal) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    double x = value[0];
    double w = weight[0];
    unsigned int count = centroidCount[0];
    double tw = totalWeight[0];
    double minV = minVal[0];
    double maxV = maxVal[0];
    double newMin = (((x < minV)) ? (x) : (minV));
    double newMax = (((x > maxV)) ? (x) : (maxV));
    double newTw = (tw + w);
    unsigned int newCount = (count + 1u);
    unsigned int idx = count;
    centroidMeans[((int)(idx))] = x;
    centroidWeights[((int)(idx))] = w;
    centroidCount[0] = newCount;
    totalWeight[0] = newTw;
    minVal[0] = newMin;
    maxVal[0] = newMax;
  }
}
