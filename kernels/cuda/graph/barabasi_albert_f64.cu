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
extern "C" __global__ void navatala_graph_barabasi_albert_f64(const unsigned int* cumDegree, const double* random, const double* randomWeights, const unsigned int* newNode, const unsigned int* totalDegree, const unsigned int* m, const unsigned int* numExistingNodes, unsigned int* srcNodes, unsigned int* dstNodes, double* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numM = m[0u];
  bool inBounds = (gid < numM);
  if (inBounds) {
    unsigned int newNodeIdx = newNode[0u];
    unsigned int totalDeg = totalDegree[0u];
    unsigned int numExisting = numExistingNodes[0u];
    double r = random[gid];
    double totalDegF = ((double)(totalDeg));
    double targetDegF = (r * totalDegF);
    unsigned int targetDeg = ((unsigned int)(targetDegF));
    unsigned int loAccum = 0u;
    unsigned int hiAccum = numExisting;
    for (int _ = 0; _ < (int)(32u); ++_) {
      unsigned int lo = loAccum;
      unsigned int hi = hiAccum;
      bool stillSearching = (lo < hi);
      if (stillSearching) {
        unsigned int mid = ((lo + hi) / 2u);
        unsigned int cumVal = cumDegree[mid];
        bool goRight = (cumVal <= targetDeg);
        unsigned int newLo = ((goRight) ? ((mid + 1u)) : (lo));
        unsigned int newHi = ((goRight) ? (hi) : (mid));
        loAccum = newLo;
        hiAccum = newHi;
      }
    }
    unsigned int finalLo = loAccum;
    unsigned int targetNode = (((finalLo > 0u)) ? ((finalLo - 1u)) : (0u));
    double w = randomWeights[gid];
    unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
    srcNodes[slot] = newNodeIdx;
    dstNodes[slot] = targetNode;
    weights[slot] = w;
  }
}
