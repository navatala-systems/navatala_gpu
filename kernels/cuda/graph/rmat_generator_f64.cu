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
extern "C" __global__ void navatala_graph_rmat_generator_f64(const double* random, const double* randomWeights, const double* a, const double* b, const double* c, const double* d, const unsigned int* n, const unsigned int* scale, const unsigned int* numEdges, unsigned int* srcNodes, unsigned int* dstNodes, double* weights, unsigned int* edgeCount) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numE = numEdges[0u];
  bool inBounds = (gid < numE);
  if (inBounds) {
    double probA = a[0u];
    double probB = b[0u];
    double probC = c[0u];
    unsigned int scaleVal = scale[0u];
    double ab = (probA + probB);
    double abc = (ab + probC);
    unsigned int srcAccum = 0u;
    unsigned int dstAccum = 0u;
    for (int level = 0; level < (int)(scaleVal); ++level) {
      unsigned int levelU32 = ((unsigned int)(level));
      unsigned int randIdx = ((gid * scaleVal) + levelU32);
      double r = random[randIdx];
      unsigned int bitPos = ((scaleVal - levelU32) - 1u);
      unsigned int bitMask = (1u << bitPos);
      bool inQ01 = (r >= probA);
      bool inQ10 = (r >= ab);
      bool inQ11 = (r >= abc);
      unsigned int srcBit = ((inQ10) ? (bitMask) : (0u));
      unsigned int currentSrc = srcAccum;
      unsigned int newSrc = (currentSrc | srcBit);
      srcAccum = newSrc;
      bool dstBitCond = ((inQ01 && (!inQ10)) || inQ11);
      unsigned int dstBit = ((dstBitCond) ? (bitMask) : (0u));
      unsigned int currentDst = dstAccum;
      unsigned int newDst = (currentDst | dstBit);
      dstAccum = newDst;
    }
    unsigned int src = srcAccum;
    unsigned int dst = dstAccum;
    double w = randomWeights[gid];
    unsigned int slot = atomicAdd(&(edgeCount[0u]), 1u);
    srcNodes[slot] = src;
    dstNodes[slot] = dst;
    weights[slot] = w;
  }
}
