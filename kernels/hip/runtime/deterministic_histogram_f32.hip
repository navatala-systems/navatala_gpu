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
extern "C" __global__ void navatala_runtime_deterministic_histogram_f32(const float* values, const unsigned int* numBins, const float* minVal, const float* maxVal, const unsigned int* count, unsigned int* histogram) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ unsigned int binIndices[256];
  __shared__ unsigned int localHist[256];
  unsigned int countVal = count[0];
  unsigned int numBinsVal = numBins[0];
  float minValF = minVal[0];
  float maxValF = maxVal[0];
  float range = (maxValF - minValF);
  float numBinsFloat = ((float)(numBinsVal));
  float binWidth = (range / numBinsFloat);
  if ((lid < numBinsVal)) {
    localHist[lid] = 0u;
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = values[gid];
    float normalized = ((val - minValF) / binWidth);
    float binIdxFloat = normalized;
    unsigned int binIdx = ((unsigned int)(binIdxFloat));
    unsigned int maxBinIdx = (numBinsVal - 1u);
    unsigned int clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  __syncthreads();
  if ((lid < numBinsVal)) {
    unsigned int myBin = lid;
    unsigned int histF32RedStride = 128u;
    for (int histF32RedStep = 0; histF32RedStep < (int)(8); ++histF32RedStep) {
      unsigned int histF32Stride = histF32RedStride;
      if ((lid < histF32Stride)) {
        unsigned int partnerIdx = (lid + histF32Stride);
        unsigned int storedBin = binIndices[partnerIdx];
        if ((storedBin == myBin)) {
          unsigned int oldCount = localHist[lid];
          unsigned int newCount = (oldCount + 1u);
          localHist[lid] = newCount;
        }
      }
      unsigned int histF32StrideToHalve = histF32RedStride;
      unsigned int histF32NextStride = (histF32StrideToHalve >> 1u);
      histF32RedStride = histF32NextStride;
      __syncthreads();
    }
  }
  __syncthreads();
  if ((lid == 0u)) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      unsigned int binU32 = ((unsigned int)(bin));
      unsigned int localCount = localHist[binU32];
      unsigned int globalCount = histogram[binU32];
      unsigned int newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}
