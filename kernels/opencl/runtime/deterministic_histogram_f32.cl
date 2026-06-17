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

__kernel void navatala_runtime_deterministic_histogram_f32(__global const float* values, __global const uint* numBins, __global const float* minVal, __global const float* maxVal, __global const uint* count, __global uint* histogram) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local uint binIndices[256];
  __local uint localHist[256];
  uint countVal = count[0];
  uint numBinsVal = numBins[0];
  float minValF = minVal[0];
  float maxValF = maxVal[0];
  float range = (maxValF - minValF);
  float numBinsFloat = ((float)(numBinsVal));
  float binWidth = (range / numBinsFloat);
  if ((lid < numBinsVal)) {
    localHist[lid] = (uint)(0u);
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = values[gid];
    float normalized = ((val - minValF) / binWidth);
    float binIdxFloat = normalized;
    uint binIdx = ((uint)(binIdxFloat));
    uint maxBinIdx = (numBinsVal - (uint)(1u));
    uint clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid < numBinsVal)) {
    uint myBin = lid;
    uint histF32RedStride = (uint)(128u);
    for (int histF32RedStep = 0; histF32RedStep < (int)(8); ++histF32RedStep) {
      uint histF32Stride = histF32RedStride;
      if ((lid < histF32Stride)) {
        uint partnerIdx = (lid + histF32Stride);
        uint storedBin = binIndices[partnerIdx];
        if ((storedBin == myBin)) {
          uint oldCount = localHist[lid];
          uint newCount = (oldCount + (uint)(1u));
          localHist[lid] = newCount;
        }
      }
      uint histF32StrideToHalve = histF32RedStride;
      uint histF32NextStride = (histF32StrideToHalve >> (uint)(1u));
      histF32RedStride = histF32NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid == (uint)(0u))) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      uint binU32 = ((uint)(bin));
      uint localCount = localHist[binU32];
      uint globalCount = histogram[binU32];
      uint newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}
