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
__kernel void navatala_runtime_deterministic_histogram_f64(__global const double* values, __global const uint* numBins, __global const double* minVal, __global const double* maxVal, __global const uint* count, __global uint* histogram) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local uint binIndices[256];
  __local uint localHist[256];
  uint countVal = count[0];
  uint numBinsVal = numBins[0];
  double minValF = minVal[0];
  double maxValF = maxVal[0];
  double range = (maxValF - minValF);
  double numBinsFloat = ((double)(numBinsVal));
  double binWidth = (range / numBinsFloat);
  if (lid < numBinsVal) {
    localHist[lid] = (uint)(0u);
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double val = values[gid];
    double normalized = ((val - minValF) / binWidth);
    double binIdxFloat = normalized;
    uint binIdx = ((uint)(binIdxFloat));
    uint maxBinIdx = (numBinsVal - (uint)(1u));
    uint clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < numBinsVal) {
    uint myBin = lid;
    uint histF64RedStride = (uint)(128u);
    for (int histF64RedStep = 0; histF64RedStep < (int)(8); ++histF64RedStep) {
      uint histF64Stride = histF64RedStride;
      if (lid < histF64Stride) {
        uint partnerIdx = (lid + histF64Stride);
        uint storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          uint oldCount = localHist[lid];
          uint newCount = (oldCount + (uint)(1u));
          localHist[lid] = newCount;
        }
      }
      uint histF64StrideToHalve = histF64RedStride;
      uint histF64NextStride = (histF64StrideToHalve >> (uint)(1u));
      histF64RedStride = histF64NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == (uint)(0u)) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      uint binU32 = ((uint)(bin));
      uint localCount = localHist[binU32];
      uint globalCount = histogram[binU32];
      uint newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}
