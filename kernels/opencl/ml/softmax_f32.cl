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

__kernel void navatala_ml_softmax_f32(__global const float* _input, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float gsMax = as_float(0xff7fc99eu);
  for (int itA = 0; itA < (int)(numIters); ++itA) {
    uint idxA = (lid + (((uint)(itA)) * (uint)(256u)));
    if (idxA < countVal) {
      float xA = _input[idxA];
      gsMax = (((gsMax > xA)) ? (gsMax) : (xA));
    }
  }
  sdata[lid] = gsMax;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint maxStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = maxStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = maxStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    maxStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float maxVal = sdata[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float gsSum = as_float(0x00000000u);
  for (int itB = 0; itB < (int)(numIters); ++itB) {
    uint idxB = (lid + (((uint)(itB)) * (uint)(256u)));
    if (idxB < countVal) {
      float eB = exp((_input[idxB] - maxVal));
      gsSum = (gsSum + eB);
    }
  }
  sdata[lid] = gsSum;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint sumStride = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = sumStride;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = sumStride;
    uint nextStride = (strideToHalve >> (uint)(1u));
    sumStride = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float sumVal = sdata[0];
  for (int itC = 0; itC < (int)(numIters); ++itC) {
    uint idxC = (lid + (((uint)(itC)) * (uint)(256u)));
    if (idxC < countVal) {
      float eC = exp((_input[idxC] - maxVal));
      _output[idxC] = (eC / sumVal);
    }
  }
}
