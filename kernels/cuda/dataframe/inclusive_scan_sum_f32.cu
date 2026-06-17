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
extern "C" __global__ void navatala_dataframe_inclusive_scan_sum_f32(const float* _input, const unsigned int* inputValid, const unsigned int* count, float* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int n = count[0u];
  __shared__ float scanData[256];
  __shared__ unsigned int scanValid[256];
  bool inBounds = (gid < n);
  unsigned int wordIdx = (gid / 32u);
  unsigned int bitIdx = (gid % 32u);
  unsigned int validWord = ((inBounds) ? (inputValid[wordIdx]) : (0u));
  unsigned int validBit = ((validWord >> bitIdx) & 1u);
  bool isValid = (validBit == 1u);
  float inputVal = (((inBounds && isValid)) ? (_input[gid]) : (__uint_as_float(0x00000000u)));
  scanData[lid] = inputVal;
  scanValid[lid] = (((inBounds && isValid)) ? (1u) : (0u));
  __syncthreads();
  bool hasNeighbor1 = (lid >= 1u);
  if (hasNeighbor1) {
    unsigned int neighborIdx1 = (lid - 1u);
    float neighborVal1 = scanData[neighborIdx1];
    unsigned int neighborValid1 = scanValid[neighborIdx1];
    float myVal1 = scanData[lid];
    unsigned int myValid1 = scanValid[lid];
    float newVal1 = (myVal1 + neighborVal1);
    unsigned int newValid1 = (myValid1 & neighborValid1);
    scanData[lid] = newVal1;
    scanValid[lid] = newValid1;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor2 = (lid >= 2u);
  if (hasNeighbor2) {
    unsigned int neighborIdx2 = (lid - 2u);
    float neighborVal2 = scanData[neighborIdx2];
    unsigned int neighborValid2 = scanValid[neighborIdx2];
    float myVal2 = scanData[lid];
    unsigned int myValid2 = scanValid[lid];
    float newVal2 = (myVal2 + neighborVal2);
    unsigned int newValid2 = (myValid2 & neighborValid2);
    scanData[lid] = newVal2;
    scanValid[lid] = newValid2;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor4 = (lid >= 4u);
  if (hasNeighbor4) {
    unsigned int neighborIdx4 = (lid - 4u);
    float neighborVal4 = scanData[neighborIdx4];
    unsigned int neighborValid4 = scanValid[neighborIdx4];
    float myVal4 = scanData[lid];
    unsigned int myValid4 = scanValid[lid];
    float newVal4 = (myVal4 + neighborVal4);
    unsigned int newValid4 = (myValid4 & neighborValid4);
    scanData[lid] = newVal4;
    scanValid[lid] = newValid4;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor8 = (lid >= 8u);
  if (hasNeighbor8) {
    unsigned int neighborIdx8 = (lid - 8u);
    float neighborVal8 = scanData[neighborIdx8];
    unsigned int neighborValid8 = scanValid[neighborIdx8];
    float myVal8 = scanData[lid];
    unsigned int myValid8 = scanValid[lid];
    float newVal8 = (myVal8 + neighborVal8);
    unsigned int newValid8 = (myValid8 & neighborValid8);
    scanData[lid] = newVal8;
    scanValid[lid] = newValid8;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor16 = (lid >= 16u);
  if (hasNeighbor16) {
    unsigned int neighborIdx16 = (lid - 16u);
    float neighborVal16 = scanData[neighborIdx16];
    unsigned int neighborValid16 = scanValid[neighborIdx16];
    float myVal16 = scanData[lid];
    unsigned int myValid16 = scanValid[lid];
    float newVal16 = (myVal16 + neighborVal16);
    unsigned int newValid16 = (myValid16 & neighborValid16);
    scanData[lid] = newVal16;
    scanValid[lid] = newValid16;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor32 = (lid >= 32u);
  if (hasNeighbor32) {
    unsigned int neighborIdx32 = (lid - 32u);
    float neighborVal32 = scanData[neighborIdx32];
    unsigned int neighborValid32 = scanValid[neighborIdx32];
    float myVal32 = scanData[lid];
    unsigned int myValid32 = scanValid[lid];
    float newVal32 = (myVal32 + neighborVal32);
    unsigned int newValid32 = (myValid32 & neighborValid32);
    scanData[lid] = newVal32;
    scanValid[lid] = newValid32;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor64 = (lid >= 64u);
  if (hasNeighbor64) {
    unsigned int neighborIdx64 = (lid - 64u);
    float neighborVal64 = scanData[neighborIdx64];
    unsigned int neighborValid64 = scanValid[neighborIdx64];
    float myVal64 = scanData[lid];
    unsigned int myValid64 = scanValid[lid];
    float newVal64 = (myVal64 + neighborVal64);
    unsigned int newValid64 = (myValid64 & neighborValid64);
    scanData[lid] = newVal64;
    scanValid[lid] = newValid64;
  }
  __syncthreads();
  __syncthreads();
  bool hasNeighbor128 = (lid >= 128u);
  if (hasNeighbor128) {
    unsigned int neighborIdx128 = (lid - 128u);
    float neighborVal128 = scanData[neighborIdx128];
    unsigned int neighborValid128 = scanValid[neighborIdx128];
    float myVal128 = scanData[lid];
    unsigned int myValid128 = scanValid[lid];
    float newVal128 = (myVal128 + neighborVal128);
    unsigned int newValid128 = (myValid128 & neighborValid128);
    scanData[lid] = newVal128;
    scanValid[lid] = newValid128;
  }
  __syncthreads();
  if (inBounds) {
    float scanResult = scanData[lid];
    unsigned int scanValidResult = scanValid[lid];
    _output[gid] = scanResult;
    unsigned int outBit = (scanValidResult << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
