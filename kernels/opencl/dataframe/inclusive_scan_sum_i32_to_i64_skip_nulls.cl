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

__kernel void navatala_dataframe_inclusive_scan_sum_i32_to_i64_skip_nulls(__global const int* _input, __global const uint* inputValid, __global const uint* count, __global long* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint n = count[(uint)(0u)];
  __local long scanData[256];
  __local uint scanValid[256];
  bool inBounds = (gid < n);
  uint wordIdx = (gid / (uint)(32u));
  uint bitIdx = (gid % (uint)(32u));
  uint validWord = ((inBounds) ? (inputValid[wordIdx]) : ((uint)(0u)));
  uint validBit = ((validWord >> bitIdx) & (uint)(1u));
  bool isValid = (validBit == (uint)(1u));
  long inputVal = (((inBounds && isValid)) ? (((long)(_input[gid]))) : (0));
  scanData[lid] = inputVal;
  scanValid[lid] = (((inBounds && isValid)) ? ((uint)(1u)) : ((uint)(0u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor1 = (lid >= (uint)(1u));
  if (hasNeighbor1) {
    uint neighborIdx1 = (lid - (uint)(1u));
    long neighborVal1 = scanData[neighborIdx1];
    long myVal1 = scanData[lid];
    long newVal1 = (myVal1 + neighborVal1);
    scanData[lid] = newVal1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor2 = (lid >= (uint)(2u));
  if (hasNeighbor2) {
    uint neighborIdx2 = (lid - (uint)(2u));
    long neighborVal2 = scanData[neighborIdx2];
    long myVal2 = scanData[lid];
    long newVal2 = (myVal2 + neighborVal2);
    scanData[lid] = newVal2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor4 = (lid >= (uint)(4u));
  if (hasNeighbor4) {
    uint neighborIdx4 = (lid - (uint)(4u));
    long neighborVal4 = scanData[neighborIdx4];
    long myVal4 = scanData[lid];
    long newVal4 = (myVal4 + neighborVal4);
    scanData[lid] = newVal4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor8 = (lid >= (uint)(8u));
  if (hasNeighbor8) {
    uint neighborIdx8 = (lid - (uint)(8u));
    long neighborVal8 = scanData[neighborIdx8];
    long myVal8 = scanData[lid];
    long newVal8 = (myVal8 + neighborVal8);
    scanData[lid] = newVal8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor16 = (lid >= (uint)(16u));
  if (hasNeighbor16) {
    uint neighborIdx16 = (lid - (uint)(16u));
    long neighborVal16 = scanData[neighborIdx16];
    long myVal16 = scanData[lid];
    long newVal16 = (myVal16 + neighborVal16);
    scanData[lid] = newVal16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor32 = (lid >= (uint)(32u));
  if (hasNeighbor32) {
    uint neighborIdx32 = (lid - (uint)(32u));
    long neighborVal32 = scanData[neighborIdx32];
    long myVal32 = scanData[lid];
    long newVal32 = (myVal32 + neighborVal32);
    scanData[lid] = newVal32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor64 = (lid >= (uint)(64u));
  if (hasNeighbor64) {
    uint neighborIdx64 = (lid - (uint)(64u));
    long neighborVal64 = scanData[neighborIdx64];
    long myVal64 = scanData[lid];
    long newVal64 = (myVal64 + neighborVal64);
    scanData[lid] = newVal64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor128 = (lid >= (uint)(128u));
  if (hasNeighbor128) {
    uint neighborIdx128 = (lid - (uint)(128u));
    long neighborVal128 = scanData[neighborIdx128];
    long myVal128 = scanData[lid];
    long newVal128 = (myVal128 + neighborVal128);
    scanData[lid] = newVal128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (inBounds) {
    long scanResult = scanData[lid];
    uint rowValid = scanValid[lid];
    _output[gid] = scanResult;
    uint outBit = (rowValid << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
