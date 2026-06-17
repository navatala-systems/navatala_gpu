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
__kernel void navatala_dataframe_inclusive_scan_sum_f64(__global const double* _input, __global const uint* inputValid, __global const uint* count, __global double* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint n = count[(uint)(0u)];
  __local double scanData[256];
  __local uint scanValid[256];
  bool inBounds = (gid < n);
  uint wordIdx = (gid / (uint)(32u));
  uint bitIdx = (gid % (uint)(32u));
  uint validWord = ((inBounds) ? (inputValid[wordIdx]) : ((uint)(0u)));
  uint validBit = ((validWord >> bitIdx) & (uint)(1u));
  bool isValid = (validBit == (uint)(1u));
  double inputVal = (((inBounds && isValid)) ? (_input[gid]) : (as_double(0x0000000000000000ul)));
  scanData[lid] = inputVal;
  scanValid[lid] = (((inBounds && isValid)) ? ((uint)(1u)) : ((uint)(0u)));
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor1 = (lid >= (uint)(1u));
  if (hasNeighbor1) {
    uint neighborIdx1 = (lid - (uint)(1u));
    double neighborVal1 = scanData[neighborIdx1];
    uint neighborValid1 = scanValid[neighborIdx1];
    double myVal1 = scanData[lid];
    uint myValid1 = scanValid[lid];
    double newVal1 = (myVal1 + neighborVal1);
    uint newValid1 = (myValid1 & neighborValid1);
    scanData[lid] = newVal1;
    scanValid[lid] = newValid1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor2 = (lid >= (uint)(2u));
  if (hasNeighbor2) {
    uint neighborIdx2 = (lid - (uint)(2u));
    double neighborVal2 = scanData[neighborIdx2];
    uint neighborValid2 = scanValid[neighborIdx2];
    double myVal2 = scanData[lid];
    uint myValid2 = scanValid[lid];
    double newVal2 = (myVal2 + neighborVal2);
    uint newValid2 = (myValid2 & neighborValid2);
    scanData[lid] = newVal2;
    scanValid[lid] = newValid2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor4 = (lid >= (uint)(4u));
  if (hasNeighbor4) {
    uint neighborIdx4 = (lid - (uint)(4u));
    double neighborVal4 = scanData[neighborIdx4];
    uint neighborValid4 = scanValid[neighborIdx4];
    double myVal4 = scanData[lid];
    uint myValid4 = scanValid[lid];
    double newVal4 = (myVal4 + neighborVal4);
    uint newValid4 = (myValid4 & neighborValid4);
    scanData[lid] = newVal4;
    scanValid[lid] = newValid4;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor8 = (lid >= (uint)(8u));
  if (hasNeighbor8) {
    uint neighborIdx8 = (lid - (uint)(8u));
    double neighborVal8 = scanData[neighborIdx8];
    uint neighborValid8 = scanValid[neighborIdx8];
    double myVal8 = scanData[lid];
    uint myValid8 = scanValid[lid];
    double newVal8 = (myVal8 + neighborVal8);
    uint newValid8 = (myValid8 & neighborValid8);
    scanData[lid] = newVal8;
    scanValid[lid] = newValid8;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor16 = (lid >= (uint)(16u));
  if (hasNeighbor16) {
    uint neighborIdx16 = (lid - (uint)(16u));
    double neighborVal16 = scanData[neighborIdx16];
    uint neighborValid16 = scanValid[neighborIdx16];
    double myVal16 = scanData[lid];
    uint myValid16 = scanValid[lid];
    double newVal16 = (myVal16 + neighborVal16);
    uint newValid16 = (myValid16 & neighborValid16);
    scanData[lid] = newVal16;
    scanValid[lid] = newValid16;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor32 = (lid >= (uint)(32u));
  if (hasNeighbor32) {
    uint neighborIdx32 = (lid - (uint)(32u));
    double neighborVal32 = scanData[neighborIdx32];
    uint neighborValid32 = scanValid[neighborIdx32];
    double myVal32 = scanData[lid];
    uint myValid32 = scanValid[lid];
    double newVal32 = (myVal32 + neighborVal32);
    uint newValid32 = (myValid32 & neighborValid32);
    scanData[lid] = newVal32;
    scanValid[lid] = newValid32;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor64 = (lid >= (uint)(64u));
  if (hasNeighbor64) {
    uint neighborIdx64 = (lid - (uint)(64u));
    double neighborVal64 = scanData[neighborIdx64];
    uint neighborValid64 = scanValid[neighborIdx64];
    double myVal64 = scanData[lid];
    uint myValid64 = scanValid[lid];
    double newVal64 = (myVal64 + neighborVal64);
    uint newValid64 = (myValid64 & neighborValid64);
    scanData[lid] = newVal64;
    scanValid[lid] = newValid64;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
  bool hasNeighbor128 = (lid >= (uint)(128u));
  if (hasNeighbor128) {
    uint neighborIdx128 = (lid - (uint)(128u));
    double neighborVal128 = scanData[neighborIdx128];
    uint neighborValid128 = scanValid[neighborIdx128];
    double myVal128 = scanData[lid];
    uint myValid128 = scanValid[lid];
    double newVal128 = (myVal128 + neighborVal128);
    uint newValid128 = (myValid128 & neighborValid128);
    scanData[lid] = newVal128;
    scanValid[lid] = newValid128;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (inBounds) {
    double scanResult = scanData[lid];
    uint scanValidResult = scanValid[lid];
    _output[gid] = scanResult;
    uint outBit = (scanValidResult << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
