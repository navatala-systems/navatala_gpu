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
__kernel void navatala_dataframe_tdigest_merge_f64(__global const double* meansA, __global const double* weightsA, __global const uint* countA, __global const double* totalWeightA, __global const double* minA, __global const double* maxA, __global const double* meansB, __global const double* weightsB, __global const uint* countB, __global const double* totalWeightB, __global const double* minB, __global const double* maxB, __global const double* compressionIn, __global double* meansOut, __global double* weightsOut, __global uint* countOut, __global double* totalWeightOut, __global double* minOut, __global double* maxOut, __global double* compressionOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    uint cA = countA[0];
    uint cB = countB[0];
    double twA = totalWeightA[0];
    double twB = totalWeightB[0];
    double minValA = minA[0];
    double maxValA = maxA[0];
    double minValB = minB[0];
    double maxValB = maxB[0];
    double delta = compressionIn[0];
    double newTw = (twA + twB);
    double newMin = (((minValA < minValB)) ? (minValA) : (minValB));
    double newMax = (((maxValA > maxValB)) ? (maxValA) : (maxValB));
    uint newCount = (cA + cB);
    countOut[0] = newCount;
    totalWeightOut[0] = newTw;
    minOut[0] = newMin;
    maxOut[0] = newMax;
    compressionOut[0] = delta;
  }
}
