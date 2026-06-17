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
__kernel void navatala_sparse_knn_sparse_top_k_f64(__global const double* distances, __global const uint* nIdxRows, __global const uint* k, __global double* outDistances, __global uint* outIndices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    uint nRows = nIdxRows[0];
    uint kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      outDistances[initIdxU32] = as_double(0x7fee42d130773b76ul);
      outIndices[initIdxU32] = (uint)(4294967295u);
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      uint iU32 = ((uint)(i));
      double dist = distances[iU32];
      uint lastIdx = (kVal - (uint)(1u));
      double kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          double checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          uint newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            uint srcPos = (shiftPos - (uint)(1u));
            double srcDist = outDistances[srcPos];
            uint srcIdx = outIndices[srcPos];
            outDistances[shiftPos] = srcDist;
            outIndices[shiftPos] = srcIdx;
          }
        }
        outDistances[finalInsertPos] = dist;
        outIndices[finalInsertPos] = iU32;
      }
    }
  }
}
