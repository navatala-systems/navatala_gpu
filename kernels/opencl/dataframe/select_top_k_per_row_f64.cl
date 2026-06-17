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
__kernel void navatala_dataframe_select_top_k_per_row_f64(__global const double* distances, __global const uint* q, __global const uint* n, __global const uint* k, __global double* outDistances, __global uint* outIndices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint nVal = n[0];
  uint kVal = k[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < qVal);
  if (inBounds) {
    uint inBase = (rowIdx * nVal);
    uint outBase = (rowIdx * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outDistIdx = (outBase + initIdxU32);
      outDistances[outDistIdx] = as_double(0x7fee42d130773b76ul);
      outIndices[outDistIdx] = (uint)(4294967295u);
    }
    for (int col = 0; col < (int)(nVal); ++col) {
      uint colU32 = ((uint)(col));
      uint inIdx = (inBase + colU32);
      double val = distances[inIdx];
      uint lastIdx = (kVal - (uint)(1u));
      uint lastOutIdx = (outBase + lastIdx);
      double kthDist = outDistances[lastOutIdx];
      bool shouldInsert = (val < kthDist);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          uint checkOutIdx = (outBase + checkPos);
          double checkDist = outDistances[checkOutIdx];
          bool isSmaller = (val < checkDist);
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
            uint srcOutIdx = (outBase + srcPos);
            uint dstOutIdx = (outBase + shiftPos);
            double srcDist = outDistances[srcOutIdx];
            uint srcIdx = outIndices[srcOutIdx];
            outDistances[dstOutIdx] = srcDist;
            outIndices[dstOutIdx] = srcIdx;
          }
        }
        uint insertOutIdx = (outBase + finalInsertPos);
        outDistances[insertOutIdx] = val;
        outIndices[insertOutIdx] = colU32;
      }
    }
  }
}
