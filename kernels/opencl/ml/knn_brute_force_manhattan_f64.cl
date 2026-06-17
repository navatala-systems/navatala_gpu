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
__kernel void navatala_ml_knn_brute_force_manhattan_f64(__global const double* queries, __global const double* data, __global const uint* q, __global const uint* n, __global const uint* d, __global const uint* k, __global uint* indices, __global double* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_double(0x7fee42d130773b76ul);
      indices[outIdx] = (uint)(4294967295u);
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      double distSumAccum = as_double(0x0000000000000000ul);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        double qv = queries[qIdx];
        double dv = data[dIdx];
        double diff = (qv - dv);
        double absDiff = fabs(diff);
        double currDistSum = distSumAccum;
        double newDistSum = (currDistSum + absDiff);
        distSumAccum = newDistSum;
      }
      double distSum = distSumAccum;
      uint worstIdx = (outBase + (kVal - (uint)(1u)));
      double worstDist = distances[worstIdx];
      bool shouldInsert = (distSum < worstDist);
      if (shouldInsert) {
        uint initInsertPos = (kVal - (uint)(1u));
        uint insertPosAccum = initInsertPos;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint checkPos = ((kVal - (uint)(1u)) - scanIdxU32);
          uint checkIdx = (outBase + checkPos);
          double checkDist = distances[checkIdx];
          bool shouldShift = (distSum < checkDist);
          uint currInsertPos = insertPosAccum;
          uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint insertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)((kVal - (uint)(1u))); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = ((kVal - (uint)(2u)) - shiftIdxU32);
          bool shouldDoShift = (shiftPos >= insertPos);
          if (shouldDoShift) {
            uint srcIdx = (outBase + shiftPos);
            uint dstIdx = (outBase + (shiftPos + (uint)(1u)));
            double shiftDist = distances[srcIdx];
            uint shiftIndex = indices[srcIdx];
            distances[dstIdx] = shiftDist;
            indices[dstIdx] = shiftIndex;
          }
        }
        uint insertIdx = (outBase + insertPos);
        distances[insertIdx] = distSum;
        indices[insertIdx] = dataIdxU32;
      }
    }
  }
}
