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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64(__global const ulong* probeOffsets, __global const double* verticalCoordinates, __global const double* alphaValues, __global const uint* nProbes, __global double* elevationOut, __global ulong* lowerPointOut, __global uint* crossingCountOut, __global uint* exactPointOut, __global uint* crossingValidOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint probeCount = nProbes[0];
  if (gid < probeCount) {
    uint nextProbe = (gid + (uint)(1u));
    ulong pointBegin = probeOffsets[gid];
    ulong pointEnd = probeOffsets[nextProbe];
    ulong pointCount = (pointEnd - pointBegin);
    ulong segmentCount = (pointCount - ((ulong)(1)));
    uint crossingCount = (uint)(0u);
    ulong exactIndex = ((ulong)(-1));
    double elevation = as_double(0x0000000000000000ul);
    ulong lowerPoint = ((ulong)(-1));
    uint exactPoint = (uint)(0u);
    for (int point = 0; point < (int)(pointCount); ++point) {
      ulong pointLocal = ((ulong)(point));
      ulong pointFlat = (pointBegin + pointLocal);
      double pointAlpha = alphaValues[pointFlat];
      if (pointAlpha == as_double(0x3fe0000000000000ul)) {
        uint exactCrossingCount = (crossingCount + (uint)(1u));
        crossingCount = exactCrossingCount;
        exactIndex = pointLocal;
      }
    }
    for (int segment = 0; segment < (int)(segmentCount); ++segment) {
      ulong segmentLocal = ((ulong)(segment));
      ulong segmentFlat = (pointBegin + segmentLocal);
      ulong nextSegmentFlat = (segmentFlat + ((ulong)(1)));
      double coordinateA = verticalCoordinates[segmentFlat];
      double coordinateB = verticalCoordinates[nextSegmentFlat];
      double a = (alphaValues[segmentFlat] - as_double(0x3fe0000000000000ul));
      double b = (alphaValues[nextSegmentFlat] - as_double(0x3fe0000000000000ul));
      if (a < as_double(0x0000000000000000ul)) {
        if (as_double(0x0000000000000000ul) < b) {
          uint strictCrossingCount = (crossingCount + (uint)(1u));
          crossingCount = strictCrossingCount;
          lowerPoint = segmentLocal;
          if (strictCrossingCount == (uint)(1u)) {
            elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
          }
        }
      } else {
        if (as_double(0x0000000000000000ul) < a) {
          if (b < as_double(0x0000000000000000ul)) {
            uint strictCrossingCount = (crossingCount + (uint)(1u));
            crossingCount = strictCrossingCount;
            lowerPoint = segmentLocal;
            if (strictCrossingCount == (uint)(1u)) {
              elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
            }
          }
        }
      }
    }
    uint crossingValid = (((crossingCount == (uint)(1u))) ? ((uint)(1u)) : ((uint)(0u)));
    if (crossingCount == (uint)(1u)) {
      if (exactIndex == ((ulong)(-1))) {
      } else {
        ulong exactFlat = (pointBegin + exactIndex);
        exactPoint = (uint)(1u);
        lowerPoint = exactIndex;
        elevation = verticalCoordinates[exactFlat];
      }
    }
    elevationOut[gid] = elevation;
    lowerPointOut[gid] = lowerPoint;
    crossingCountOut[gid] = crossingCount;
    exactPointOut[gid] = exactPoint;
    crossingValidOut[gid] = crossingValid;
  }
}
