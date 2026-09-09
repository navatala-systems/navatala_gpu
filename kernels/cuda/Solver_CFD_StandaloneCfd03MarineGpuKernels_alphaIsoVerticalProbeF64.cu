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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_alphaIsoVerticalProbeF64(const unsigned long long* probeOffsets, const double* verticalCoordinates, const double* alphaValues, const unsigned int* nProbes, double* elevationOut, unsigned long long* lowerPointOut, unsigned int* crossingCountOut, unsigned int* exactPointOut, unsigned int* crossingValidOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int probeCount = nProbes[0];
  if (gid < probeCount) {
    unsigned int nextProbe = (gid + 1u);
    unsigned long long pointBegin = probeOffsets[gid];
    unsigned long long pointEnd = probeOffsets[nextProbe];
    unsigned long long pointCount = (pointEnd - pointBegin);
    unsigned long long segmentCount = (pointCount - ((unsigned long long)(1)));
    unsigned int crossingCount = 0u;
    unsigned long long exactIndex = ((unsigned long long)(-1));
    double elevation = __longlong_as_double(0x0000000000000000ull);
    unsigned long long lowerPoint = ((unsigned long long)(-1));
    unsigned int exactPoint = 0u;
    for (int point = 0; point < (int)(pointCount); ++point) {
      unsigned long long pointLocal = ((unsigned long long)(point));
      unsigned long long pointFlat = (pointBegin + pointLocal);
      double pointAlpha = alphaValues[pointFlat];
      if (pointAlpha == __longlong_as_double(0x3fe0000000000000ull)) {
        unsigned int exactCrossingCount = (crossingCount + 1u);
        crossingCount = exactCrossingCount;
        exactIndex = pointLocal;
      }
    }
    for (int segment = 0; segment < (int)(segmentCount); ++segment) {
      unsigned long long segmentLocal = ((unsigned long long)(segment));
      unsigned long long segmentFlat = (pointBegin + segmentLocal);
      unsigned long long nextSegmentFlat = (segmentFlat + ((unsigned long long)(1)));
      double coordinateA = verticalCoordinates[segmentFlat];
      double coordinateB = verticalCoordinates[nextSegmentFlat];
      double a = (alphaValues[segmentFlat] - __longlong_as_double(0x3fe0000000000000ull));
      double b = (alphaValues[nextSegmentFlat] - __longlong_as_double(0x3fe0000000000000ull));
      if (a < __longlong_as_double(0x0000000000000000ull)) {
        if (__longlong_as_double(0x0000000000000000ull) < b) {
          unsigned int strictCrossingCount = (crossingCount + 1u);
          crossingCount = strictCrossingCount;
          lowerPoint = segmentLocal;
          if (strictCrossingCount == 1u) {
            elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
          }
        }
      } else {
        if (__longlong_as_double(0x0000000000000000ull) < a) {
          if (b < __longlong_as_double(0x0000000000000000ull)) {
            unsigned int strictCrossingCount = (crossingCount + 1u);
            crossingCount = strictCrossingCount;
            lowerPoint = segmentLocal;
            if (strictCrossingCount == 1u) {
              elevation = (coordinateA + (((coordinateB - coordinateA) * (-a)) / (b - a)));
            }
          }
        }
      }
    }
    unsigned int crossingValid = (((crossingCount == 1u)) ? (1u) : (0u));
    if (crossingCount == 1u) {
      if (exactIndex == ((unsigned long long)(-1))) {
      } else {
        unsigned long long exactFlat = (pointBegin + exactIndex);
        exactPoint = 1u;
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
