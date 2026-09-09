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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPressureFaceTractionF64(const double* fluidAreaVectorX, const double* fluidAreaVectorY, const double* fluidAreaVectorZ, const double* faceCentroidX, const double* faceCentroidY, const double* faceCentroidZ, const double* ownerCentroidX, const double* ownerCentroidY, const double* ownerCentroidZ, const double* pressureGradientOwnerX, const double* pressureGradientOwnerY, const double* pressureGradientOwnerZ, const double* pRghOwner, const double* densityOwner, const double* hullPressureDatum, const unsigned int* nFaces, double* areaOut, double* pRghFaceOut, double* absolutePressureOut, double* relativePressureOut, double* forceXOut, double* forceYOut, double* forceZOut, double* momentXOut, double* momentYOut, double* momentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double inverseArea = (__longlong_as_double(0x3ff0000000000000ull) / area);
    double normalX = (areaX * inverseArea);
    double normalY = (areaY * inverseArea);
    double normalZ = (areaZ * inverseArea);
    double normalGradient = (((pressureGradientOwnerX[gid] * normalX) + (pressureGradientOwnerY[gid] * normalY)) + (pressureGradientOwnerZ[gid] * normalZ));
    double tangentGradientX = (pressureGradientOwnerX[gid] - (normalX * normalGradient));
    double tangentGradientY = (pressureGradientOwnerY[gid] - (normalY * normalGradient));
    double tangentGradientZ = (pressureGradientOwnerZ[gid] - (normalZ * normalGradient));
    double ownerToFaceX = (faceCentroidX[gid] - ownerCentroidX[gid]);
    double ownerToFaceY = (faceCentroidY[gid] - ownerCentroidY[gid]);
    double ownerToFaceZ = (faceCentroidZ[gid] - ownerCentroidZ[gid]);
    double pRghFace = (pRghOwner[gid] + (((tangentGradientX * ownerToFaceX) + (tangentGradientY * ownerToFaceY)) + (tangentGradientZ * ownerToFaceZ)));
    double referenceToFaceX = (faceCentroidX[gid] - hullPressureDatum[3]);
    double referenceToFaceY = (faceCentroidY[gid] - hullPressureDatum[4]);
    double referenceToFaceZ = (faceCentroidZ[gid] - hullPressureDatum[5]);
    double head = (densityOwner[gid] * (((hullPressureDatum[0] * referenceToFaceX) + (hullPressureDatum[1] * referenceToFaceY)) + (hullPressureDatum[2] * referenceToFaceZ)));
    double absolutePressure = (pRghFace + head);
    double relativePressure = ((pRghFace - hullPressureDatum[9]) + head);
    double forceX = (areaX * relativePressure);
    double forceY = (areaY * relativePressure);
    double forceZ = (areaZ * relativePressure);
    double momentArmX = (faceCentroidX[gid] - hullPressureDatum[6]);
    double momentArmY = (faceCentroidY[gid] - hullPressureDatum[7]);
    double momentArmZ = (faceCentroidZ[gid] - hullPressureDatum[8]);
    double momentX = ((momentArmY * forceZ) - (momentArmZ * forceY));
    double momentY = ((momentArmZ * forceX) - (momentArmX * forceZ));
    double momentZ = ((momentArmX * forceY) - (momentArmY * forceX));
    areaOut[gid] = area;
    pRghFaceOut[gid] = pRghFace;
    absolutePressureOut[gid] = absolutePressure;
    relativePressureOut[gid] = relativePressure;
    forceXOut[gid] = forceX;
    forceYOut[gid] = forceY;
    forceZOut[gid] = forceZ;
    momentXOut[gid] = momentX;
    momentYOut[gid] = momentY;
    momentZOut[gid] = momentZ;
  }
}
