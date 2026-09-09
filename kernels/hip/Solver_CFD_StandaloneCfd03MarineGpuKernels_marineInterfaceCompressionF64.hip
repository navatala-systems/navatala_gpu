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

#include <hip/hip_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64(const double* volumeFlux, const double* faceArea, const double* maximumIncidentFaceSpeed, const double* interfaceNormalDotArea, const double* alphaFace, const double* compressionDatum, const unsigned int* nFaces, double* compressionSpeedOut, double* compressionFluxOut, double* alphaCompressionFluxOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int faceCount = nFaces[0];
  if (gid < faceCount) {
    double coefficient = compressionDatum[0];
    double alpha = alphaFace[gid];
    double compressionSpeed = ((((coefficient * maximumIncidentFaceSpeed[gid]) < ((coefficient * ((((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid]))) ? ((coefficient * maximumIncidentFaceSpeed[gid])) : (((coefficient * ((((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((volumeFlux[gid] < __longlong_as_double(0x0000000000000000ull))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid])));
    double compressionFlux = (compressionSpeed * interfaceNormalDotArea[gid]);
    double alphaCompressionFlux = ((compressionFlux * alpha) * (__longlong_as_double(0x3ff0000000000000ull) - alpha));
    compressionSpeedOut[gid] = compressionSpeed;
    compressionFluxOut[gid] = compressionFlux;
    alphaCompressionFluxOut[gid] = alphaCompressionFlux;
  }
}
