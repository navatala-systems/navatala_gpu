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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineInterfaceCompressionF64(__global const double* volumeFlux, __global const double* faceArea, __global const double* maximumIncidentFaceSpeed, __global const double* interfaceNormalDotArea, __global const double* alphaFace, __global const double* compressionDatum, __global const uint* nFaces, __global double* compressionSpeedOut, __global double* compressionFluxOut, __global double* alphaCompressionFluxOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint faceCount = nFaces[0];
  if (gid < faceCount) {
    double coefficient = compressionDatum[0];
    double alpha = alphaFace[gid];
    double compressionSpeed = ((((coefficient * maximumIncidentFaceSpeed[gid]) < ((coefficient * ((((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid]))) ? ((coefficient * maximumIncidentFaceSpeed[gid])) : (((coefficient * ((((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid])) == as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((((volumeFlux[gid] < as_double(0x0000000000000000ul))) ? ((-volumeFlux[gid])) : (volumeFlux[gid]))))) / faceArea[gid])));
    double compressionFlux = (compressionSpeed * interfaceNormalDotArea[gid]);
    double alphaCompressionFlux = ((compressionFlux * alpha) * (as_double(0x3ff0000000000000ul) - alpha));
    compressionSpeedOut[gid] = compressionSpeed;
    compressionFluxOut[gid] = compressionFlux;
    alphaCompressionFluxOut[gid] = alphaCompressionFlux;
  }
}
