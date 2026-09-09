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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_barthJespersenCellF64(__global const double* alphaCell, __global const double* alphaMinimum, __global const double* alphaMaximum, __global const ulong* incidentOffsets, __global const double* gradientDotDisplacements, __global const uint* nCells, __global double* psiOut, __global ulong* limitingFaceOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint cellCount = nCells[0];
  if (gid < cellCount) {
    uint nextCell = (gid + (uint)(1u));
    ulong incidentBegin = incidentOffsets[gid];
    ulong incidentEnd = incidentOffsets[nextCell];
    ulong incidentCount = (incidentEnd - incidentBegin);
    double alpha = alphaCell[gid];
    double alphaMin = alphaMinimum[gid];
    double alphaMax = alphaMaximum[gid];
    double psi = as_double(0x3ff0000000000000ul);
    ulong limitingFace = ((ulong)(-1));
    for (int face = 0; face < (int)(incidentCount); ++face) {
      ulong flatIndex = (incidentBegin + ((ulong)(face)));
      double delta = gradientDotDisplacements[flatIndex];
      double rawCandidate = (((as_double(0x0000000000000000ul) < delta)) ? ((((((alphaMax - alpha) / delta) < as_double(0x3ff0000000000000ul))) ? (((alphaMax - alpha) / delta)) : (as_double(0x3ff0000000000000ul)))) : ((((delta < as_double(0x0000000000000000ul))) ? ((((((alphaMin - alpha) / delta) < as_double(0x3ff0000000000000ul))) ? (((alphaMin - alpha) / delta)) : (as_double(0x3ff0000000000000ul)))) : (as_double(0x3ff0000000000000ul)))));
      double lowerClamped = (((rawCandidate < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (rawCandidate));
      double candidate = (((as_double(0x3ff0000000000000ul) < lowerClamped)) ? (as_double(0x3ff0000000000000ul)) : (lowerClamped));
      if (candidate < psi) {
        psi = candidate;
        limitingFace = ((ulong)(face));
      }
    }
    psiOut[gid] = psi;
    limitingFaceOut[gid] = limitingFace;
  }
}
