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

__kernel void navatala_ml_select_tournament_f32(__global const float* fitness, __global const uint* tournamentIndices, __global const uint* nTournaments, __global const uint* tournamentSize, __global uint* selectedIndices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nTournamentsVal = nTournaments[0];
  uint tournamentSizeVal = tournamentSize[0];
  bool inBounds = (gid < nTournamentsVal);
  if (inBounds) {
    uint baseIdx = (gid * tournamentSizeVal);
    uint firstParticipantIdx = tournamentIndices[baseIdx];
    float firstFitness = fitness[firstParticipantIdx];
    uint bestIdx = firstParticipantIdx;
    float bestFit = firstFitness;
    int tournamentSizeI32 = ((int)(tournamentSizeVal));
    for (int k = 0; k < (int)(tournamentSizeI32); ++k) {
      uint kU32 = ((uint)(k));
      bool kGt0 = (kU32 > (uint)(0u));
      if (kGt0) {
        uint participantIdxLoc = (baseIdx + kU32);
        uint participantIdx = tournamentIndices[participantIdxLoc];
        float participantFitness = fitness[participantIdx];
        float currBestFit = bestFit;
        bool isBetter = (participantFitness < currBestFit);
        if (isBetter) {
          bestIdx = participantIdx;
          bestFit = participantFitness;
        }
      }
    }
    uint winner = bestIdx;
    selectedIndices[gid] = winner;
  }
}
