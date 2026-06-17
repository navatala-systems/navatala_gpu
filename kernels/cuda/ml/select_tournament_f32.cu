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
extern "C" __global__ void navatala_ml_select_tournament_f32(const float* fitness, const unsigned int* tournamentIndices, const unsigned int* nTournaments, const unsigned int* tournamentSize, unsigned int* selectedIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nTournamentsVal = nTournaments[0];
  unsigned int tournamentSizeVal = tournamentSize[0];
  bool inBounds = (gid < nTournamentsVal);
  if (inBounds) {
    unsigned int baseIdx = (gid * tournamentSizeVal);
    unsigned int firstParticipantIdx = tournamentIndices[baseIdx];
    float firstFitness = fitness[firstParticipantIdx];
    unsigned int bestIdx = firstParticipantIdx;
    float bestFit = firstFitness;
    int tournamentSizeI32 = ((int)(tournamentSizeVal));
    for (int k = 0; k < (int)(tournamentSizeI32); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      bool kGt0 = (kU32 > 0u);
      if (kGt0) {
        unsigned int participantIdxLoc = (baseIdx + kU32);
        unsigned int participantIdx = tournamentIndices[participantIdxLoc];
        float participantFitness = fitness[participantIdx];
        float currBestFit = bestFit;
        bool isBetter = (participantFitness < currBestFit);
        if (isBetter) {
          bestIdx = participantIdx;
          bestFit = participantFitness;
        }
      }
    }
    unsigned int winner = bestIdx;
    selectedIndices[gid] = winner;
  }
}
