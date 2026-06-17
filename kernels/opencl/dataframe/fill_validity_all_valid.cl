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

__kernel void navatala_dataframe_fill_validity_all_valid(__global const uint* numElements, __global uint* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = numElements[(uint)(0u)];
  uint numWords = ((n / (uint)(32u)) + ((((n % (uint)(32u)) == (uint)(0u))) ? ((uint)(0u)) : ((uint)(1u))));
  bool inBounds = (gid < numWords);
  if (inBounds) {
    bool isLastWord = (gid == (numWords - (uint)(1u)));
    uint remainingBits = (n % (uint)(32u));
    bool hasRemainder = (remainingBits != (uint)(0u));
    uint mask = (((isLastWord && hasRemainder)) ? ((((uint)(1u) << remainingBits) - (uint)(1u))) : ((uint)(4294967295u)));
    _output[gid] = mask;
  }
}
