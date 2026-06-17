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
extern "C" __global__ void navatala_dataframe_str_title_case_char(const signed char* chars, const int* offsets, const unsigned int* prevIsSpace, const unsigned int* charCount, signed char* outputChars) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = charCount[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    signed char c = chars[gid];
    int cInt = ((int)(c));
    unsigned int prevSpace = prevIsSpace[gid];
    bool afterSpace = (prevSpace == 1u);
    bool isLower = ((cInt >= 97) && (cInt <= 122));
    bool shouldCapitalize = (afterSpace && isLower);
    int upperInt = ((shouldCapitalize) ? ((cInt - 32)) : (cInt));
    signed char outChar = ((signed char)(upperInt));
    outputChars[gid] = outChar;
  }
}
