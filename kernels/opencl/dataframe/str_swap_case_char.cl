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

__kernel void navatala_dataframe_str_swap_case_char(__global const char* chars, __global const uint* charCount, __global char* outputChars) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = charCount[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    char c = chars[gid];
    int cInt = ((int)(c));
    bool isUpper = ((cInt >= 65) && (cInt <= 90));
    bool isLower = ((cInt >= 97) && (cInt <= 122));
    int toLower = (cInt + 32);
    int toUpper = (cInt - 32);
    int swappedInt = ((isUpper) ? (toLower) : (((isLower) ? (toUpper) : (cInt))));
    char outChar = ((char)(swappedInt));
    outputChars[gid] = outChar;
  }
}
