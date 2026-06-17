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

__kernel void navatala_dataframe_str_normalize_whitespace_char(__global const char* chars, __global const uint* prevIsSpace, __global const uint* charCount, __global uint* shouldKeep) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = charCount[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    char c = chars[gid];
    int cInt = ((int)(c));
    bool isSpace = (cInt == 32);
    bool isTab = (cInt == 9);
    bool isNewline = (cInt == 10);
    bool isCR = (cInt == 13);
    bool isWhitespace = (isSpace || (isTab || (isNewline || isCR)));
    uint prevSpace = prevIsSpace[gid];
    bool prevWasSpace = (prevSpace == (uint)(1u));
    bool notWhitespace = (isWhitespace == false);
    bool firstWhitespace = (isWhitespace && (prevWasSpace == false));
    bool keep = (notWhitespace || firstWhitespace);
    uint keepFlag = ((keep) ? ((uint)(1u)) : ((uint)(0u)));
    shouldKeep[gid] = keepFlag;
  }
}
