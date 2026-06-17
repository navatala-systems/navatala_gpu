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

#pragma once

#include "fvMesh.H"
#include "Time.H"

namespace Navatala
{
namespace Cfd
{

class FallbackHandler
{
public:
    FallbackHandler(Foam::Time& runTime, const Foam::fvMesh& mesh);

    void fatal(const char* eqn, const char* field);
    void fallbackCPU(const char* eqn, const char* field);
    void reduceDt(const char* eqn, const char* field);
    void retry(const char* eqn, const char* field);
    void abort(const char* eqn, const char* field);

private:
    Foam::Time& runTime_;
    const Foam::fvMesh& mesh_;
};

} // namespace Cfd
} // namespace Navatala