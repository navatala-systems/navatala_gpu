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
#include "volFields.H"
#include "surfaceFields.H"

#include "FieldAdapter.H"
#include "FallbackHandler.H"
#include "gpu_runtime.h"

namespace Navatala
{
namespace Cfd
{

class VofPressureOrchestrator
{
public:
    struct Config
    {
        Foam::label nOuterCorr = 3;
        Foam::label nNonOrthCorr = 1;
        Foam::label nSubCycles = 1;
        Foam::label pcgIters = 500;
        Foam::scalar pcgTol = 1e-6;
        Foam::scalar pcgRelTol = 0.01;
        bool cpuPressure = false;
    };

    VofPressureOrchestrator(
        GpuRuntime::Device& device,
        GpuRuntime::Queue& queue,
        FieldAdapter& adapter,
        FallbackHandler& fallbackHandler,
        const Config& config
    );

    // Execute the full orchestrator — replaces the entire step loop.
    // Returns true on success, false on fatal failure.
    bool execute(Foam::Time& runTime, const Foam::fvMesh& mesh);

private:
    GpuRuntime::Device& device_;
    GpuRuntime::Queue& queue_;
    FieldAdapter& adapter_;
    FallbackHandler& fallbackHandler_;
    Config config_;
};

} // namespace Cfd
} // namespace Navatala