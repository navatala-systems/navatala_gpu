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

#include "VofPressureOrchestrator.H"

namespace Navatala
{
namespace Cfd
{

VofPressureOrchestrator::VofPressureOrchestrator(
    GpuRuntime::Device& device,
    GpuRuntime::Queue& queue,
    FieldAdapter& adapter,
    FallbackHandler& fallbackHandler,
    const Config& config
)
:
    device_(device),
    queue_(queue),
    adapter_(adapter),
    fallbackHandler_(fallbackHandler),
    config_(config)
{}

bool VofPressureOrchestrator::execute(Foam::Time& runTime, const Foam::fvMesh& mesh)
{
    // FieldSync: owner → device
    adapter_.fieldSync("owner", true);
    // FieldSync: neighbour → device
    adapter_.fieldSync("neighbour", true);
    // FieldSync: weights → device
    adapter_.fieldSync("weights", true);
    // FieldSync: offsets → device
    adapter_.fieldSync("offsets", true);
    // FieldSync: faceIdx → device
    adapter_.fieldSync("faceIdx", true);
    // FieldSync: sign → device
    adapter_.fieldSync("sign", true);
    // FieldSync: signF → device
    adapter_.fieldSync("signF", true);
    // FieldSync: counts → device
    adapter_.fieldSync("counts", true);
    // FieldSync: params → device
    adapter_.fieldSync("params", true);
    // FieldSync: paramsF → device
    adapter_.fieldSync("paramsF", true);
    // FieldSync: sfX → device
    adapter_.fieldSync("sfX", true);
    // FieldSync: sfY → device
    adapter_.fieldSync("sfY", true);
    // FieldSync: sfZ → device
    adapter_.fieldSync("sfZ", true);
    // FieldSync: magSf → device
    adapter_.fieldSync("magSf", true);
    // FieldSync: vol → device
    adapter_.fieldSync("vol", true);
    // FieldSync: delta → device
    adapter_.fieldSync("delta", true);
    // FieldSync: deltaCoeffs → device
    adapter_.fieldSync("deltaCoeffs", true);
    // FieldSync: cf → device
    adapter_.fieldSync("cf", true);
    // FieldSync: coeff → device
    adapter_.fieldSync("coeff", true);
    // FieldSync: bcX → device
    adapter_.fieldSync("bcX", true);
    // FieldSync: bcY → device
    adapter_.fieldSync("bcY", true);
    // FieldSync: bcZ → device
    adapter_.fieldSync("bcZ", true);
    // FieldSync: bcVal → device
    adapter_.fieldSync("bcVal", true);
    // FieldSync: bcMask → device
    adapter_.fieldSync("bcMask", true);
    // FieldSync: bcSnGrad → device
    adapter_.fieldSync("bcSnGrad", true);
    // FieldSync: bcSnGradMask → device
    adapter_.fieldSync("bcSnGradMask", true);
    // FieldSync: rhoBcVal → device
    adapter_.fieldSync("rhoBcVal", true);
    // FieldSync: rhoBcMask → device
    adapter_.fieldSync("rhoBcMask", true);
    // FieldSync: alpha1 → device
    adapter_.fieldSync("alpha1", true);
    // FieldSync: alpha2 → device
    adapter_.fieldSync("alpha2", true);
    // FieldSync: p_rgh → device
    adapter_.fieldSync("p_rgh", true);
    // FieldSync: pCell → device
    adapter_.fieldSync("pCell", true);
    // FieldSync: ux → device
    adapter_.fieldSync("ux", true);
    // FieldSync: uy → device
    adapter_.fieldSync("uy", true);
    // FieldSync: uz → device
    adapter_.fieldSync("uz", true);
    // FieldSync: hbx → device
    adapter_.fieldSync("hbx", true);
    // FieldSync: hby → device
    adapter_.fieldSync("hby", true);
    // FieldSync: hbz → device
    adapter_.fieldSync("hbz", true);
    // FieldSync: rAU → device
    adapter_.fieldSync("rAU", true);
    // FieldSync: rAUf → device
    adapter_.fieldSync("rAUf", true);
    // FieldSync: phig → device
    adapter_.fieldSync("phig", true);
    // FieldSync: faceFlux → device
    adapter_.fieldSync("faceFlux", true);
    // FieldSync: faceFluxCorrection → device
    adapter_.fieldSync("faceFluxCorrection", true);
    // FieldSync: b → device
    adapter_.fieldSync("b", true);
    // FieldSync: matDiag → device
    adapter_.fieldSync("matDiag", true);
    // FieldSync: matUpper → device
    adapter_.fieldSync("matUpper", true);
    // FieldSync: matLower → device
    adapter_.fieldSync("matLower", true);
    // FieldSync: rhs → device
    adapter_.fieldSync("rhs", true);
    // FieldSync: p_pcg → device
    adapter_.fieldSync("p_pcg", true);
    // FieldSync: x → device
    adapter_.fieldSync("x", true);
    while (runTime.loop()) {
      // FieldSync: U → host
      adapter_.fieldSync("U", false);
      // FieldSync: phi → host
      adapter_.fieldSync("phi", false);
      // @FFI: U.oldTime()
      bool ok = U.oldTime();
      // @FFI: phi.oldTime()
      bool ok = phi.oldTime();
      // FieldSync: U → device
      adapter_.fieldSync("U", true);
      {
        /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Primitives.PhiFromU.phiFromU", device); */

        auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Primitives.PhiFromU.phiFromU");
        if (!prog) return false;
        KernelBindingBuilder binding;
        if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Primitives.PhiFromU.phiFromU", binding)) return false;
        auto* buf_owner = findBufferWithWorkspace(buffers, workspace, "owner");
        if (!buf_owner) return false;
        binding.bind("owner", buf_owner);
        auto* buf_neighbour = findBufferWithWorkspace(buffers, workspace, "neighbour");
        if (!buf_neighbour) return false;
        binding.bind("neighbour", buf_neighbour);
        auto* buf_weights = findBufferWithWorkspace(buffers, workspace, "weights");
        if (!buf_weights) return false;
        binding.bind("weights", buf_weights);
        auto* buf_sfX = findBufferWithWorkspace(buffers, workspace, "sfX");
        if (!buf_sfX) return false;
        binding.bind("sfX", buf_sfX);
        auto* buf_sfY = findBufferWithWorkspace(buffers, workspace, "sfY");
        if (!buf_sfY) return false;
        binding.bind("sfY", buf_sfY);
        auto* buf_sfZ = findBufferWithWorkspace(buffers, workspace, "sfZ");
        if (!buf_sfZ) return false;
        binding.bind("sfZ", buf_sfZ);
        auto* buf_ux = findBufferWithWorkspace(buffers, workspace, "ux");
        if (!buf_ux) return false;
        binding.bind("ux", buf_ux);
        auto* buf_uy = findBufferWithWorkspace(buffers, workspace, "uy");
        if (!buf_uy) return false;
        binding.bind("uy", buf_uy);
        auto* buf_uz = findBufferWithWorkspace(buffers, workspace, "uz");
        if (!buf_uz) return false;
        binding.bind("uz", buf_uz);
        auto* buf_bcX = findBufferWithWorkspace(buffers, workspace, "bcX");
        if (!buf_bcX) return false;
        binding.bind("bcX", buf_bcX);
        auto* buf_bcY = findBufferWithWorkspace(buffers, workspace, "bcY");
        if (!buf_bcY) return false;
        binding.bind("bcY", buf_bcY);
        auto* buf_bcZ = findBufferWithWorkspace(buffers, workspace, "bcZ");
        if (!buf_bcZ) return false;
        binding.bind("bcZ", buf_bcZ);
        auto* buf_bcMask = findBufferWithWorkspace(buffers, workspace, "bcMask");
        if (!buf_bcMask) return false;
        binding.bind("bcMask", buf_bcMask);
        auto* buf_params = findBufferWithWorkspace(buffers, workspace, "params");
        if (!buf_params) return false;
        binding.bind("params", buf_params);
        auto* buf_outPhi = findBufferWithWorkspace(buffers, workspace, "phi");
        if (!buf_outPhi) return false;
        binding.bind("outPhi", buf_outPhi);
        FinalizedKernelBinding finalizedBinding;
        std::string bindingError;
        if (!binding.finalize(finalizedBinding, &bindingError)) return false;
        queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
      }
      adapter_.boundaryRefresh("U");
      adapter_.boundaryRefresh("p_rgh");
      // FieldSync: phi → host
      adapter_.fieldSync("phi", false);
      adapter_.haloExchange("U");
      adapter_.haloExchange("p_rgh");
      adapter_.haloExchange("phi");
      // FieldSync: phi → host
      adapter_.fieldSync("phi", false);
      // @FFI: alphaEqn.solve(alpha1, phi, alphaScheme)
      bool ok = alphaEqn.solve(alpha1, phi, alphaScheme);
      // FieldSync: alpha1 → device
      adapter_.fieldSync("alpha1", true);
      // FieldSync: alpha2 → device
      adapter_.fieldSync("alpha2", true);
      {
        // ConservationCheck: alphaBounded
        adapter_.fieldSync("alpha1", false);
        // readsOutcomes: none
        // Runtime conservation-law evaluation is lowered separately;
        // this node currently materializes dependency sync placement.
      }
      // @FFI: mixture.correct()
      bool ok = mixture.correct();
      {
        /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Primitives.RhoFromAlpha.rhoFromAlpha", device); */

        auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Primitives.RhoFromAlpha.rhoFromAlpha");
        if (!prog) return false;
        KernelBindingBuilder binding;
        if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Primitives.RhoFromAlpha.rhoFromAlpha", binding)) return false;
        auto* buf_alpha = findBufferWithWorkspace(buffers, workspace, "alpha1");
        if (!buf_alpha) return false;
        binding.bind("alpha", buf_alpha);
        auto* buf_paramsF = findBufferWithWorkspace(buffers, workspace, "paramsF");
        if (!buf_paramsF) return false;
        binding.bind("paramsF", buf_paramsF);
        auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
        if (!buf_counts) return false;
        binding.bind("counts", buf_counts);
        auto* buf_outRho = findBufferWithWorkspace(buffers, workspace, "rho");
        if (!buf_outRho) return false;
        binding.bind("outRho", buf_outRho);
        FinalizedKernelBinding finalizedBinding;
        std::string bindingError;
        if (!binding.finalize(finalizedBinding, &bindingError)) return false;
        queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
      }
      {
        // ConservationCheck: totalMass
        adapter_.fieldSync("rho", false);
        // readsOutcomes: none
        // Runtime conservation-law evaluation is lowered separately;
        // this node currently materializes dependency sync placement.
      }
      {
        /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Primitives.RhoPhiFromRhoAndPhi.rhoPhiFromRhoAndPhi", device); */

        auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Primitives.RhoPhiFromRhoAndPhi.rhoPhiFromRhoAndPhi");
        if (!prog) return false;
        KernelBindingBuilder binding;
        if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Primitives.RhoPhiFromRhoAndPhi.rhoPhiFromRhoAndPhi", binding)) return false;
        auto* buf_phiAll = findBufferWithWorkspace(buffers, workspace, "phi");
        if (!buf_phiAll) return false;
        binding.bind("phiAll", buf_phiAll);
        auto* buf_rhoCell = findBufferWithWorkspace(buffers, workspace, "rho");
        if (!buf_rhoCell) return false;
        binding.bind("rhoCell", buf_rhoCell);
        auto* buf_owner = findBufferWithWorkspace(buffers, workspace, "owner");
        if (!buf_owner) return false;
        binding.bind("owner", buf_owner);
        auto* buf_neighbour = findBufferWithWorkspace(buffers, workspace, "neighbour");
        if (!buf_neighbour) return false;
        binding.bind("neighbour", buf_neighbour);
        auto* buf_weights = findBufferWithWorkspace(buffers, workspace, "weights");
        if (!buf_weights) return false;
        binding.bind("weights", buf_weights);
        auto* buf_rhoBcVal = findBufferWithWorkspace(buffers, workspace, "rhoBcVal");
        if (!buf_rhoBcVal) return false;
        binding.bind("rhoBcVal", buf_rhoBcVal);
        auto* buf_rhoBcMask = findBufferWithWorkspace(buffers, workspace, "rhoBcMask");
        if (!buf_rhoBcMask) return false;
        binding.bind("rhoBcMask", buf_rhoBcMask);
        auto* buf_params = findBufferWithWorkspace(buffers, workspace, "params");
        if (!buf_params) return false;
        binding.bind("params", buf_params);
        auto* buf_outRhoPhi = findBufferWithWorkspace(buffers, workspace, "rhoPhi");
        if (!buf_outRhoPhi) return false;
        binding.bind("outRhoPhi", buf_outRhoPhi);
        FinalizedKernelBinding finalizedBinding;
        std::string bindingError;
        if (!binding.finalize(finalizedBinding, &bindingError)) return false;
        queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
      }
      // FieldSync: U → host
      adapter_.fieldSync("U", false);
      // FieldSync: phi → host
      adapter_.fieldSync("phi", false);
      // FieldSync: rho → host
      adapter_.fieldSync("rho", false);
      // FieldSync: rhoPhi → host
      adapter_.fieldSync("rhoPhi", false);
      // @FFI: turbulence->correct()
      bool ok = turbulence->correct();
      // FieldSync: nut → device
      adapter_.fieldSync("nut", true);
      for (int __outer = 0; __outer < 3; ++__outer) {
        // @FFI: momentumPredictor(UEqn, rho, p_rgh, ghf, rhof)
        bool ok = momentumPredictor(UEqn, rho, p_rgh, ghf, rhof);
        // FieldSync: HbyA → device
        adapter_.fieldSync("HbyA", true);
        // FieldSync: rAU → device
        adapter_.fieldSync("rAU", true);
        // FieldSync: phiHbyA → device
        adapter_.fieldSync("phiHbyA", true);
        for (int __corr = 0; __corr < 1; ++__corr) {
          // Begin solver call: pEqn for field p_rgh
          {
            {
              /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.laplacianFromFaceFlux", device); */

              auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.laplacianFromFaceFlux");
              if (!prog) return false;
              KernelBindingBuilder binding;
              if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.laplacianFromFaceFlux", binding)) return false;
              auto* buf_faceFlux = findBufferWithWorkspace(buffers, workspace, "faceFlux");
              if (!buf_faceFlux) return false;
              binding.bind("faceFlux", buf_faceFlux);
              auto* buf_offsets = findBufferWithWorkspace(buffers, workspace, "offsets");
              if (!buf_offsets) return false;
              binding.bind("offsets", buf_offsets);
              auto* buf_faceIdx = findBufferWithWorkspace(buffers, workspace, "faceIdx");
              if (!buf_faceIdx) return false;
              binding.bind("faceIdx", buf_faceIdx);
              auto* buf_signF = findBufferWithWorkspace(buffers, workspace, "signF");
              if (!buf_signF) return false;
              binding.bind("signF", buf_signF);
              auto* buf_vol = findBufferWithWorkspace(buffers, workspace, "vol");
              if (!buf_vol) return false;
              binding.bind("vol", buf_vol);
              auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
              if (!buf_counts) return false;
              binding.bind("counts", buf_counts);
              auto* buf_outLap = findBufferWithWorkspace(buffers, workspace, "ax");
              if (!buf_outLap) return false;
              binding.bind("outLap", buf_outLap);
              FinalizedKernelBinding finalizedBinding;
              std::string bindingError;
              if (!binding.finalize(finalizedBinding, &bindingError)) return false;
              queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
            }
            {
              /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.bcSnGradFaceFlux", device); */

              auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.bcSnGradFaceFlux");
              if (!prog) return false;
              KernelBindingBuilder binding;
              if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.bcSnGradFaceFlux", binding)) return false;
              auto* buf_cf = findBufferWithWorkspace(buffers, workspace, "cf");
              if (!buf_cf) return false;
              binding.bind("cf", buf_cf);
              auto* buf_delta = findBufferWithWorkspace(buffers, workspace, "delta");
              if (!buf_delta) return false;
              binding.bind("delta", buf_delta);
              auto* buf_bcSnGrad = findBufferWithWorkspace(buffers, workspace, "bcSnGrad");
              if (!buf_bcSnGrad) return false;
              binding.bind("bcSnGrad", buf_bcSnGrad);
              auto* buf_bcSnGradMask = findBufferWithWorkspace(buffers, workspace, "bcSnGradMask");
              if (!buf_bcSnGradMask) return false;
              binding.bind("bcSnGradMask", buf_bcSnGradMask);
              auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
              if (!buf_counts) return false;
              binding.bind("counts", buf_counts);
              auto* buf_outFlux = findBufferWithWorkspace(buffers, workspace, "bndSrc");
              if (!buf_outFlux) return false;
              binding.bind("outFlux", buf_outFlux);
              FinalizedKernelBinding finalizedBinding;
              std::string bindingError;
              if (!binding.finalize(finalizedBinding, &bindingError)) return false;
              queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
            }
            {
              /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.rhsSub", device); */

              auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.rhsSub");
              if (!prog) return false;
              KernelBindingBuilder binding;
              if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.rhsSub", binding)) return false;
              auto* buf_b = findBufferWithWorkspace(buffers, workspace, "b");
              if (!buf_b) return false;
              binding.bind("b", buf_b);
              auto* buf_ax = findBufferWithWorkspace(buffers, workspace, "ax");
              if (!buf_ax) return false;
              binding.bind("ax", buf_ax);
              auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
              if (!buf_counts) return false;
              binding.bind("counts", buf_counts);
              auto* buf_r = findBufferWithWorkspace(buffers, workspace, "r");
              if (!buf_r) return false;
              binding.bind("r", buf_r);
              FinalizedKernelBinding finalizedBinding;
              std::string bindingError;
              if (!binding.finalize(finalizedBinding, &bindingError)) return false;
              queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
            }
            // @SafetyContract(SignConvention): atomic toSPD transformation
            // Negates ALL components simultaneously — matrix + RHS + boundary
            {
              auto* buf_diag = findBufferWithWorkspace(buffers, workspace, "matDiag");
              auto* buf_upper = findBufferWithWorkspace(buffers, workspace, "matUpper");
              auto* buf_lower = findBufferWithWorkspace(buffers, workspace, "matLower");
              auto* buf_rhs = findBufferWithWorkspace(buffers, workspace, "rhs");
              auto* buf_bnd = findBufferWithWorkspace(buffers, workspace, "bndSrc");
              if (!buf_diag || !buf_upper || !buf_lower || !buf_rhs || !buf_bnd) return false;
              // Negate all coefficient arrays for SPD convention
              negateBuffer(queue, *buf_diag);
              negateBuffer(queue, *buf_upper);
              negateBuffer(queue, *buf_lower);
              negateBuffer(queue, *buf_rhs);
              negateBuffer(queue, *buf_bnd);
            }
            for (int __iter = 0; __iter < 500; ++__iter) {
              {
              const bool __cond = true;
              if (!__cond) break;
              }
              {
                /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.invDiag", device); */

                auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.invDiag");
                if (!prog) return false;
                KernelBindingBuilder binding;
                if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.invDiag", binding)) return false;
                auto* buf_r = findBufferWithWorkspace(buffers, workspace, "r");
                if (!buf_r) return false;
                binding.bind("r", buf_r);
                auto* buf_diag = findBufferWithWorkspace(buffers, workspace, "matDiag");
                if (!buf_diag) return false;
                binding.bind("diag", buf_diag);
                auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
                if (!buf_counts) return false;
                binding.bind("counts", buf_counts);
                auto* buf_z = findBufferWithWorkspace(buffers, workspace, "z");
                if (!buf_z) return false;
                binding.bind("z", buf_z);
                FinalizedKernelBinding finalizedBinding;
                std::string bindingError;
                if (!binding.finalize(finalizedBinding, &bindingError)) return false;
                queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
              }
              {
                /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.axpyInPlace", device); */

                auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.axpyInPlace");
                if (!prog) return false;
                KernelBindingBuilder binding;
                if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.axpyInPlace", binding)) return false;
                auto* buf_x = findBufferWithWorkspace(buffers, workspace, "p_pcg");
                if (!buf_x) return false;
                binding.bind("x", buf_x);
                auto* buf_y = findBufferWithWorkspace(buffers, workspace, "x");
                if (!buf_y) return false;
                binding.bind("y", buf_y);
                auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
                if (!buf_counts) return false;
                binding.bind("counts", buf_counts);
                auto* buf_paramsF = findBufferWithWorkspace(buffers, workspace, "paramsF");
                if (!buf_paramsF) return false;
                binding.bind("paramsF", buf_paramsF);
                FinalizedKernelBinding finalizedBinding;
                std::string bindingError;
                if (!binding.finalize(finalizedBinding, &bindingError)) return false;
                queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
              }
              {
                /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.dotPartials", device); */

                auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.dotPartials");
                if (!prog) return false;
                KernelBindingBuilder binding;
                if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PoissonPcg.dotPartials", binding)) return false;
                auto* buf_a = findBufferWithWorkspace(buffers, workspace, "r");
                if (!buf_a) return false;
                binding.bind("a", buf_a);
                auto* buf_b = findBufferWithWorkspace(buffers, workspace, "z");
                if (!buf_b) return false;
                binding.bind("b", buf_b);
                auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
                if (!buf_counts) return false;
                binding.bind("counts", buf_counts);
                auto* buf_outPartials = findBufferWithWorkspace(buffers, workspace, "dotOut");
                if (!buf_outPartials) return false;
                binding.bind("outPartials", buf_outPartials);
                FinalizedKernelBinding finalizedBinding;
                std::string bindingError;
                if (!binding.finalize(finalizedBinding, &bindingError)) return false;
                queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
              }
              // @SafetyContract(ConvergenceGuard): typed convergence check
              {
                auto* buf_res = findBufferWithWorkspace(buffers, workspace, "residual");
                auto* buf_tol = findBufferWithWorkspace(buffers, workspace, "tolerance");
                auto* buf_it = findBufferWithWorkspace(buffers, workspace, "iters");
                if (!buf_res || !buf_tol || !buf_it) return false;
                queue.synchronize();
                float residual = *static_cast<float*>(buf_res->getHostPointer());
                float tolerance = *static_cast<float*>(buf_tol->getHostPointer());
                int iters = *static_cast<int*>(buf_it->getHostPointer());
                if (!std::isfinite(residual)) {
                  // NonFinite: always fatal (R7 invariant)
                  return false;
                }
                if (iters >= 500 && residual > tolerance) {
                  // MaxIters: route through fallback policy
                  break; // exit convergence loop
                }
                if (residual <= tolerance) {
                  break; // converged
                }
              }
            }
            // Convergence routing: nonFinite is always fatal
            auto* buf_residual = findBufferWithWorkspace(buffers, workspace, "residual");
            auto* buf_tolerance = findBufferWithWorkspace(buffers, workspace, "tolerance");
            auto* buf_iters = findBufferWithWorkspace(buffers, workspace, "iters");
            if (!buf_residual || !buf_tolerance || !buf_iters) return false;
            queue.synchronize();
            float residual = *static_cast<float*>(buf_residual->getHostPointer());
            float tolerance = *static_cast<float*>(buf_tolerance->getHostPointer());
            int iters = *static_cast<int*>(buf_iters->getHostPointer());
            if (!std::isfinite(residual)) {
              fallbackHandler_.fatal("pEqn", "p_rgh"); return false;
            } else if (iters >= 500 && residual > tolerance) {
            fallbackHandler_.fallbackCPU("pEqn", "p_rgh"); // onMaxIters
            } else if (residual > tolerance * 1e6f) {
            fallbackHandler_.abort("pEqn", "p_rgh"); return false; // onDiverged
            }
          }
        }
        {
          /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.UCorrectReconstruct.uCorrectReconstruct", device); */

          auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.UCorrectReconstruct.uCorrectReconstruct");
          if (!prog) return false;
          KernelBindingBuilder binding;
          if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.UCorrectReconstruct.uCorrectReconstruct", binding)) return false;
          auto* buf_pCell = findBufferWithWorkspace(buffers, workspace, "pCell");
          if (!buf_pCell) return false;
          binding.bind("pCell", buf_pCell);
          auto* buf_rAU = findBufferWithWorkspace(buffers, workspace, "rAU");
          if (!buf_rAU) return false;
          binding.bind("rAU", buf_rAU);
          auto* buf_hbx = findBufferWithWorkspace(buffers, workspace, "hbx");
          if (!buf_hbx) return false;
          binding.bind("hbx", buf_hbx);
          auto* buf_hby = findBufferWithWorkspace(buffers, workspace, "hby");
          if (!buf_hby) return false;
          binding.bind("hby", buf_hby);
          auto* buf_hbz = findBufferWithWorkspace(buffers, workspace, "hbz");
          if (!buf_hbz) return false;
          binding.bind("hbz", buf_hbz);
          auto* buf_rAUf = findBufferWithWorkspace(buffers, workspace, "rAUf");
          if (!buf_rAUf) return false;
          binding.bind("rAUf", buf_rAUf);
          auto* buf_phig = findBufferWithWorkspace(buffers, workspace, "phig");
          if (!buf_phig) return false;
          binding.bind("phig", buf_phig);
          auto* buf_owner = findBufferWithWorkspace(buffers, workspace, "owner");
          if (!buf_owner) return false;
          binding.bind("owner", buf_owner);
          auto* buf_neighbour = findBufferWithWorkspace(buffers, workspace, "neighbour");
          if (!buf_neighbour) return false;
          binding.bind("neighbour", buf_neighbour);
          auto* buf_sfX = findBufferWithWorkspace(buffers, workspace, "sfX");
          if (!buf_sfX) return false;
          binding.bind("sfX", buf_sfX);
          auto* buf_sfY = findBufferWithWorkspace(buffers, workspace, "sfY");
          if (!buf_sfY) return false;
          binding.bind("sfY", buf_sfY);
          auto* buf_sfZ = findBufferWithWorkspace(buffers, workspace, "sfZ");
          if (!buf_sfZ) return false;
          binding.bind("sfZ", buf_sfZ);
          auto* buf_magSf = findBufferWithWorkspace(buffers, workspace, "magSf");
          if (!buf_magSf) return false;
          binding.bind("magSf", buf_magSf);
          auto* buf_deltaCoeffs = findBufferWithWorkspace(buffers, workspace, "deltaCoeffs");
          if (!buf_deltaCoeffs) return false;
          binding.bind("deltaCoeffs", buf_deltaCoeffs);
          auto* buf_offsets = findBufferWithWorkspace(buffers, workspace, "offsets");
          if (!buf_offsets) return false;
          binding.bind("offsets", buf_offsets);
          auto* buf_faceIdx = findBufferWithWorkspace(buffers, workspace, "faceIdx");
          if (!buf_faceIdx) return false;
          binding.bind("faceIdx", buf_faceIdx);
          auto* buf_sign = findBufferWithWorkspace(buffers, workspace, "sign");
          if (!buf_sign) return false;
          binding.bind("sign", buf_sign);
          auto* buf_vol = findBufferWithWorkspace(buffers, workspace, "vol");
          if (!buf_vol) return false;
          binding.bind("vol", buf_vol);
          auto* buf_bcVal = findBufferWithWorkspace(buffers, workspace, "bcVal");
          if (!buf_bcVal) return false;
          binding.bind("bcVal", buf_bcVal);
          auto* buf_bcMask = findBufferWithWorkspace(buffers, workspace, "bcMask");
          if (!buf_bcMask) return false;
          binding.bind("bcMask", buf_bcMask);
          auto* buf_bcSnGrad = findBufferWithWorkspace(buffers, workspace, "bcSnGrad");
          if (!buf_bcSnGrad) return false;
          binding.bind("bcSnGrad", buf_bcSnGrad);
          auto* buf_bcSnGradMask = findBufferWithWorkspace(buffers, workspace, "bcSnGradMask");
          if (!buf_bcSnGradMask) return false;
          binding.bind("bcSnGradMask", buf_bcSnGradMask);
          auto* buf_faceFluxCorrection = findBufferWithWorkspace(buffers, workspace, "faceFluxCorrection");
          if (!buf_faceFluxCorrection) return false;
          binding.bind("faceFluxCorrection", buf_faceFluxCorrection);
          auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
          if (!buf_counts) return false;
          binding.bind("counts", buf_counts);
          auto* buf_paramsF = findBufferWithWorkspace(buffers, workspace, "paramsF");
          if (!buf_paramsF) return false;
          binding.bind("paramsF", buf_paramsF);
          auto* buf_outX = findBufferWithWorkspace(buffers, workspace, "ux");
          if (!buf_outX) return false;
          binding.bind("outX", buf_outX);
          auto* buf_outY = findBufferWithWorkspace(buffers, workspace, "uy");
          if (!buf_outY) return false;
          binding.bind("outY", buf_outY);
          auto* buf_outZ = findBufferWithWorkspace(buffers, workspace, "uz");
          if (!buf_outZ) return false;
          binding.bind("outZ", buf_outZ);
          FinalizedKernelBinding finalizedBinding;
          std::string bindingError;
          if (!binding.finalize(finalizedBinding, &bindingError)) return false;
          queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
        }
        adapter_.boundaryRefresh("U");
        adapter_.haloExchange("U");
        {
          /* std::string selectedVariant = selectOptimalVariant("Calculations.GPU.NavatalaCFD.Pressure.PhiCorrect.phiCorrect", device); */

          auto* prog = getOrCreateProgram(device, backend, "Calculations.GPU.NavatalaCFD.Pressure.PhiCorrect.phiCorrect");
          if (!prog) return false;
          KernelBindingBuilder binding;
          if (!KernelBindings::make(backend, "Calculations.GPU.NavatalaCFD.Pressure.PhiCorrect.phiCorrect", binding)) return false;
          auto* buf_phiIn = findBufferWithWorkspace(buffers, workspace, "phi");
          if (!buf_phiIn) return false;
          binding.bind("phiIn", buf_phiIn);
          auto* buf_pCell = findBufferWithWorkspace(buffers, workspace, "pCell");
          if (!buf_pCell) return false;
          binding.bind("pCell", buf_pCell);
          auto* buf_owner = findBufferWithWorkspace(buffers, workspace, "owner");
          if (!buf_owner) return false;
          binding.bind("owner", buf_owner);
          auto* buf_neighbour = findBufferWithWorkspace(buffers, workspace, "neighbour");
          if (!buf_neighbour) return false;
          binding.bind("neighbour", buf_neighbour);
          auto* buf_magSf = findBufferWithWorkspace(buffers, workspace, "magSf");
          if (!buf_magSf) return false;
          binding.bind("magSf", buf_magSf);
          auto* buf_deltaCoeffs = findBufferWithWorkspace(buffers, workspace, "deltaCoeffs");
          if (!buf_deltaCoeffs) return false;
          binding.bind("deltaCoeffs", buf_deltaCoeffs);
          auto* buf_coeff = findBufferWithWorkspace(buffers, workspace, "coeff");
          if (!buf_coeff) return false;
          binding.bind("coeff", buf_coeff);
          auto* buf_bcVal = findBufferWithWorkspace(buffers, workspace, "bcVal");
          if (!buf_bcVal) return false;
          binding.bind("bcVal", buf_bcVal);
          auto* buf_bcMask = findBufferWithWorkspace(buffers, workspace, "bcMask");
          if (!buf_bcMask) return false;
          binding.bind("bcMask", buf_bcMask);
          auto* buf_bcSnGrad = findBufferWithWorkspace(buffers, workspace, "bcSnGrad");
          if (!buf_bcSnGrad) return false;
          binding.bind("bcSnGrad", buf_bcSnGrad);
          auto* buf_bcSnGradMask = findBufferWithWorkspace(buffers, workspace, "bcSnGradMask");
          if (!buf_bcSnGradMask) return false;
          binding.bind("bcSnGradMask", buf_bcSnGradMask);
          auto* buf_faceFluxCorrection = findBufferWithWorkspace(buffers, workspace, "faceFluxCorrection");
          if (!buf_faceFluxCorrection) return false;
          binding.bind("faceFluxCorrection", buf_faceFluxCorrection);
          auto* buf_counts = findBufferWithWorkspace(buffers, workspace, "counts");
          if (!buf_counts) return false;
          binding.bind("counts", buf_counts);
          auto* buf_paramsF = findBufferWithWorkspace(buffers, workspace, "paramsF");
          if (!buf_paramsF) return false;
          binding.bind("paramsF", buf_paramsF);
          auto* buf_phiOut = findBufferWithWorkspace(buffers, workspace, "phi");
          if (!buf_phiOut) return false;
          binding.bind("phiOut", buf_phiOut);
          FinalizedKernelBinding finalizedBinding;
          std::string bindingError;
          if (!binding.finalize(finalizedBinding, &bindingError)) return false;
          queue.submit(*prog, finalizedBinding.args(), 1, 1, 1, 256, 1, 1);
        }
        // FieldSync: phi → host
        adapter_.fieldSync("phi", false);
        adapter_.haloExchange("phi");
        {
          // ConservationCheck: cellContinuity
          adapter_.fieldSync("phi", false);
          // readsOutcomes: none
          // Runtime conservation-law evaluation is lowered separately;
          // this node currently materializes dependency sync placement.
        }
      }
      // FieldSync: U → host
      adapter_.fieldSync("U", false);
      // FieldSync: p_rgh → host
      adapter_.fieldSync("p_rgh", false);
      // FieldSync: phi → host
      adapter_.fieldSync("phi", false);
      // FieldSync: alpha1 → host
      adapter_.fieldSync("alpha1", false);
      // @FFI: diagnostics(runTime, mesh, U, p_rgh, phi, alpha1)
      bool ok = diagnostics(runTime, mesh, U, p_rgh, phi, alpha1);
      // Generated write policy (CheckpointOutput, interval=500)
      if (runTime.writeTime() || (runTime.timeIndex() % 500 == 0)) {
        adapter_.syncAllToHost();
        runTime.write();
      }
    }
    return true;
}

} // namespace Cfd
} // namespace Navatala