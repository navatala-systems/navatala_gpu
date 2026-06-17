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

#include "navatala/_safety_contracts.hpp"

#include <cstring>
#include <sstream>

namespace NavatalaRegistry { namespace R4Safety {

static const char* safe(const char* s) { return s == nullptr ? "" : s; }

static void appendJsonString(std::ostringstream& os, const char* value) {
  os << '"';
  for (const char* p = safe(value); *p != '\0'; ++p) {
    if (*p == '"' || *p == '\\') os << '\\' << *p;
    else if (*p == '\n') os << "\\n";
    else if (*p == '\r') os << "\\r";
    else if (*p == '\t') os << "\\t";
    else os << *p;
  }
  os << '"';
}

std::string formatPhaseTimingJson(const PhaseTimingRecord& r) {
  std::ostringstream os;
  os << '{';
  os << "\"kind\":\"PHASE_TIMING\",";
  os << "\"phaseName\":"; appendJsonString(os, r.phaseName); os << ',';
  os << "\"rank\":" << r.rank << ',';
  os << "\"backend\":"; appendJsonString(os, r.backend); os << ',';
  os << "\"hostMs\":" << r.hostMs << ',';
  os << "\"deviceMs\":" << r.deviceMs << ',';
  os << "\"mpiWaitMs\":" << r.mpiWaitMs << ',';
  os << "\"h2dBytes\":" << r.h2dBytes << ',';
  os << "\"d2hBytes\":" << r.d2hBytes << ',';
  os << "\"kernelLaunchCount\":" << r.kernelLaunchCount << ',';
  os << "\"allocatedBytes\":" << r.allocatedBytes << ',';
  os << "\"peakDeviceBytes\":" << r.peakDeviceBytes << ',';
  os << "\"fallbackUsed\":" << (r.fallbackUsed ? "true" : "false");
  os << '}';
  return os.str();
}

std::string formatHaloExchangeTrace(const HaloTraceRecord& r) {
  std::ostringstream os;
  os << "HALO_EXCHANGE"
     << " rank=" << r.rank
     << " neigh=" << r.neigh
     << " exchange=" << safe(r.exchange)
     << " module=" << safe(r.module)
     << " phase=" << safe(r.phase)
     << " timeIndex=" << r.timeIndex
     << " outerCorr=" << safe(r.outerCorr)
     << " pressureCorr=" << safe(r.pressureCorr)
     << " nonOrthCorr=" << safe(r.nonOrthCorr)
     << " alphaSubCycle=" << safe(r.alphaSubCycle)
     << " turbulenceCorr=" << safe(r.turbulenceCorr)
     << " tag=" << r.tag
     << " arity=" << r.arity
     << " type=" << safe(r.type)
     << " sendCount=" << r.sendCount
     << " recvCount=" << r.recvCount
     << " sendBytes=" << r.sendBytes
     << " recvBytes=" << r.recvBytes
     << " patchLocal=" << r.patchLocal
     << " patchRemote=" << r.patchRemote
     << " comm=" << safe(r.comm);
  return os.str();
}

std::string formatFieldStateTrace(const FieldStateRecord& r, const char* transfer, std::uint64_t bytes) {
  std::ostringstream os;
  os << "FIELD_STATE"
     << " phase=" << safe(r.phaseName)
     << " field=" << safe(r.fieldName)
     << " requirement=" << safe(r.phaseRequirement)
     << " hostInterior=" << safe(r.hostInterior)
     << " deviceInterior=" << safe(r.deviceInterior)
     << " hostBoundary=" << safe(r.hostBoundary)
     << " deviceBoundary=" << safe(r.deviceBoundary)
     << " halo=" << safe(r.processorHalo)
     << " oldTime=" << safe(r.oldTime)
     << " residency=" << safe(r.residencyPolicy)
     << " transferPolicy=" << safe(r.transferPolicy)
     << " lastProducer=" << safe(r.lastProducerPhase)
     << " lastConsumer=" << safe(r.lastConsumerPhase)
     << " requiresHaloFresh=" << (r.requiresHaloFresh ? "true" : "false")
     << " requiresBoundaryFresh=" << (r.requiresBoundaryFresh ? "true" : "false")
     << " transfer=" << safe(transfer)
     << " bytes=" << bytes;
  return os.str();
}

std::string formatMMatrixStats(const MMatrixRepairStatsRecord& r) {
  std::ostringstream os;
  os << "M_MATRIX_STATS"
     << " level=" << r.level
     << " positiveOffDiagCount=" << r.positiveOffDiagCount
     << " positiveOffDiagSum=" << r.positiveOffDiagSum
     << " rowSumDriftBefore=" << r.rowSumDriftBefore
     << " rowSumDriftAfter=" << r.rowSumDriftAfter
     << " diagMinBefore=" << r.diagMinBefore
     << " diagMinAfter=" << r.diagMinAfter
     << " repairApplied=" << (r.repairApplied ? "true" : "false");
  return os.str();
}

std::string formatSolverDecisionTrace(const SolverCompatibilityPolicy& p, bool accepted, const char* reason) {
  std::ostringstream os;
  os << "SOLVER_DECISION"
     << " solver=" << safe(p.solverKind)
     << " operatorClass=" << safe(p.operatorClass)
     << " preconditionerClass=" << safe(p.preconditionerClass)
     << " smoother=" << safe(p.smoother)
     << " variablePreconditioner=" << (p.allowVariablePreconditioner ? "true" : "false")
     << " recheckOnOperatorClassChange=" << (p.recheckOnOperatorClassChange ? "true" : "false")
     << " accepted=" << (accepted ? "true" : "false")
     << " reason=" << safe(reason);
  return os.str();
}

const char* firstExceededRunGateThreshold(const RunGateMetrics& observed, const RunGateMetrics& thresholds) {
  if (observed.fallbackCount > thresholds.fallbackCount) return "fallbackCount";
  if (observed.mpiErrorCount > thresholds.mpiErrorCount) return "mpiErrorCount";
  if (observed.nonFiniteCount > thresholds.nonFiniteCount) return "nonFiniteCount";
  if (observed.dtCollapseRatio > thresholds.dtCollapseRatio) return "dtCollapseRatio";
  if (observed.pressureIterationTrend > thresholds.pressureIterationTrend) return "pressureIterationTrend";
  if (observed.memoryGrowthPercent > thresholds.memoryGrowthPercent) return "memoryGrowthPercent";
  if (observed.rankDivergence > thresholds.rankDivergence) return "rankDivergence";
  return nullptr;
}

bool hasPassingRunGatePrerequisite(const RunGateDefinition& gate, const std::vector<RunGateRecord>& records, const char* gitCommit, const char* caseId) {
  if (std::strcmp(safe(gate.requiredPreviousGate), "") == 0) return true;
  for (const auto& r : records) {
    if (r.passed
        && std::strcmp(safe(r.kind), safe(gate.requiredPreviousGate)) == 0
        && std::strcmp(safe(r.gitCommit), safe(gitCommit)) == 0
        && std::strcmp(safe(r.caseId), safe(caseId)) == 0) {
      return true;
    }
  }
  return false;
}

bool validateFieldStateRecord(const FieldStateRecord& r) {
  if (std::strcmp(safe(r.phaseName), "") == 0 || std::strcmp(safe(r.fieldName), "") == 0) return false;
  if (std::strcmp(safe(r.phaseRequirement), "gpuRequired") == 0
      && std::strcmp(safe(r.residencyPolicy), "hostOnlyAllowed") == 0) return false;
  return true;
}

bool validateMMatrixRepairStats(const MMatrixRepairStatsRecord& r, double maxRowDrift) {
  if (!r.repairApplied) return true;
  const double drift = r.rowSumDriftAfter < 0 ? -r.rowSumDriftAfter : r.rowSumDriftAfter;
  return drift <= maxRowDrift;
}

static bool isCgCompatibleClass(const char* c) {
  return std::strcmp(safe(c), "spd") == 0 || std::strcmp(safe(c), "spsd") == 0;
}

bool solverCompatibilityAccepted(const SolverCompatibilityPolicy& p) {
  if (std::strcmp(safe(p.solverKind), "pcg") == 0) {
    return isCgCompatibleClass(p.operatorClass)
        && isCgCompatibleClass(p.preconditionerClass)
        && !p.allowVariablePreconditioner;
  }
  if (std::strcmp(safe(p.solverKind), "fcg") == 0
      || std::strcmp(safe(p.solverKind), "fgmres") == 0
      || std::strcmp(safe(p.solverKind), "standaloneGAMG") == 0) {
    return std::strcmp(safe(p.operatorClass), "unknown") != 0;
  }
  return false;
}

bool validateSignTransformVerificationPolicy(const SignTransformVerificationPolicy& p) {
  if (std::strcmp(safe(p.matrix), "") == 0 || std::strcmp(safe(p.referenceMatrix), "") == 0) return false;
  if (std::strcmp(safe(p.mode), "disabled") != 0 && p.tolerance <= 0.0) return false;
  if (std::strcmp(safe(p.mode), "startupSample") == 0
      && std::strcmp(safe(p.startupSampleFailurePolicy), "fatalAtStartup") != 0) return false;
  return true;
}

bool validateProlongationNormalizationPolicy(const ProlongationNormalizationPolicy& p) {
  if (std::strcmp(safe(p.policy), "") == 0) return false;
  if (p.maxConstantModeError <= 0.0) return false;
  if (p.hasBenchmarkConditionEstimate && p.maxBenchmarkConditionEstimate <= 0.0) return false;
  return true;
}

bool validateOperatorParityPolicy(const OperatorParityPolicy& p) {
  if (p.absoluteTolerance < 0.0 || p.relativeTolerance < 0.0) return false;
  if (std::strcmp(safe(p.backend), "") == 0 || std::strcmp(safe(p.precisionMode), "") == 0) return false;
  return std::strcmp(safe(p.ciFailurePolicy), "failCiGate") == 0;
}

bool validatePhaseRollbackPolicy(const PhaseRollbackPolicy& p) {
  return !p.rollbackFields.empty()
      || !p.rollbackDeviceBuffers.empty()
      || !p.rollbackScalars.empty()
      || p.rollbackBoundaryState
      || p.rollbackSolverState;
}

bool validatePhaseTransitionPolicy(const PhaseTransitionPolicy& p) {
  if (std::strcmp(safe(p.sourceNode), "") == 0 || std::strcmp(safe(p.targetNode), "") == 0) return false;
  return !p.freshnessTags.empty() || !p.transferPolicyUpdates.empty();
}

bool validateNonOrthogonalCorrectorPolicy(const NonOrthogonalCorrectorPolicy& p) {
  if (p.nCorr == 0) return false;
  if (p.corrIndex >= p.nCorr) return false;
  if (p.nonOrthIndex > p.nNonOrthCorr) return false;
  return true;
}

bool validateBoundsGuardPolicy(const BoundsGuardPolicy& p) {
  if (std::strcmp(safe(p.fieldName), "") == 0) return false;
  if (p.warnLower > p.warnUpper || p.clipLower > p.clipUpper || p.fatalLower > p.fatalUpper) return false;
  if (p.fatalLower > p.clipLower || p.clipLower > p.warnLower) return false;
  if (p.warnUpper > p.clipUpper || p.clipUpper > p.fatalUpper) return false;
  return true;
}

bool validateRelaxationPolicy(const RelaxationPolicy& p) {
  if (p.applyBeforeSolve && p.equationRelaxVariables.empty()) return false;
  if (p.applyAfterSolve && p.fieldRelaxVariables.empty()) return false;
  return true;
}

const char* firstMismatchedContractHash(const std::vector<ContractHashRecord>& observed, const std::vector<ContractHashRecord>& expected) {
  for (const auto& e : expected) {
    bool matched = false;
    for (const auto& o : observed) {
      if (std::strcmp(safe(o.contractName), safe(e.contractName)) == 0) {
        matched = std::strcmp(safe(o.hashValue), safe(e.hashValue)) == 0;
        break;
      }
    }
    if (!matched) return e.contractName;
  }
  return nullptr;
}

bool backendCompilerMatches(const char* backend, const char* runtimeCompilerKind) {
  return (std::strcmp(safe(backend), "cuda") == 0 && std::strcmp(safe(runtimeCompilerKind), "nvrtc") == 0)
      || (std::strcmp(safe(backend), "hip") == 0 && std::strcmp(safe(runtimeCompilerKind), "hiprtc") == 0)
      || (std::strcmp(safe(backend), "metal") == 0 && std::strcmp(safe(runtimeCompilerKind), "metal") == 0)
      || (std::strcmp(safe(backend), "cpu") == 0 && std::strcmp(safe(runtimeCompilerKind), "none") == 0);
}

bool validateBackendCapability(const BackendCapability& c) {
  return backendCompilerMatches(c.backend, c.runtimeCompilerKind)
      && c.maxThreadsPerBlock > 0
      && c.warpOrWavefrontSize > 0;
}

const std::vector<std::string>& declaredPrograms() {
  static const std::vector<std::string> values = {
  "Solver.CFD.VofPressureAMGPressureSolve",
  "Solver.CFD.VofPressureOrchestrator",
  };
  return values;
}

const std::vector<RankConsistencyPolicy>& rankConsistencyPolicies() {
  static const std::vector<RankConsistencyPolicy> values = {
    {true, true, "CFD_ASSEMBLY_OPERATOR_R4"},
    {true, true, "CFD_ASSEMBLY_OPERATOR_R4"},
  };
  return values;
}

const std::vector<ContractHashRecord>& rankContractHashes() {
  static const std::vector<ContractHashRecord> values = {
    {"pressureHaloRegistry", "vofpressure-r4-halo-v1"},
    {"vofPressureFieldState", "vofpressure-r4-field-v1"},
    {"pressureSolverDecision", "vofpressure-r4-solver-v1"},
    {"backendCapabilities", "vofpressure-r4-backend-v1"},
    {"pressureHaloRegistry", "vofpressure-r4-halo-v1"},
    {"vofPressureFieldState", "vofpressure-r4-field-v1"},
    {"pressureSolverDecision", "vofpressure-r4-solver-v1"},
    {"backendCapabilities", "vofpressure-r4-backend-v1"},
  };
  return values;
}

const std::vector<FieldStateRecord>& fieldStateRecords() {
  static const std::vector<FieldStateRecord> values = {
    {"pressure", "phiHbyA", "gpuRequired", "clean", "clean", "clean", "unallocated", "clean", "clean", "deviceRequired", "allowExplicit", "initial", "", false, false},
    {"pressure", "rAUf", "gpuRequired", "clean", "clean", "clean", "unallocated", "clean", "clean", "deviceRequired", "allowExplicit", "initial", "", false, false},
    {"pressure", "p_rgh", "gpuRequired", "clean", "clean", "clean", "unallocated", "clean", "clean", "mirrorRequired", "allowExplicit", "initial", "", false, false},
    {"pressure", "phiHbyA", "gpuRequired", "clean", "clean", "clean", "unallocated", "clean", "clean", "deviceRequired", "allowExplicit", "initial", "", false, false},
    {"pressure", "rAUf", "gpuRequired", "clean", "clean", "clean", "unallocated", "clean", "clean", "deviceRequired", "allowExplicit", "initial", "", false, false},
    {"pressure", "p_rgh", "gpuRequired", "clean", "clean", "clean", "unallocated", "clean", "clean", "mirrorRequired", "allowExplicit", "initial", "", false, false},
  };
  return values;
}

const std::vector<RunGateDefinition>& runGateDefinitions() {
  static const std::vector<RunGateDefinition> values = {
    {"smoke21", "", 21, {0, 0, 0, 0.500000, 2.000000, 5.000000, 0}},
    {"crossing130", "smoke21", 130, {0, 0, 0, 0.500000, 2.000000, 5.000000, 0}},
    {"soak1000", "crossing130", 1000, {0, 0, 0, 0.500000, 2.000000, 5.000000, 0}},
    {"production15000", "soak1000", 15000, {0, 0, 0, 0.500000, 2.000000, 5.000000, 0}},
  };
  return values;
}

const std::vector<SolverCompatibilityPolicy>& solverCompatibilityPolicies() {
  static const std::vector<SolverCompatibilityPolicy> values = {
    {"pcg", "spd", "spd", "jacobi(1.000000)", false, true},
    {"pcg", "spd", "spd", "jacobi(1.000000)", false, true},
  };
  return values;
}

const std::vector<PreconditionerSymmetryPolicy>& preconditionerSymmetryPolicies() {
  static const std::vector<PreconditionerSymmetryPolicy> values = {
    {"navatalaGAMG", "pressureLduMatrix", "pcg", "spd", 0, 0.000010, 1.0e-12, true, "disabled", "emitStructuredDiagnostic"},
  };
  return values;
}

const std::vector<PreconditionerEffectivenessPolicy>& preconditionerEffectivenessPolicies() {
  static const std::vector<PreconditionerEffectivenessPolicy> values = {
    {"navatalaGAMG", "DIC", "pcg", 0, 1.000000, 1.000000, 0.000000, false, false, false, "disabled", "emitStructuredDiagnostic"},
  };
  return values;
}

const std::vector<CoarseGalerkinParityPolicy>& coarseGalerkinParityPolicies() {
  static const std::vector<CoarseGalerkinParityPolicy> values = {
    {"navatalaGAMG", "pressureLduMatrix", 1, 0.000010, 0.000010, 1.0e-7, 64, true, "startupSample", "rollbackAndFallback"},
  };
  return values;
}

const std::vector<OperatorNullspaceInvariantPolicy>& operatorNullspaceInvariantPolicies() {
  static const std::vector<OperatorNullspaceInvariantPolicy> values = {
    {"navatalaGAMG", "pressureLduMatrix", 0.000001, 1.0e-8, 0, true, "startupSample", "rollbackAndFallback"},
  };
  return values;
}

const std::vector<SignTransformVerificationPolicy>& signTransformVerificationPolicies() {
  static const std::vector<SignTransformVerificationPolicy> values = {
    {"pressureDeviceMatrix", "pressureHostReferenceMatrix", 0.000001, "startupSample", "warnAndContinue", "failCiGate", "fatalAtStartup"},
  };
  return values;
}

const std::vector<ProlongationNormalizationPolicy>& prolongationNormalizationPolicies() {
  static const std::vector<ProlongationNormalizationPolicy> values = {
    {"diagonalWeightedMeanOne", 0.000010, true, 1000000.000000},
  };
  return values;
}

const std::vector<OperatorParityPolicy>& operatorParityPolicies() {
  static const std::vector<OperatorParityPolicy> values = {
    {"mixed", "cuda", 0.000001, 0.000010, "onConvergenceFailure", "emitStructuredDiagnostic", "failCiGate"},
  };
  return values;
}

const std::vector<PhaseRollbackPolicy>& phaseRollbackPolicies() {
  static const std::vector<PhaseRollbackPolicy> values = {
    {{"p_rgh"}, {"devP", "devSource"}, {}, true, true, "afterMatrixSolve", "afterMatrixSolve", "afterMatrixSolve"},
  };
  return values;
}

const std::vector<PhaseTransitionPolicy>& phaseTransitionPolicies() {
  static const std::vector<PhaseTransitionPolicy> values = {
    {"pressure", "postPressure", {"p_rgh", "phi"}, {"allowExplicit"}},
  };
  return values;
}

const std::vector<NonOrthogonalCorrectorPolicy>& nonOrthogonalCorrectorPolicies() {
  static const std::vector<NonOrthogonalCorrectorPolicy> values = {
    {1, 0, 0, 0, true, true, "updateEveryCorrection", "updateFinalOnly", "updateFinalOnly", "updateEveryCorrection", "updateEveryCorrection"},
  };
  return values;
}

const std::vector<BoundsGuardPolicy>& boundsGuardPolicies() {
  static const std::vector<BoundsGuardPolicy> values = {
    {"alpha1", 0.000000, 1.000000, -0.000100, 1.000100, -0.500000, 1.500000, true, 0, 0.000000},
  };
  return values;
}

const std::vector<RelaxationPolicy>& relaxationPolicies() {
  static const std::vector<RelaxationPolicy> values = {
    {{}, {"pEqn"}, true, false},
  };
  return values;
}

const std::vector<MMatrixRepairStatsRecord>& mMatrixRepairStatsRecords() {
  static const std::vector<MMatrixRepairStatsRecord> values = {
  };
  return values;
}

const std::vector<BackendCapability>& backendCapabilities() {
  static const std::vector<BackendCapability> values = {
    {"cuda", "nvrtc", false, true, false, true, false, true, true, 256, 32, "hostStaged"},
    {"hip", "hiprtc", false, true, false, true, false, true, true, 256, 64, "hostStaged"},
    {"metal", "metal", false, true, false, false, false, true, false, 256, 32, "hostStaged"},
    {"cuda", "nvrtc", false, true, false, true, false, true, true, 256, 32, "hostStaged"},
    {"hip", "hiprtc", false, true, false, true, false, true, true, 256, 64, "hostStaged"},
    {"metal", "metal", false, true, false, false, false, true, false, 256, 32, "hostStaged"},
  };
  return values;
}

const std::vector<HaloDescriptor>& haloDescriptors() {
  static const std::vector<HaloDescriptor> values = {
    {1, "pressureProcessorPatchHalo", "GpuSnGrad", "pressure", "patchPair", "Float32", 1, 700000},
    {1, "pressureProcessorPatchHalo", "GpuSnGrad", "pressure", "patchPair", "Float32", 1, 700000},
  };
  return values;
}

const std::vector<WorkspacePlan>& workspacePlans() {
  static const std::vector<WorkspacePlan> values = {
    {0, "cuda", 0, 0, 0, "mixed", 1, 1},
    {0, "cuda", 0, 0, 0, "mixed", 1, 1},
  };
  return values;
}

} } // namespace NavatalaRegistry::R4Safety
