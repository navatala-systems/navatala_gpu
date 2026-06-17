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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace NavatalaRegistry { namespace R4Safety {

struct PhaseTimingRecord {
  const char* phaseName;
  std::uint32_t rank;
  const char* backend;
  double hostMs;
  double deviceMs;
  double mpiWaitMs;
  std::uint64_t h2dBytes;
  std::uint64_t d2hBytes;
  std::uint64_t kernelLaunchCount;
  std::uint64_t allocatedBytes;
  std::uint64_t peakDeviceBytes;
  bool fallbackUsed;
};

struct HaloTraceRecord {
  std::uint32_t rank;
  std::uint32_t neigh;
  const char* exchange;
  const char* module;
  const char* phase;
  std::uint64_t timeIndex;
  const char* outerCorr;
  const char* pressureCorr;
  const char* nonOrthCorr;
  const char* alphaSubCycle;
  const char* turbulenceCorr;
  std::uint32_t tag;
  std::uint32_t arity;
  const char* type;
  std::uint64_t sendCount;
  std::uint64_t recvCount;
  std::uint64_t sendBytes;
  std::uint64_t recvBytes;
  std::uint32_t patchLocal;
  std::uint32_t patchRemote;
  const char* comm;
};

struct RunGateMetrics {
  std::uint64_t fallbackCount;
  std::uint64_t mpiErrorCount;
  std::uint64_t nonFiniteCount;
  double dtCollapseRatio;
  double pressureIterationTrend;
  double memoryGrowthPercent;
  std::uint64_t rankDivergence;
};

struct MMatrixRepairStatsRecord {
  std::uint64_t level;
  std::uint64_t positiveOffDiagCount;
  double positiveOffDiagSum;
  double rowSumDriftBefore;
  double rowSumDriftAfter;
  double diagMinBefore;
  double diagMinAfter;
  bool repairApplied;
};

struct SolverCompatibilityPolicy {
  const char* solverKind;
  const char* operatorClass;
  const char* preconditionerClass;
  const char* smoother;
  bool allowVariablePreconditioner;
  bool recheckOnOperatorClassChange;
};

struct PreconditionerSymmetryPolicy {
  const char* preconditionerName;
  const char* operatorName;
  const char* solverKind;
  const char* preconditionerClass;
  std::uint64_t sampleCount;
  double maxRelativeAsymmetry;
  double maxAbsoluteAsymmetry;
  bool requireRankConsistentSamples;
  const char* mode;
  const char* failurePolicy;
};

struct PreconditionerEffectivenessPolicy {
  const char* preconditionerName;
  const char* baselinePreconditionerName;
  const char* solverKind;
  std::uint64_t sampleCount;
  double maxPcgResidualRatio;
  double maxResidualRatioVsBaseline;
  double maxGuardRejectFraction;
  bool requireFallbackOnReject;
  bool requireDisableOnReject;
  bool safeBlendDefaultAllowed;
  const char* mode;
  const char* failurePolicy;
};

struct CoarseGalerkinParityPolicy {
  const char* hierarchyName;
  const char* operatorName;
  std::uint64_t sampleCount;
  double maxRelativeL2;
  double maxRelativeL1;
  double maxAbsoluteDiff;
  std::uint64_t maxLargeRows;
  bool includeProcessorInterfaces;
  const char* mode;
  const char* failurePolicy;
};

struct OperatorNullspaceInvariantPolicy {
  const char* hierarchyName;
  const char* operatorName;
  double maxRelativeRowSum;
  double maxAbsoluteRowSum;
  std::uint64_t maxLargeRows;
  bool includeProcessorInterfaces;
  const char* mode;
  const char* failurePolicy;
};

struct SignTransformVerificationPolicy {
  const char* matrix;
  const char* referenceMatrix;
  double tolerance;
  const char* mode;
  const char* runtimeDebugFailurePolicy;
  const char* ciFailurePolicy;
  const char* startupSampleFailurePolicy;
};

struct ProlongationNormalizationPolicy {
  const char* policy;
  double maxConstantModeError;
  bool hasBenchmarkConditionEstimate;
  double maxBenchmarkConditionEstimate;
};

struct OperatorParityPolicy {
  const char* precisionMode;
  const char* backend;
  double absoluteTolerance;
  double relativeTolerance;
  const char* samplingPolicy;
  const char* runtimeFailurePolicy;
  const char* ciFailurePolicy;
};

struct PhaseRollbackPolicy {
  std::vector<std::string> rollbackFields;
  std::vector<std::string> rollbackDeviceBuffers;
  std::vector<std::string> rollbackScalars;
  bool rollbackBoundaryState;
  bool rollbackSolverState;
  const char* snapshotPoint;
  const char* restorePoint;
  const char* commitPoint;
};

struct PhaseTransitionPolicy {
  const char* sourceNode;
  const char* targetNode;
  std::vector<std::string> freshnessTags;
  std::vector<std::string> transferPolicyUpdates;
};

struct NonOrthogonalCorrectorPolicy {
  std::uint64_t nCorr;
  std::uint64_t nNonOrthCorr;
  std::uint64_t corrIndex;
  std::uint64_t nonOrthIndex;
  bool finalPressureCorrection;
  bool finalNonOrthogonalCorrection;
  const char* pUpdatePolicy;
  const char* phiUpdatePolicy;
  const char* UUpdatePolicy;
  const char* rhsRefreshPolicy;
  const char* operatorRefreshPolicy;
};

struct BoundsGuardPolicy {
  const char* fieldName;
  double warnLower;
  double warnUpper;
  double clipLower;
  double clipUpper;
  double fatalLower;
  double fatalUpper;
  bool clampAllowed;
  std::uint64_t maxClippedCells;
  double maxClippedFraction;
};

struct RelaxationPolicy {
  std::vector<std::string> fieldRelaxVariables;
  std::vector<std::string> equationRelaxVariables;
  bool applyBeforeSolve;
  bool applyAfterSolve;
};

struct RunGateDefinition {
  const char* kind;
  const char* requiredPreviousGate;
  std::uint64_t nSteps;
  RunGateMetrics thresholds;
};

struct RunGateRecord {
  const char* kind;
  const char* gitCommit;
  const char* caseId;
  bool passed;
};

struct ContractHashRecord {
  const char* contractName;
  const char* hashValue;
};

struct RankConsistencyPolicy {
  bool enforceIdenticalPhases;
  bool enforceIdenticalFallback;
  const char* contractVersionString;
};

struct FieldStateRecord {
  const char* phaseName;
  const char* fieldName;
  const char* phaseRequirement;
  const char* hostInterior;
  const char* deviceInterior;
  const char* hostBoundary;
  const char* deviceBoundary;
  const char* processorHalo;
  const char* oldTime;
  const char* residencyPolicy;
  const char* transferPolicy;
  const char* lastProducerPhase;
  const char* lastConsumerPhase;
  bool requiresHaloFresh;
  bool requiresBoundaryFresh;
};

struct BackendCapability {
  const char* backend;
  const char* runtimeCompilerKind;
  bool supportsGpuAwareMpi;
  bool supportsHostPinnedMapping;
  bool supportsZeroCopy;
  bool supportsFp64;
  bool supportsFp64Atomics;
  bool supportsRuntimeCompilation;
  bool supportsDeviceLinking;
  std::uint32_t maxThreadsPerBlock;
  std::uint32_t warpOrWavefrontSize;
  const char* preferredHaloPath;
};

struct HaloDescriptor {
  std::uint32_t exchangeId;
  const char* exchangeName;
  const char* moduleName;
  const char* phaseName;
  const char* communicatorKind;
  const char* payloadType;
  std::uint32_t payloadArity;
  std::uint32_t tag;
};

struct WorkspacePlan {
  std::uint32_t rank;
  const char* backend;
  std::uint64_t rankLocalCells;
  std::uint64_t rankLocalFaces;
  std::uint64_t processorPatchFaces;
  const char* precisionMode;
  std::uint64_t peakBytesPlanned;
  std::uint64_t headroomBytes;
};

std::string formatPhaseTimingJson(const PhaseTimingRecord& r);
std::string formatHaloExchangeTrace(const HaloTraceRecord& r);
std::string formatFieldStateTrace(const FieldStateRecord& r, const char* transfer, std::uint64_t bytes);
std::string formatMMatrixStats(const MMatrixRepairStatsRecord& r);
std::string formatSolverDecisionTrace(const SolverCompatibilityPolicy& p, bool accepted, const char* reason);
const char* firstExceededRunGateThreshold(const RunGateMetrics& observed, const RunGateMetrics& thresholds);
bool hasPassingRunGatePrerequisite(const RunGateDefinition& gate, const std::vector<RunGateRecord>& records, const char* gitCommit, const char* caseId);
bool validateFieldStateRecord(const FieldStateRecord& r);
bool validateMMatrixRepairStats(const MMatrixRepairStatsRecord& r, double maxRowDrift);
bool solverCompatibilityAccepted(const SolverCompatibilityPolicy& p);
bool validateSignTransformVerificationPolicy(const SignTransformVerificationPolicy& p);
bool validateProlongationNormalizationPolicy(const ProlongationNormalizationPolicy& p);
bool validateOperatorParityPolicy(const OperatorParityPolicy& p);
bool validatePhaseRollbackPolicy(const PhaseRollbackPolicy& p);
bool validatePhaseTransitionPolicy(const PhaseTransitionPolicy& p);
bool validateNonOrthogonalCorrectorPolicy(const NonOrthogonalCorrectorPolicy& p);
bool validateBoundsGuardPolicy(const BoundsGuardPolicy& p);
bool validateRelaxationPolicy(const RelaxationPolicy& p);
const char* firstMismatchedContractHash(const std::vector<ContractHashRecord>& observed, const std::vector<ContractHashRecord>& expected);
bool backendCompilerMatches(const char* backend, const char* runtimeCompilerKind);
bool validateBackendCapability(const BackendCapability& c);

const std::vector<std::string>& declaredPrograms();
const std::vector<RankConsistencyPolicy>& rankConsistencyPolicies();
const std::vector<ContractHashRecord>& rankContractHashes();
const std::vector<FieldStateRecord>& fieldStateRecords();
const std::vector<RunGateDefinition>& runGateDefinitions();
const std::vector<SolverCompatibilityPolicy>& solverCompatibilityPolicies();
const std::vector<PreconditionerSymmetryPolicy>& preconditionerSymmetryPolicies();
const std::vector<PreconditionerEffectivenessPolicy>& preconditionerEffectivenessPolicies();
const std::vector<CoarseGalerkinParityPolicy>& coarseGalerkinParityPolicies();
const std::vector<OperatorNullspaceInvariantPolicy>& operatorNullspaceInvariantPolicies();
const std::vector<SignTransformVerificationPolicy>& signTransformVerificationPolicies();
const std::vector<ProlongationNormalizationPolicy>& prolongationNormalizationPolicies();
const std::vector<OperatorParityPolicy>& operatorParityPolicies();
const std::vector<PhaseRollbackPolicy>& phaseRollbackPolicies();
const std::vector<PhaseTransitionPolicy>& phaseTransitionPolicies();
const std::vector<NonOrthogonalCorrectorPolicy>& nonOrthogonalCorrectorPolicies();
const std::vector<BoundsGuardPolicy>& boundsGuardPolicies();
const std::vector<RelaxationPolicy>& relaxationPolicies();
const std::vector<MMatrixRepairStatsRecord>& mMatrixRepairStatsRecords();
const std::vector<BackendCapability>& backendCapabilities();
const std::vector<HaloDescriptor>& haloDescriptors();
const std::vector<WorkspacePlan>& workspacePlans();

} } // namespace NavatalaRegistry::R4Safety
