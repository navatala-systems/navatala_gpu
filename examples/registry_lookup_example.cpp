// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Worked example: query the generated host-side kernel registry without
// creating a GPU context. This is a lightweight smoke test for kernel source
// and ABI-manifest discoverability.

#include "navatala/_registry.hpp"

#include <cstdio>
#include <string>

namespace NavatalaRegistry {
bool tryGetKernelSource_cuda_samples(
    const std::string& backend,
    const std::string& kernelName,
    GpuRuntime::ProgramSource& out);
bool tryGetKernelAbiManifest_cuda_samples(
    const std::string& backend,
    const std::string& kernelName,
    const KernelAbiManifestInfo*& out);
}  // namespace NavatalaRegistry

namespace {

const char* kind_name(GpuRuntime::ProgramSource::Kind kind) {
    switch (kind) {
        case GpuRuntime::ProgramSource::Kind::CudaCpp: return "CudaCpp";
        case GpuRuntime::ProgramSource::Kind::HipCpp: return "HipCpp";
        case GpuRuntime::ProgramSource::Kind::OpenClC: return "OpenClC";
        case GpuRuntime::ProgramSource::Kind::OpenClBinary: return "OpenClBinary";
        case GpuRuntime::ProgramSource::Kind::Glsl: return "Glsl";
        case GpuRuntime::ProgramSource::Kind::Spirv: return "Spirv";
        case GpuRuntime::ProgramSource::Kind::Msl: return "Msl";
        case GpuRuntime::ProgramSource::Kind::Ptx: return "Ptx";
        case GpuRuntime::ProgramSource::Kind::Cubin: return "Cubin";
        case GpuRuntime::ProgramSource::Kind::Hsaco: return "Hsaco";
        case GpuRuntime::ProgramSource::Kind::Metallib: return "Metallib";
    }
    return "Unknown";
}

}  // namespace

int main() {
    const std::string backend = "cuda";
    const std::string kernel_name = "navatala_samples_float32_add";

    GpuRuntime::ProgramSource source;
    if (!NavatalaRegistry::tryGetKernelSource_cuda_samples(backend, kernel_name, source)) {
        std::fprintf(stderr,
                     "FAIL: kernel source not found for backend=%s kernel=%s\n",
                     backend.c_str(),
                     kernel_name.c_str());
        return 1;
    }

    const NavatalaRegistry::KernelAbiManifestInfo* manifest = nullptr;
    if (!NavatalaRegistry::tryGetKernelAbiManifest_cuda_samples(backend, kernel_name, manifest) ||
        manifest == nullptr) {
        std::fprintf(stderr,
                     "FAIL: ABI manifest not found for backend=%s kernel=%s\n",
                     backend.c_str(),
                     kernel_name.c_str());
        return 1;
    }

    if (source.entryPoint != kernel_name ||
        std::string(manifest->backendEntrypoint) != kernel_name ||
        manifest->argumentCount != 3 ||
        source.bytes.empty()) {
        std::fprintf(stderr,
                     "FAIL: registry returned inconsistent metadata for %s\n",
                     kernel_name.c_str());
        return 1;
    }

    std::printf("PASS: registry partition found %s/%s (%s, %zu bytes, %zu args)\n",
                backend.c_str(),
                kernel_name.c_str(),
                kind_name(source.kind),
                source.bytes.size(),
                manifest->argumentCount);
    return 0;
}
