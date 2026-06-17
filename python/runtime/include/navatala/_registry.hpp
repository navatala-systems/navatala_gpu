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
#include <gpu_runtime.h>
#include <program_cache.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace NavatalaRegistry {

enum class FieldLayoutKind { AoS, PackedSoA };
struct FieldLayoutInfo { FieldLayoutKind layout; std::uint32_t components; GpuRuntime::DataType elemType; };
bool tryGetFieldLayout(const std::string& layoutName, FieldLayoutInfo& out);

using NamedBuffer = std::pair<std::string, GpuRuntime::Buffer*>;
bool tryGetKernelSource(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out);

enum class KernelArgumentRole { Input, Output, InputOutput, Params, Count, Scratch };
enum class KernelAccessMode { ReadOnly, ReadWrite, WriteOnly };
struct KernelArgumentInfo {
  const char* semanticName;
  std::uint32_t ordinal;
  KernelArgumentRole role;
  KernelAccessMode access;
  GpuRuntime::MemoryKind memoryKind;
  bool required;
  std::size_t minimumByteSize;
  std::size_t maximumByteSize;
  std::size_t alignmentBytes;
  const char* paramsLayoutHash;
  std::size_t scalarCount;
  std::size_t scalarByteSize;
};
struct KernelAbiManifestInfo {
  std::uint32_t abiVersion;
  const char* kernelName;
  const char* backend;
  const char* backendEntrypoint;
  const char* artifactId;
  const char* sourceSignatureHash;
  const char* wrapperExpectedHash;
  std::size_t argumentCount;
  const KernelArgumentInfo* arguments;
};
bool tryGetKernelAbiManifest(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out);
struct KernelBoundArgument {
  std::string semanticName;
  GpuRuntime::Buffer* buffer = nullptr;
  const char* paramsLayoutHash = nullptr;
  std::size_t scalarCount = 0;
  std::size_t scalarByteSize = 0;
};
class FinalizedKernelBinding {
public:
  const std::vector<GpuRuntime::Buffer*>& args() const { return args_; }
private:
  friend class KernelBindingBuilder;
  std::vector<GpuRuntime::Buffer*> args_;
};
class KernelBindingBuilder {
public:
  KernelBindingBuilder() = default;
  explicit KernelBindingBuilder(const KernelAbiManifestInfo& manifest);
  bool valid() const { return manifest_ != nullptr; }
  KernelBindingBuilder& bind(const std::string& semanticName, GpuRuntime::Buffer* value);
  KernelBindingBuilder& bindParams(const std::string& semanticName, GpuRuntime::Buffer* value, const char* paramsLayoutHash);
  KernelBindingBuilder& bindCount(const std::string& semanticName, GpuRuntime::Buffer* value, std::size_t scalarCount, std::size_t scalarByteSize);
  bool finalize(FinalizedKernelBinding& out, std::string* error = nullptr) const;
private:
  const KernelAbiManifestInfo* manifest_ = nullptr;
  std::vector<KernelBoundArgument> bindings_;
};
namespace KernelBindings {
bool make(const std::string& backend, const std::string& kernelName, KernelBindingBuilder& out);
} // namespace KernelBindings

bool tryRunProgram(const std::string& backend, const std::string& programName, GpuRuntime::Device& device, GpuRuntime::Queue& queue, const std::vector<NamedBuffer>& buffers);

} // namespace NavatalaRegistry
