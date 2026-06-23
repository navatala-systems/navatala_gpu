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

// GPU Runtime - Vulkan Backend Implementation
// Full Vulkan backend with Device, Queue, Buffer, Program, Event

#include <gpu_runtime.h>
#include <vulkan/vulkan.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace GpuRuntime {

// ---------------------------------------------------------------------------
// Minimal GLSL → SPIR-V compilation via glslc (Vulkan SDK / shaderc).
// ---------------------------------------------------------------------------
namespace {

static std::vector<std::uint8_t> compileGlslToSpirv(const std::string& glslText) {
    namespace fs = std::filesystem;

    // Create a unique temporary directory.
    const fs::path tmpDir = fs::temp_directory_path() / ("gpu_rt_glslc_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(tmpDir);

    const fs::path glslPath = tmpDir / "kernel.comp";
    const fs::path spvPath  = tmpDir / "kernel.spv";
    const fs::path logPath  = tmpDir / "glslc.log";

    // Write GLSL source.
    {
        std::ofstream ofs(glslPath, std::ios::binary);
        ofs.write(glslText.data(), static_cast<std::streamsize>(glslText.size()));
    }

    // Invoke glslc.
    const std::string cmd = "glslc -O \"" + glslPath.string() + "\" -o \"" +
                            spvPath.string() + "\" 2>\"" + logPath.string() + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::string err;
        { std::ifstream ifs(logPath); std::ostringstream ss; ss << ifs.rdbuf(); err = ss.str(); }
        fs::remove_all(tmpDir);
        throw std::runtime_error(
            "Failed to compile GLSL to SPIR-V via glslc (exit=" + std::to_string(rc) + "):\n" + err);
    }

    // Read SPIR-V bytes.
    std::ifstream ifs(spvPath, std::ios::binary | std::ios::ate);
    const auto sz = ifs.tellg();
    ifs.seekg(0);
    std::vector<std::uint8_t> spv(static_cast<std::size_t>(sz));
    ifs.read(reinterpret_cast<char*>(spv.data()), sz);

    fs::remove_all(tmpDir);
    return spv;
}

}  // anonymous namespace

namespace {

// Tracks live VkDevice handles so that objects (e.g. cached shader modules) can avoid
// calling Vulkan destroy functions after the device has already been destroyed.
// This situation can happen when higher-level code caches `Program` objects beyond the
// lifetime of a `Device` instance (e.g. process-global caches).
std::mutex g_liveDeviceMutex;
std::unordered_set<VkDevice> g_liveDevices;

static void registerLiveDevice(VkDevice d) {
    std::lock_guard<std::mutex> lock(g_liveDeviceMutex);
    g_liveDevices.insert(d);
}

static void unregisterLiveDevice(VkDevice d) {
    std::lock_guard<std::mutex> lock(g_liveDeviceMutex);
    g_liveDevices.erase(d);
}

static bool isLiveDevice(VkDevice d) {
    std::lock_guard<std::mutex> lock(g_liveDeviceMutex);
    return g_liveDevices.find(d) != g_liveDevices.end();
}

inline void* toNativeHandle(VkShaderModule h) {
#if defined(VK_USE_64_BIT_PTR_DEFINES) && (VK_USE_64_BIT_PTR_DEFINES == 1)
    return reinterpret_cast<void*>(h);
#else
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(h));
#endif
}

inline VkShaderModule fromNativeHandle(void* p) {
#if defined(VK_USE_64_BIT_PTR_DEFINES) && (VK_USE_64_BIT_PTR_DEFINES == 1)
    return reinterpret_cast<VkShaderModule>(p);
#else
    return static_cast<VkShaderModule>(reinterpret_cast<std::uintptr_t>(p));
#endif
}

}  // namespace

// ============================================================================
// VulkanEvent - Native Vulkan event implementation
// ============================================================================

class VulkanEvent : public Event {
public:
    VulkanEvent(VkDevice device, VkEvent event) : device_(device), event_(event) {}

    ~VulkanEvent() override {
        if (event_ && isLiveDevice(device_)) {
            vkDestroyEvent(device_, event_, nullptr);
        }
    }

    bool isComplete() const override {
        VkResult result = vkGetEventStatus(device_, event_);
        return result == VK_EVENT_SET;
    }

    void wait() override {
        // Vulkan doesn't have vkWaitForEvents - poll with vkGetEventStatus
        auto start = std::chrono::steady_clock::now();
        uint64_t timeout_ns = 5 * 1000000000ULL;  // 5 second default timeout

        while (true) {
            VkResult result = vkGetEventStatus(device_, event_);
            if (result == VK_EVENT_SET) return;
            if (result != VK_EVENT_RESET) {
                throw std::runtime_error("Failed to wait for Vulkan event");
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start);
            if (elapsed.count() >= timeout_ns) {
                throw std::runtime_error("Timeout waiting for Vulkan event");
            }

            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void* nativeHandle() override {
        return reinterpret_cast<void*>(event_);
    }

    VkEvent get() const { return event_; }

private:
    VkDevice device_;
    VkEvent event_;
};

// ============================================================================
// VulkanDescriptorPool - Manages descriptor set allocations ()
// ============================================================================

class VulkanDescriptorPool {
public:
    VulkanDescriptorPool(VkDevice device, size_t maxSets = 100)
        : device_(device), maxSets_(maxSets), pool_(VK_NULL_HANDLE) {

        // Pool sizes for storage buffers (most common for compute)
        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(maxSets * 16) },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(maxSets * 4) }
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(maxSets);
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan descriptor pool");
        }
    }

    ~VulkanDescriptorPool() {
        if (pool_ && isLiveDevice(device_)) {
            vkDestroyDescriptorPool(device_, pool_, nullptr);
        }
    }

    VkDescriptorSet allocateDescriptorSet(VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &layout;

        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate Vulkan descriptor set");
        }
        return descriptorSet;
    }

    void freeDescriptorSet(VkDescriptorSet descriptorSet, VkDescriptorSetLayout layout) {
        vkFreeDescriptorSets(device_, pool_, 1, &descriptorSet);
        (void)layout;  // Layout not needed for freeing
    }

    void reset() {
        vkResetDescriptorPool(device_, pool_, 0);
    }

    VkDescriptorPool get() const { return pool_; }

private:
    VkDevice device_;
    size_t maxSets_;
    VkDescriptorPool pool_;
};

// ============================================================================
// VulkanDescriptorSetLayout - Defines descriptor set layout ()
// ============================================================================

class VulkanDescriptorSetLayout {
public:
    VulkanDescriptorSetLayout(VkDevice device,
                              const std::vector<VkDescriptorSetLayoutBinding>& bindings)
        : device_(device), layout_(VK_NULL_HANDLE) {

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &layout_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan descriptor set layout");
        }
    }

    ~VulkanDescriptorSetLayout() {
        if (layout_ && isLiveDevice(device_)) {
            vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        }
    }

    VkDescriptorSetLayout get() const { return layout_; }

private:
    VkDevice device_;
    VkDescriptorSetLayout layout_;
};

// ============================================================================
// VulkanFence - Fence-based synchronization for Vulkan ()
// ============================================================================

class VulkanFence : public Event {
public:
    VulkanFence(VkDevice device) : device_(device), fence_(VK_NULL_HANDLE), signaled_(false) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = 0;  // Start unsignaled

        if (vkCreateFence(device_, &fenceInfo, nullptr, &fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan fence");
        }
    }

    ~VulkanFence() override {
        if (fence_ && isLiveDevice(device_)) {
            vkDestroyFence(device_, fence_, nullptr);
        }
    }

    bool isComplete() const override {
        if (signaled_) return true;
        VkResult result = vkGetFenceStatus(device_, fence_);
        if (result == VK_SUCCESS) {
            signaled_ = true;
            return true;
        }
        return false;
    }

    void wait() override {
        if (signaled_) return;

        VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to wait for Vulkan fence");
        }
        signaled_ = true;
    }

    void* nativeHandle() override {
        return &fence_;
    }

    VkFence get() const { return fence_; }

    // Reset fence to unsignaled state
    void reset() {
        if (vkResetFences(device_, 1, &fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to reset Vulkan fence");
        }
        signaled_ = false;
    }

private:
    VkDevice device_;
    VkFence fence_;
    mutable bool signaled_;
};

// ============================================================================
// VulkanQueue - Vulkan queue implementation
// ============================================================================

class VulkanQueue : public Queue {
public:
    VulkanQueue(VkDevice device, VkQueue queue, uint32_t queueFamilyIndex)
        : device_(device), queue_(queue), queueFamilyIndex_(queueFamilyIndex),
          commandPool_(VK_NULL_HANDLE), descriptorPool_(nullptr) {
        // Create command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan command pool");
        }

        // Create descriptor pool for kernel arguments
        descriptorPool_ = std::make_unique<VulkanDescriptorPool>(device_, 100);
    }

    ~VulkanQueue() override {
        // Ensure GPU has finished using any in-flight resources we may clean up on sync.
        try { synchronize(); } catch (...) {}
        descriptorPool_.reset();  // Destroy descriptor pool first
        if (commandPool_) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
    }

    void submit(Program& program, const std::vector<Buffer*>& args,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z) override {
        (void)block_x;
        (void)block_y;
        (void)block_z;

        // Vulkan `Program::nativeHandle()` is a VkShaderModule for now.
        // We create a compute pipeline per-dispatch with a matching descriptor set layout.
        VkShaderModule shaderModule = fromNativeHandle(program.nativeHandle());
        if (!shaderModule) {
            throw std::runtime_error("Invalid Vulkan program (missing shader module)");
        }

        // Create descriptor set layout and descriptor set for arguments
        if (!args.empty()) {
            // Create descriptor set layout bindings for each argument
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            for (size_t i = 0; i < args.size(); ++i) {
                VkDescriptorSetLayoutBinding binding{};
                binding.binding = static_cast<uint32_t>(i);
                binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                binding.descriptorCount = 1;
                binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                binding.pImmutableSamplers = nullptr;
                bindings.push_back(binding);
            }

            // Create descriptor set layout with extended lifetime
            auto layout = std::make_unique<VulkanDescriptorSetLayout>(device_, bindings);
            VkDescriptorSetLayout layoutHandle = layout->get();

            // Allocate descriptor set
            VkDescriptorSet descriptorSet = descriptorPool_->allocateDescriptorSet(layoutHandle);

            // Create a simple pipeline layout for this descriptor set layout
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &layoutHandle;

            VkPipelineLayout pipelineLayout;
            if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Vulkan pipeline layout");
            }

            // Create compute pipeline for this dispatch.
            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shaderModule;
            // NOTE: SPIR-V compute shaders use "main" as the entry point.
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = pipelineLayout;

            VkPipeline pipeline = VK_NULL_HANDLE;
            VkResult pipelineResult = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
            if (pipelineResult != VK_SUCCESS) {
                vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
                throw std::runtime_error(
                    "Failed to create Vulkan compute pipeline (VkResult=" +
                    std::to_string(static_cast<int>(pipelineResult)) +
                    ", buffers=" + std::to_string(args.size()) +
                    ", entry=\"" + std::string(stageInfo.pName) + "\")");
            }

            // Update descriptor set with buffer bindings.
            //
            // IMPORTANT: `VkWriteDescriptorSet` stores pointers to `VkDescriptorBufferInfo`,
            // so those structs must outlive the `vkUpdateDescriptorSets` call.
            std::vector<VkDescriptorBufferInfo> bufferInfos(args.size());
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(args.size());
            for (size_t i = 0; i < args.size(); ++i) {
                bufferInfos[i].buffer = reinterpret_cast<VkBuffer>(args[i]->nativeHandle());
                bufferInfos[i].offset = 0;
                bufferInfos[i].range = static_cast<VkDeviceSize>(args[i]->sizeBytes());

                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = descriptorSet;
                write.dstBinding = static_cast<uint32_t>(i);
                write.dstArrayElement = 0;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.descriptorCount = 1;
                write.pBufferInfo = &bufferInfos[i];
                writes.push_back(write);
            }

            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            // Create command buffer
            VkCommandBuffer cmdBuf = allocateCommandBuffer();

            // Begin command buffer
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmdBuf, &beginInfo);

            // Make host writes to mapped buffers visible to the compute shader.
            // (Especially important for host-visible buffers used as "Managed" memory.)
            VkMemoryBarrier hostToShader{};
            hostToShader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            hostToShader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            hostToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(
                cmdBuf,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &hostToShader,
                0, nullptr,
                0, nullptr
            );

            // Bind pipeline
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

            // Bind descriptor set
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipelineLayout,  // Use proper pipeline layout
                                   0, 1, &descriptorSet, 0, nullptr);

            // Dispatch
            vkCmdDispatch(cmdBuf, grid_x, grid_y, grid_z);

            // Make shader writes visible to the host after queue completion.
            VkMemoryBarrier shaderToHost{};
            shaderToHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            shaderToHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            shaderToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(
                cmdBuf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                1, &shaderToHost,
                0, nullptr,
                0, nullptr
            );

            // End command buffer
            vkEndCommandBuffer(cmdBuf);

            // Submit to queue
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuf;

            if (vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                vkDestroyPipeline(device_, pipeline, nullptr);
                vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
                throw std::runtime_error("Failed to submit Vulkan commands");
            }

            // Defer cleanup until `synchronize()` (which waits for queue idle).
            inflightCmdBuffers_.push_back(cmdBuf);
            inflightPipelines_.push_back(pipeline);
            inflightPipelineLayouts_.push_back(pipelineLayout);
            inflightSetLayouts_.push_back(std::move(layout));

            // Current runtime semantics require that sequential `submit()` calls behave like
            // an in-order compute stream with implicit memory dependencies (like CUDA/OpenCL).
            // Until we implement proper semaphore-based synchronization, we conservatively
            // wait for completion here to guarantee visibility between submissions.
            if (vkQueueWaitIdle(queue_) != VK_SUCCESS) {
                throw std::runtime_error("Failed to synchronize Vulkan queue after submit");
            }
            cleanupInflight();
        } else {
            // No arguments - simple dispatch
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = nullptr;

            VkPipelineLayout pipelineLayout;
            if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Vulkan pipeline layout");
            }

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = pipelineLayout;

            VkPipeline pipeline = VK_NULL_HANDLE;
            VkResult pipelineResult2 = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
            if (pipelineResult2 != VK_SUCCESS) {
                vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
                throw std::runtime_error(
                    "Failed to create Vulkan compute pipeline (VkResult=" +
                    std::to_string(static_cast<int>(pipelineResult2)) +
                    ", buffers=" + std::to_string(args.size()) +
                    ", entry=\"" + std::string(stageInfo.pName) + "\")");
            }

            VkCommandBuffer cmdBuf = allocateCommandBuffer();

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmdBuf, &beginInfo);

            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdDispatch(cmdBuf, grid_x, grid_y, grid_z);

            vkEndCommandBuffer(cmdBuf);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuf;

            if (vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                vkDestroyPipeline(device_, pipeline, nullptr);
                vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
                throw std::runtime_error("Failed to submit Vulkan commands");
            }

            inflightCmdBuffers_.push_back(cmdBuf);
            inflightPipelines_.push_back(pipeline);
            inflightPipelineLayouts_.push_back(pipelineLayout);

            if (vkQueueWaitIdle(queue_) != VK_SUCCESS) {
                throw std::runtime_error("Failed to synchronize Vulkan queue after submit");
            }
            cleanupInflight();
        }
    }

    void memcpy(Buffer& dst, const Buffer& src, size_t size) override {
        // Vulkan doesn't have direct memcpy - need to use buffer copy
        VkCommandBuffer cmdBuf = allocateCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;

        VkBuffer dstBuf = reinterpret_cast<VkBuffer>(dst.nativeHandle());
        VkBuffer srcBuf = reinterpret_cast<VkBuffer>(src.nativeHandle());

        vkCmdCopyBuffer(cmdBuf, srcBuf, dstBuf, 1, &copyRegion);

        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;

        if (vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit Vulkan copy commands");
        }
        inflightCmdBuffers_.push_back(cmdBuf);

        if (vkQueueWaitIdle(queue_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to synchronize Vulkan queue after memcpy");
        }
        cleanupInflight();
    }

    void synchronize() override {
        if (vkQueueWaitIdle(queue_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to synchronize Vulkan queue");
        }
        cleanupInflight();
    }

    void* nativeHandle() override {
        return queue_;
    }

    // Event recording
    void record(Event& event) override {
        VulkanEvent* vkEvent = static_cast<VulkanEvent*>(&event);
        VkCommandBuffer cmdBuf = allocateCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        vkCmdSetEvent(cmdBuf, vkEvent->get(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;

        if (vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record Vulkan event");
        }
        inflightCmdBuffers_.push_back(cmdBuf);
    }

    // Wait for event
    void wait(Event& event) override {
        VulkanEvent* vkEvent = static_cast<VulkanEvent*>(&event);
        VkEvent ev = vkEvent->get();  // Store event to avoid temporary
        VkCommandBuffer cmdBuf = allocateCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        // vkCmdWaitEvents requires srcStageMask, dstStageMask, and barrier arrays
        // For simple event waiting, we use minimal barriers
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdWaitEvents(
            cmdBuf,
            1, &ev,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,      // srcStageMask
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,      // dstStageMask
            1, &memoryBarrier,                          // memoryBarrierCount, pMemoryBarriers
            0, nullptr,                                 // bufferMemoryBarrierCount, pBufferMemoryBarriers
            0, nullptr                                  // imageMemoryBarrierCount, pImageMemoryBarriers
        );

        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;

        if (vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to wait on Vulkan event");
        }
        inflightCmdBuffers_.push_back(cmdBuf);
    }

    // Wait with timeout
    bool waitFor(Event& event, uint64_t timeout_us) override {
        VulkanEvent* vkEvent = static_cast<VulkanEvent*>(&event);
        uint64_t timeout_ns = timeout_us * 1000;

        // Vulkan doesn't have vkWaitForEvents - poll with vkGetEventStatus
        auto start = std::chrono::steady_clock::now();
        while (true) {
            VkResult result = vkGetEventStatus(device_, vkEvent->get());
            if (result == VK_EVENT_SET) return true;
            if (result != VK_EVENT_RESET) return false;  // Error

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start);
            if (elapsed.count() >= timeout_ns) return false;

            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Multi-queue synchronization
    void wait(Event& event, Queue& other) override {
        // Vulkan events are timeline-compatible by default
        (void)other;
        wait(event);
    }

private:
    VkCommandBuffer allocateCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool_;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuf;
        if (vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuf) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate Vulkan command buffer");
        }
        return cmdBuf;
    }

    void cleanupInflight() {
        if (!inflightCmdBuffers_.empty()) {
            vkFreeCommandBuffers(device_, commandPool_,
                                 static_cast<uint32_t>(inflightCmdBuffers_.size()),
                                 inflightCmdBuffers_.data());
            inflightCmdBuffers_.clear();
        }

        for (VkPipeline p : inflightPipelines_) {
            if (p) vkDestroyPipeline(device_, p, nullptr);
        }
        inflightPipelines_.clear();

        for (VkPipelineLayout l : inflightPipelineLayouts_) {
            if (l) vkDestroyPipelineLayout(device_, l, nullptr);
        }
        inflightPipelineLayouts_.clear();

        inflightSetLayouts_.clear();

        // Descriptor sets are allocated per-dispatch; after `vkQueueWaitIdle` they are safe to recycle.
        if (descriptorPool_) {
            descriptorPool_->reset();
        }
    }

    VkDevice device_;
    VkQueue queue_;
    uint32_t queueFamilyIndex_;
    VkCommandPool commandPool_;
    std::unique_ptr<VulkanDescriptorPool> descriptorPool_;  //

    std::vector<VkCommandBuffer> inflightCmdBuffers_;
    std::vector<VkPipeline> inflightPipelines_;
    std::vector<VkPipelineLayout> inflightPipelineLayouts_;
    std::vector<std::unique_ptr<VulkanDescriptorSetLayout>> inflightSetLayouts_;
};

// ============================================================================
// VulkanBuffer - Vulkan device memory buffer
// ============================================================================

class VulkanBuffer : public Buffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                size_t size, MemoryKind kind)
        : device_(device), size_(size), kind_(kind) {

        VkMemoryRequirements memReq{};
        VkBuffer buffer;

        // Create buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        // Note: Host visibility is handled by memory property selection in findMemoryType(),
        // not by buffer usage flags. VK_BUFFER_USAGE_HOST_VISIBLE_BIT is a memory property.

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan buffer");
        }

        vkGetBufferMemoryRequirements(device_, buffer, &memReq);

        // Find memory type.
        //
        // Vulkan has no "unified/managed" memory like CUDA. For API consistency we treat
        // `Managed` as host-visible memory (similar to HostPinned) so `map()` works.
        VkMemoryPropertyFlags memProps = 0;
        switch (kind) {
            case MemoryKind::Device:
                memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                break;
            case MemoryKind::HostPinned:
            case MemoryKind::Managed:
                // Prefer coherent host-visible memory for simplicity.
                memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                break;
        }

        uint32_t memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, memProps);

        // Allocate memory
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        VkDeviceMemory memory;
        if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("Failed to allocate Vulkan memory");
        }

        // Bind memory
        if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
            vkFreeMemory(device_, memory, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("Failed to bind Vulkan buffer memory");
        }

        buffer_ = buffer;
        memory_ = memory;

        // Map if host-visible (HostPinned + Managed).
        if (kind == MemoryKind::HostPinned || kind == MemoryKind::Managed) {
            if (vkMapMemory(device_, memory, 0, size, 0, &mappedPtr_) != VK_SUCCESS) {
                throw std::runtime_error("Failed to map Vulkan buffer");
            }
        }
    }

    ~VulkanBuffer() override {
        if (!isLiveDevice(device_)) {
            // The device was already destroyed (e.g. process-global caches outlived the device).
            // We must not call Vulkan destruction APIs in that case.
            return;
        }
        if (mappedPtr_) {
            vkUnmapMemory(device_, memory_);
        }
        if (memory_) {
            vkFreeMemory(device_, memory_, nullptr);
        }
        if (buffer_) {
            vkDestroyBuffer(device_, buffer_, nullptr);
        }
    }

    size_t sizeBytes() const override { return size_; }
    MemoryKind memoryKind() const override { return kind_; }

    void* getDevicePointer() override {
        // Vulkan doesn't expose device pointers directly
        return nullptr;
    }

    void* getHostPointer() override {
        return mappedPtr_;
    }

    void map(MapMode mode) override {
        if (kind_ != MemoryKind::HostPinned && kind_ != MemoryKind::Managed) {
            throw std::runtime_error("Cannot map non-host-visible Vulkan buffer");
        }

        // Ensure device writes are visible to the host for reads.
        // For HOST_COHERENT memory this is a no-op, but it's safe.
        if (mode == MapMode::Read || mode == MapMode::ReadWrite) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = memory_;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(device_, 1, &range);
        }
    }

    void unmap() override {
        if (kind_ != MemoryKind::HostPinned && kind_ != MemoryKind::Managed) {
            return;
        }

        // Ensure host writes are visible to the device.
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = memory_;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device_, 1, &range);
    }

    VkBuffer getBuffer() const { return buffer_; }

    void* nativeHandle() const override {
        return reinterpret_cast<void*>(buffer_);
    }

private:
    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits,
                             VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }

    VkDevice device_;
    size_t size_;
    MemoryKind kind_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mappedPtr_ = nullptr;
};

// ============================================================================
// VulkanProgram - Compiled Vulkan compute shader
// ============================================================================

class VulkanProgram : public Program {
public:
    VulkanProgram(VkDevice device, const ProgramSource& source) : device_(device) {
        if (source.kind == ProgramSource::Kind::Glsl) {
            // Compile GLSL → SPIR-V on-the-fly via glslc.
            std::string glslText(source.bytes.begin(), source.bytes.end());
            spirvBytes_ = compileGlslToSpirv(glslText);
        } else if (source.kind == ProgramSource::Kind::Spirv) {
            spirvBytes_ = source.bytes;
        } else {
            throw std::runtime_error(
                "Vulkan backend requires SPIR-V or GLSL (ProgramSource::Kind::Spirv or Kind::Glsl)");
        }

        if (spirvBytes_.empty() || (spirvBytes_.size() % 4) != 0) {
            throw std::runtime_error("Invalid SPIR-V bytecode (size must be a non-zero multiple of 4)");
        }

        // Ensure `pCode` is 4-byte aligned (VkShaderModuleCreateInfo requires uint32_t words).
        std::vector<std::uint32_t> words(spirvBytes_.size() / 4);
        std::memcpy(words.data(), spirvBytes_.data(), spirvBytes_.size());

        // Create shader module
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spirvBytes_.size();
        createInfo.pCode = words.data();

        if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan shader module");
        }

        // This requires descriptor set layout, pipeline layout, etc.
        // we'll skip full pipeline creation
    }

    ~VulkanProgram() override {
        if (shaderModule_) {
            // See `g_liveDevices` note above: avoid destroying after device teardown.
            if (isLiveDevice(device_)) {
                vkDestroyShaderModule(device_, shaderModule_, nullptr);
            }
        }
    }

    void* nativeHandle() override {
        return toNativeHandle(shaderModule_);
    }

    // Get compiled SPIR-V bytes for disk caching
    std::vector<std::uint8_t> getCompiledBytes() const override {
        return spirvBytes_;
    }

    // Get compiled format identifier
    std::string getCompiledFormat() const override {
        return "spv";
    }

private:
    VkDevice device_;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    std::vector<std::uint8_t> spirvBytes_;  // Store original SPIR-V bytes
};

// ============================================================================
// VulkanDevice - Vulkan device implementation
// ============================================================================

class VulkanDevice : public Device {
public:
    VulkanDevice(int device_id, VkInstance instance, VkPhysicalDevice physicalDevice)
        : instance_(instance), physicalDevice_(physicalDevice), device_(VK_NULL_HANDLE) {
        (void)device_id;

        // Get queue family properties
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        // Find compute queue family
        uint32_t queueFamilyIndex = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                break;
            }
        }

        if (queueFamilyIndex == UINT32_MAX) {
            throw std::runtime_error("Vulkan device has no compute queue family");
        }

        // Create device
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        // Enable extensions opportunistically. None of these are currently required for the
        // runtime hot path (we mostly use VkEvent/Fence + dispatch).
        const char* enabledExts[1];
        uint32_t enabledExtCount = 0;
        {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            if (extCount > 0) {
                vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());
            }
            const auto hasExt = [&](const char* name) {
                for (const auto& e : exts) {
                    if (std::strcmp(e.extensionName, name) == 0) return true;
                }
                return false;
            };
            if (hasExt(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
                enabledExts[enabledExtCount++] = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
            }
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.enabledExtensionCount = enabledExtCount;
        createInfo.ppEnabledExtensionNames = (enabledExtCount > 0) ? enabledExts : nullptr;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan device");
        }
        registerLiveDevice(device_);

        queueFamilyIndex_ = queueFamilyIndex;
    }

    ~VulkanDevice() override {
        if (device_) {
            unregisterLiveDevice(device_);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    std::unique_ptr<Queue> createQueue(StreamPriority priority) override {
        // Get queue
        VkQueue queue;
        vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue);
        return std::make_unique<VulkanQueue>(device_, queue, queueFamilyIndex_);
    }

    std::unique_ptr<Buffer> createBuffer(size_t size, MemoryKind kind) override {
        return std::make_unique<VulkanBuffer>(device_, physicalDevice_, size, kind);
    }

    std::unique_ptr<Program> createProgram(const ProgramSource& source) override {
        return std::make_unique<VulkanProgram>(device_, source);
    }

    // Graph creation (not supported for Vulkan)
    std::unique_ptr<Graph> createGraph() override {
        throw std::runtime_error("Vulkan backend does not support graphs");
    }

    std::unique_ptr<Event> createEvent() override {
        // Use VkEvent for GPU-side record/wait semantics (vkCmdSetEvent/vkCmdWaitEvents).
        VkEventCreateInfo eventInfo{};
        eventInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
        eventInfo.flags = 0;

        VkEvent ev = VK_NULL_HANDLE;
        if (vkCreateEvent(device_, &eventInfo, nullptr, &ev) != VK_SUCCESS) {
            return nullptr;
        }
        return std::make_unique<VulkanEvent>(device_, ev);
    }

    // Device metadata
    std::string getName() const override {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        return props.deviceName;
    }

    size_t getTotalMemory() const override {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
        return memProps.memoryHeaps[0].size;
    }

    std::string getComputeCapability() const override {
        return "1.3";  // Vulkan version
    }

    bool supportsFeature(const std::string& feature) const override {
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(physicalDevice_, &features);
        if (feature == "float64" || feature == "fp64") return features.shaderFloat64 == VK_TRUE;
        // Be conservative: Float16 requires additional feature structs; only claim support when
        // explicitly enabled/queried elsewhere.
        if (feature == "float16" || feature == "fp16") return false;
        if (feature == "bfloat16" || feature == "bf16") return false;
        if (feature == "graphs") return false;  // Vulkan doesn't have CUDA/HIP-style graphs
        if (feature == "managedMemory") return false;
        if (feature == "sharedMem") return true;
        if (feature == "streams") return true;
        if (feature == "events") return true;
        if (feature == "tensor_cores") return false;
        if (feature == "memory_pools") return true;
        if (feature == "nurbs.rational.evaluate") return features.shaderFloat64 == VK_TRUE;
        if (feature == "nurbs.rational.derivatives") return features.shaderFloat64 == VK_TRUE;
        if (feature == "nurbs.rational.project") return features.shaderFloat64 == VK_TRUE;
        return false;
    }

    // Extended capability queries
    DeviceVersion getVersion() const override {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        return DeviceVersion{
            static_cast<int>(VK_API_VERSION_MAJOR(props.apiVersion)),
            static_cast<int>(VK_API_VERSION_MINOR(props.apiVersion)),
            static_cast<int>(VK_API_VERSION_PATCH(props.apiVersion))
        };
    }

    bool hasExtension(const std::string& extension) const override {
        // Query available extensions
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extensionCount, extensions.data());

        for (const auto& ext : extensions) {
            if (extension == ext.extensionName) return true;
        }
        return false;
    }

    uint32_t getSubgroupSize() const override {
        // Query subgroup properties
        VkPhysicalDeviceSubgroupProperties subgroupProps{};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;

        vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);
        return subgroupProps.subgroupSize;
    }

    bool supportsAtomicOp(AtomicOpType op, DataType type) const override {
        (void)op;
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(physicalDevice_, &features);

        // Be conservative: without explicit `VK_EXT_shader_atomic_float*` feature probing
        // and enabling, we do not claim float atomic support.
	        switch (type) {
	            case DataType::Float64:
	            case DataType::Float32:
	            case DataType::Float16:
	            case DataType::BFloat16:
	                return false;
	            case DataType::Complex64:
	            case DataType::Complex128:
	                return false;
	            case DataType::Int64:
	            case DataType::UInt64:
	                return features.shaderInt64;
	            case DataType::Int32:
	            case DataType::UInt32:
	                return true;  // Vulkan 1.0+ supports 32-bit atomics
	            case DataType::Int16:
	            case DataType::UInt16:
	            case DataType::Int8:
	            case DataType::UInt8:
	                return false;  // Small integer atomics not directly supported
	        }
        return false;
    }

    size_t getMaxSharedMemoryPerWorkgroup() const override {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        return props.limits.maxComputeSharedMemorySize;
    }

    uint32_t getMaxWorkgroupSize() const override {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        return props.limits.maxComputeWorkGroupInvocations;
    }

    std::vector<std::string> getSupportedExtensions() const override {
        // Query available extensions
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extensionCount, extensions.data());

        std::vector<std::string> result;
        for (const auto& ext : extensions) {
            result.push_back(ext.extensionName);
        }
        return result;
    }

private:
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    uint32_t queueFamilyIndex_;
};

// Factory function (returns nullptr if Vulkan not available)
std::unique_ptr<Device> VulkanDevice_tryCreate(int device_id) {
    try {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "gpu_runtime";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "gpu_runtime";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        // On macOS with MoltenVK we need the portability enumeration
        // extension and the corresponding create flag.
        std::vector<const char*> instanceExts;
#ifdef __APPLE__
        instanceExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
        createInfo.ppEnabledExtensionNames = instanceExts.empty() ? nullptr : instanceExts.data();
#ifdef __APPLE__
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        uint32_t deviceCount = 0;
        VkResult enumRes = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (enumRes != VK_SUCCESS || deviceCount == 0) {
            vkDestroyInstance(instance, nullptr);
            throw std::runtime_error("No Vulkan physical devices found");
        }
        std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
        if (vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()) != VK_SUCCESS) {
            vkDestroyInstance(instance, nullptr);
            throw std::runtime_error("Failed to enumerate Vulkan physical devices");
        }

        if (device_id < 0 || static_cast<uint32_t>(device_id) >= deviceCount) {
            vkDestroyInstance(instance, nullptr);
            throw std::runtime_error("Invalid Vulkan device ID");
        }

        VkPhysicalDevice physicalDevice = physicalDevices[static_cast<uint32_t>(device_id)];
        // VulkanDevice takes ownership of instance and destroys it in its destructor.
        return std::make_unique<VulkanDevice>(device_id, instance, physicalDevice);
    } catch (const std::exception& e) {
        std::cerr << "Vulkan backend unavailable: " << e.what() << "\n";
        return nullptr;
    }
}

} // namespace GpuRuntime
