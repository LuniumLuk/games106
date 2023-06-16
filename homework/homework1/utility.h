#pragma once

#include "vulkanexamplebase.h"

namespace hw1 {

	struct Image2D {
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory mem = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkSampler sampler = VK_NULL_HANDLE;
		VkDescriptorImageInfo descriptor{};
	};

	void createImage2D(vks::VulkanDevice const& device, Image2D* image, VkFormat format, VkExtent3D extent, VkImageUsageFlags usage, VkImageAspectFlags aspect,
		VkFilter filter = VK_FILTER_LINEAR, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

	void destroyImage2D(VkDevice device, Image2D* image);

	void createRenderPass(VkDevice device, VkRenderPass* renderPass, VkFormat colorFormat, VkFormat depthFormat = VK_FORMAT_UNDEFINED);

	void createFrameBuffer(VkDevice device, VkFramebuffer* frameBuffer, VkExtent2D extent, VkRenderPass renderPass, VkImageView colorView, VkImageView depthView = VK_NULL_HANDLE);

	struct Timer {
		Timer();
		void update() noexcept;
		double deltaTime() noexcept;
		double totalTime() noexcept;

	private:
		std::chrono::steady_clock::time_point startTimePoint;
		std::chrono::steady_clock::time_point prevTimePoint;
		double delta = 0.0;
	};

} // namespace hw1
