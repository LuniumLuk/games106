#include "utility.h"

namespace hw1 {

	void createImage2D(vks::VulkanDevice const& device, Image2D* image, VkFormat format, VkExtent3D extent, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = extent;
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = usage;

		VK_CHECK_RESULT(vkCreateImage(device.logicalDevice, &imageCI, nullptr, &image->image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device.logicalDevice, image->image, &memReqs);

		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = device.getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device.logicalDevice, &memAllloc, nullptr, &image->mem));
		VK_CHECK_RESULT(vkBindImageMemory(device.logicalDevice, image->image, image->mem, 0));

		VkImageViewCreateInfo imageViewCI{};
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = image->image;
		imageViewCI.format = format;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.aspectMask = aspect;
		VK_CHECK_RESULT(vkCreateImageView(device.logicalDevice, &imageViewCI, nullptr, &image->view));

		if (usage & VK_IMAGE_USAGE_SAMPLED_BIT) {
			VkSamplerCreateInfo samplerCI = {};
			samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerCI.magFilter = VK_FILTER_LINEAR;
			samplerCI.minFilter = VK_FILTER_LINEAR;
			samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerCI.mipLodBias = 0.0f;
			samplerCI.compareOp = VK_COMPARE_OP_NEVER;
			samplerCI.minLod = 0.0f;
			samplerCI.maxLod = 0.0f;
			samplerCI.maxAnisotropy = 1.0f;
			VK_CHECK_RESULT(vkCreateSampler(device.logicalDevice, &samplerCI, nullptr, &image->sampler));

			image->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			image->descriptor.imageView = image->view;
			image->descriptor.sampler = image->sampler;
		}
	}

	void destroyImage2D(VkDevice device, Image2D* image) {
		if (image->view != VK_NULL_HANDLE) vkDestroyImageView(device, image->view, nullptr);
		if (image->image != VK_NULL_HANDLE) vkDestroyImage(device, image->image, nullptr);
		if (image->sampler != VK_NULL_HANDLE) vkDestroySampler(device, image->sampler, nullptr);
		if (image->mem != VK_NULL_HANDLE) vkFreeMemory(device, image->mem, nullptr);
	}

	void createRenderPass(VkDevice device, VkRenderPass* renderPass, VkFormat colorFormat, VkFormat depthFormat) {
		std::vector<VkAttachmentDescription> attachments(1);
		// Color attachment
		attachments[0].format = colorFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		if (depthFormat != VK_FORMAT_UNDEFINED) {
			attachments.resize(2);
			// Depth attachment
			attachments[1].format = depthFormat;
			attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		}

		VkAttachmentReference colorReference = {};
		colorReference.attachment = 0;
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 1;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = (depthFormat != VK_FORMAT_UNDEFINED) ? &depthReference : nullptr;
		subpassDescription.inputAttachmentCount = 0;
		subpassDescription.pInputAttachments = nullptr;
		subpassDescription.preserveAttachmentCount = 0;
		subpassDescription.pPreserveAttachments = nullptr;
		subpassDescription.pResolveAttachments = nullptr;

		// Subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		dependencies[0].dependencyFlags = 0;

		dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].dstSubpass = 0;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = 0;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
		dependencies[1].dependencyFlags = 0;

		VkRenderPassCreateInfo renderPassCI = {};
		renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassCI.pAttachments = attachments.data();
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassCI.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, renderPass));
	}

	void createFrameBuffer(VkDevice device, VkFramebuffer* frameBuffer, VkExtent2D extent, VkRenderPass renderPass, VkImageView colorView, VkImageView depthView) {
		std::vector<VkImageView> attachments = { colorView };
		if (depthView != VK_NULL_HANDLE) {
			attachments.push_back(depthView);
		}

		VkFramebufferCreateInfo frameBufferCI = {};
		frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferCI.pNext = NULL;
		frameBufferCI.renderPass = renderPass;
		frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
		frameBufferCI.pAttachments = attachments.data();
		frameBufferCI.width = extent.width;
		frameBufferCI.height = extent.height;
		frameBufferCI.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, frameBuffer));
	}

	Timer::Timer() {
		startTimePoint = std::chrono::steady_clock::now();
		prevTimePoint = startTimePoint;
	}

	void Timer::update() noexcept {
		auto now = std::chrono::steady_clock::now();
		uint64_t deltaTimeCount = std::chrono::duration<double, std::micro>(now - prevTimePoint).count();
		delta = 0.000001 * deltaTimeCount;
		prevTimePoint = now;
	}

	double Timer::deltaTime() noexcept {
		return delta;
	}

	double Timer::totalTime() noexcept {
		uint64_t totalTimeCount = std::chrono::duration<double, std::micro>(prevTimePoint - startTimePoint).count();
		return 0.000001 * totalTimeCount;
	}

} // namespace hw1