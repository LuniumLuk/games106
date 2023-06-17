/*
* Vulkan Example - Variable rate shading
*
* Copyright (C) 2020-2022 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "homework2.h"

VulkanExample::VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
{
	title = "Variable rate shading";
	apiVersion = VK_API_VERSION_1_1;
	camera.type = Camera::CameraType::firstperson;
	camera.flipY = true;
	camera.setPosition(glm::vec3(0.0f, 1.0f, 0.0f));
	camera.setRotation(glm::vec3(0.0f, -90.0f, 0.0f));
	camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	camera.setRotationSpeed(0.25f);
	enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
	enabledDeviceExtensions.push_back(VK_NV_SHADING_RATE_IMAGE_EXTENSION_NAME);
	enabledDeviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
	enabledDeviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
}

VulkanExample::~VulkanExample()
{
	vkDestroyPipeline(device, main.basePipelines.masked, nullptr);
	vkDestroyPipeline(device, main.basePipelines.opaque, nullptr);
	vkDestroyPipeline(device, main.shadingRatePipelines.masked, nullptr);
	vkDestroyPipeline(device, main.shadingRatePipelines.opaque, nullptr);
	vkDestroyPipelineLayout(device, main.pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, main.descriptorSetLayout, nullptr);
	vkDestroyRenderPass(device, main.renderPass, nullptr);
	vkDestroyFramebuffer(device, main.frameBuffer.frameBuffer, nullptr);
	vkDestroySemaphore(device, main.semaphore, nullptr);

	vkDestroyPipeline(device, present.pipeline, nullptr);
	vkDestroyPipelineLayout(device, present.pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, present.descriptorSetLayout, nullptr);

	vkDestroyPipeline(device, compute.pipeline, nullptr);
	vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
	vkDestroySemaphore(device, compute.semaphore, nullptr);
	vkDestroyCommandPool(device, compute.commandPool, nullptr);

	main.frameBuffer.colorImage.destroy();
	main.frameBuffer.depthImage.destroy();
	main.frameBuffer.shadingRateVisualizeImage.destroy();
	compute.errorHistoryImage.destroy();
	shadingRateImage.destroy();
	sceneUBO.buffer.destroy();
	computeUBO.buffer.destroy();
}

void VulkanExample::getEnabledFeatures()
{
	enabledFeatures.samplerAnisotropy = deviceFeatures.samplerAnisotropy;
	// POI
	enabledPhysicalDeviceShadingRateImageFeaturesNV = {};
	enabledPhysicalDeviceShadingRateImageFeaturesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES_NV;
	enabledPhysicalDeviceShadingRateImageFeaturesNV.shadingRateImage = VK_TRUE;
	deviceCreatepNextChain = &enabledPhysicalDeviceShadingRateImageFeaturesNV;

	floatFeatures = {};
	floatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	floatFeatures.shaderSharedFloat32Atomics = VK_TRUE;
	floatFeatures.shaderSharedFloat32AtomicAdd = VK_TRUE;
	enabledPhysicalDeviceShadingRateImageFeaturesNV.pNext = &floatFeatures;
}

/*
	If the window has been resized, we need to recreate the shading rate image
*/
void VulkanExample::handleResize()
{
	// Delete allocated resources
	vkDestroyImageView(device, shadingRateImage.view, nullptr);
	vkDestroyImage(device, shadingRateImage.image, nullptr);
	vkFreeMemory(device, shadingRateImage.deviceMemory, nullptr);
	// Recreate image
	prepareShadingRateImage();
	resized = false;
}

void VulkanExample::buildCommandBuffers()
{
	if (resized)
	{
		handleResize();
	}

	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

	std::vector<VkClearValue> clearValues(3);
	clearValues[0].color = defaultClearColor;
	clearValues[1].depthStencil = { 1.0f, 0 };
	clearValues[2].color = { { 1.0f, 1.0f, 1.0f, 1.0f } };

	VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset.x = 0;
	renderPassBeginInfo.renderArea.offset.y = 0;
	renderPassBeginInfo.renderArea.extent.width = width;
	renderPassBeginInfo.renderArea.extent.height = height;
	renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues = clearValues.data();

	const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
	const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

	for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
	{
		VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

		renderPassBeginInfo.renderPass = main.renderPass;
		renderPassBeginInfo.framebuffer = main.frameBuffer.frameBuffer;
		vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
		vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

		vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, main.pipelineLayout, 0, 1, &main.descriptorSet, 0, nullptr);

		// POI: Bind the image that contains the shading rate patterns
		if (enableShadingRate) {
			vkCmdBindShadingRateImageNV(drawCmdBuffers[i], shadingRateImage.view, VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV);
		};

		// render the scene
		Pipelines& pipelines = enableShadingRate ? main.shadingRatePipelines : main.basePipelines;
		vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.opaque);
		scene.draw(drawCmdBuffers[i], vkglTF::RenderFlags::BindImages | vkglTF::RenderFlags::RenderOpaqueNodes, main.pipelineLayout);
		vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.masked);
		scene.draw(drawCmdBuffers[i], vkglTF::RenderFlags::BindImages | vkglTF::RenderFlags::RenderAlphaMaskedNodes, main.pipelineLayout);

		vkCmdEndRenderPass(drawCmdBuffers[i]);

		renderPassBeginInfo.renderPass = present.renderPass;
		renderPassBeginInfo.framebuffer = frameBuffers[i];
		// present to swapChain
		vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
		vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

		vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, present.pipelineLayout, 0, 1, &present.descriptorSet, 0, nullptr);

		vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, present.pipeline);
		vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

		drawUI(drawCmdBuffers[i]);
		vkCmdEndRenderPass(drawCmdBuffers[i]);

		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		{
			VkImageMemoryBarrier imageMemoryBarrier{};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.subresourceRange = subresourceRange;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV;
			imageMemoryBarrier.image = shadingRateImage.image;
			vkCmdPipelineBarrier(drawCmdBuffers[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageMemoryBarrier.image = main.frameBuffer.colorImage.image;
			vkCmdPipelineBarrier(drawCmdBuffers[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageMemoryBarrier.image = main.frameBuffer.shadingRateVisualizeImage.image;
			vkCmdPipelineBarrier(drawCmdBuffers[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}

		VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
	}
}

void VulkanExample::loadAssets()
{
	vkglTF::descriptorBindingFlags = vkglTF::DescriptorBindingFlags::ImageBaseColor | vkglTF::DescriptorBindingFlags::ImageNormalMap;
	scene.loadFromFile(getAssetPath() + "models/sponza/sponza.gltf", vulkanDevice, queue, vkglTF::FileLoadingFlags::PreTransformVertices);
}

void VulkanExample::setupDescriptors()
{
	// Pool
	const std::vector<VkDescriptorPoolSize> poolSizes = {
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4),
	};
	VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);
	VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

	// Descriptor set layout
	const std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
	};
	VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &main.descriptorSetLayout));

	// Pipeline layout
	const std::vector<VkDescriptorSetLayout> setLayouts = {
		main.descriptorSetLayout,
		vkglTF::descriptorSetLayoutImage,
	};
	VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), 2);
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &main.pipelineLayout));

	// Descriptor set
	VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &main.descriptorSetLayout, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &main.descriptorSet));
	std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
		vks::initializers::writeDescriptorSet(main.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &sceneUBO.buffer.descriptor),
	};
	vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

// [POI]
void VulkanExample::prepareShadingRateImage()
{
	// Shading rate image size depends on shading rate texel size
	// For each texel in the target image, there is a corresponding shading texel size width x height block in the shading rate image
	VkExtent3D imageExtent{};
	imageExtent.width = static_cast<uint32_t>(ceil(width / (float)physicalDeviceShadingRateImagePropertiesNV.shadingRateTexelSize.width));
	imageExtent.height = static_cast<uint32_t>(ceil(height / (float)physicalDeviceShadingRateImagePropertiesNV.shadingRateTexelSize.height));
	imageExtent.depth = 1;

	{
		shadingRateImage.width = imageExtent.width;
		shadingRateImage.height = imageExtent.height;

		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = VK_FORMAT_R8_UINT;
		imageCI.extent = imageExtent;
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		// this image will also be used as storage target in the compute shader
		imageCI.usage = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		// If compute and graphics queue family indices differ, we create an image that can be shared between them
		// This can result in worse performance than exclusive sharing mode, but save some synchronization to keep the sample simple
		std::vector<uint32_t> queueFamilyIndices;
		if (vulkanDevice->queueFamilyIndices.graphics != vulkanDevice->queueFamilyIndices.compute) {
			queueFamilyIndices = {
				vulkanDevice->queueFamilyIndices.graphics,
				vulkanDevice->queueFamilyIndices.compute
			};
			imageCI.sharingMode = VK_SHARING_MODE_CONCURRENT;
			imageCI.queueFamilyIndexCount = 2;
			imageCI.pQueueFamilyIndices = queueFamilyIndices.data();
		}

		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &shadingRateImage.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device, shadingRateImage.image, &memReqs);

		VkDeviceSize bufferSize = imageExtent.width * imageExtent.height * sizeof(uint8_t);

		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &shadingRateImage.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, shadingRateImage.image, shadingRateImage.deviceMemory, 0));

		VkImageViewCreateInfo imageViewCI{};
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = shadingRateImage.image;
		imageViewCI.format = VK_FORMAT_R8_UINT;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &shadingRateImage.view));

		shadingRateImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		shadingRateImage.descriptor.imageView = shadingRateImage.view;
		shadingRateImage.descriptor.sampler = shadingRateImage.sampler = VK_NULL_HANDLE;
		shadingRateImage.device = vulkanDevice;

		// Populate with lowest possible shading rate pattern
		uint8_t val = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X4_PIXELS_NV;
		uint8_t* shadingRatePatternData = new uint8_t[bufferSize];
		memset(shadingRatePatternData, val, bufferSize);

		// Create a circular pattern with decreasing sampling rates outwards (max. range, pattern)
		std::map<float, VkShadingRatePaletteEntryNV> patternLookup = {
			{ 8.0f, VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_PIXEL_NV },
			{ 12.0f, VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X1_PIXELS_NV },
			{ 16.0f, VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_1X2_PIXELS_NV },
			{ 18.0f, VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X2_PIXELS_NV },
			{ 20.0f, VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X2_PIXELS_NV },
			{ 24.0f, VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X4_PIXELS_NV }
		};

		uint8_t* ptrData = shadingRatePatternData;
		for (uint32_t y = 0; y < imageExtent.height; y++) {
			for (uint32_t x = 0; x < imageExtent.width; x++) {
				const float deltaX = (float)imageExtent.width / 2.0f - (float)x;
				const float deltaY = ((float)imageExtent.height / 2.0f - (float)y) * ((float)width / (float)height);
				const float dist = std::sqrt(deltaX * deltaX + deltaY * deltaY);
				for (auto pattern : patternLookup) {
					if (dist < pattern.first) {
						*ptrData = pattern.second;
						break;
					}
				}
				ptrData++;
			}
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;

		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = bufferSize;
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &stagingBuffer));
		VkMemoryAllocateInfo memAllocInfo{};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memReqs = {};
		vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

		uint8_t* mapped;
		VK_CHECK_RESULT(vkMapMemory(device, stagingMemory, 0, memReqs.size, 0, (void**)&mapped));
		memcpy(mapped, shadingRatePatternData, bufferSize);
		vkUnmapMemory(device, stagingMemory);

		delete[] shadingRatePatternData;

		// Upload
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		{
			VkImageMemoryBarrier imageMemoryBarrier{};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.srcAccessMask = 0;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.image = shadingRateImage.image;
			imageMemoryBarrier.subresourceRange = subresourceRange;
			// to solve VUID-VkImageMemoryBarrier-synchronization2-03857
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}
		VkBufferImageCopy bufferCopyRegion{};
		bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		bufferCopyRegion.imageSubresource.layerCount = 1;
		bufferCopyRegion.imageExtent.width = imageExtent.width;
		bufferCopyRegion.imageExtent.height = imageExtent.height;
		bufferCopyRegion.imageExtent.depth = 1;
		vkCmdCopyBufferToImage(copyCmd, stagingBuffer, shadingRateImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferCopyRegion);
		{
			VkImageMemoryBarrier imageMemoryBarrier{};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = 0;
			imageMemoryBarrier.image = shadingRateImage.image;
			imageMemoryBarrier.subresourceRange = subresourceRange;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		vkFreeMemory(device, stagingMemory, nullptr);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
	}
	// error history image
	{
		auto const format = VK_FORMAT_R32G32_SFLOAT;

		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = imageExtent;
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &compute.errorHistoryImage.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device, compute.errorHistoryImage.image, &memReqs);

		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &compute.errorHistoryImage.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, compute.errorHistoryImage.image, compute.errorHistoryImage.deviceMemory, 0));

		VkImageViewCreateInfo imageViewCI{};
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = compute.errorHistoryImage.image;
		imageViewCI.format = format;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &compute.errorHistoryImage.view));

		compute.errorHistoryImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		compute.errorHistoryImage.descriptor.imageView = compute.errorHistoryImage.view;
		compute.errorHistoryImage.descriptor.sampler = compute.errorHistoryImage.sampler = VK_NULL_HANDLE;
		compute.errorHistoryImage.device = vulkanDevice;

		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		{
			VkImageMemoryBarrier imageMemoryBarrier{};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = compute.errorHistoryImage.image;
			imageMemoryBarrier.subresourceRange = subresourceRange;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}
		vulkanDevice->flushCommandBuffer(commandBuffer, queue, true);
	}
}

void VulkanExample::preparePipelines()
{
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
	VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
	const std::vector<VkPipelineColorBlendAttachmentState> blendAttachmentStateCIs = {
		vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
		vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
	};
	VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(static_cast<uint32_t>(blendAttachmentStateCIs.size()), blendAttachmentStateCIs.data());
	VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
	VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
	VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
	const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
	std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

	VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(main.pipelineLayout, main.renderPass, 0);
	pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
	pipelineCI.pRasterizationState = &rasterizationStateCI;
	pipelineCI.pColorBlendState = &colorBlendStateCI;
	pipelineCI.pMultisampleState = &multisampleStateCI;
	pipelineCI.pViewportState = &viewportStateCI;
	pipelineCI.pDepthStencilState = &depthStencilStateCI;
	pipelineCI.pDynamicState = &dynamicStateCI;
	pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineCI.pStages = shaderStages.data();
	pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Tangent });

	shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework2/scene.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
	shaderStages[1] = loadShader(getHomeworkShadersPath() + "homework2/scene.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

	// Properties for alpha masked materials will be passed via specialization constants
	struct SpecializationData {
		VkBool32 alphaMask;
		float alphaMaskCutoff;
	} specializationData;
	specializationData.alphaMask = false;
	specializationData.alphaMaskCutoff = 0.5f;
	const std::vector<VkSpecializationMapEntry> specializationMapEntries = {
		vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, alphaMask), sizeof(SpecializationData::alphaMask)),
		vks::initializers::specializationMapEntry(1, offsetof(SpecializationData, alphaMaskCutoff), sizeof(SpecializationData::alphaMaskCutoff)),
	};
	VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(specializationMapEntries, sizeof(specializationData), &specializationData);
	shaderStages[1].pSpecializationInfo = &specializationInfo;

	// Create pipeline without shading rate 
	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &main.basePipelines.opaque));
	specializationData.alphaMask = true;
	rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &main.basePipelines.masked));
	rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
	specializationData.alphaMask = false;

	// Create pipeline with shading rate enabled
	// [POI] Possible per-Viewport shading rate palette entries
	const std::vector<VkShadingRatePaletteEntryNV> shadingRatePaletteEntries = {
		VK_SHADING_RATE_PALETTE_ENTRY_NO_INVOCATIONS_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_16_INVOCATIONS_PER_PIXEL_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_8_INVOCATIONS_PER_PIXEL_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_4_INVOCATIONS_PER_PIXEL_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_2_INVOCATIONS_PER_PIXEL_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_PIXEL_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X1_PIXELS_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_1X2_PIXELS_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X2_PIXELS_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X2_PIXELS_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X4_PIXELS_NV,
		VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X4_PIXELS_NV,
	};
	VkShadingRatePaletteNV shadingRatePalette{};
	shadingRatePalette.shadingRatePaletteEntryCount = static_cast<uint32_t>(shadingRatePaletteEntries.size());
	shadingRatePalette.pShadingRatePaletteEntries = shadingRatePaletteEntries.data();
	VkPipelineViewportShadingRateImageStateCreateInfoNV pipelineViewportShadingRateImageStateCI{};
	pipelineViewportShadingRateImageStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO_NV;
	pipelineViewportShadingRateImageStateCI.shadingRateImageEnable = VK_TRUE;
	pipelineViewportShadingRateImageStateCI.viewportCount = 1;
	pipelineViewportShadingRateImageStateCI.pShadingRatePalettes = &shadingRatePalette;
	viewportStateCI.pNext = &pipelineViewportShadingRateImageStateCI;
	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &main.shadingRatePipelines.opaque));
	specializationData.alphaMask = true;
	rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &main.shadingRatePipelines.masked));
}

void VulkanExample::prepareUniformBuffers()
{
	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&sceneUBO.buffer,
		sizeof(sceneUBO.values)));
	VK_CHECK_RESULT(sceneUBO.buffer.map());

	VK_CHECK_RESULT(vulkanDevice->createBuffer(
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&computeUBO.buffer,
		sizeof(computeUBO.values)));
	VK_CHECK_RESULT(computeUBO.buffer.map());

	updateUniformBuffers();
}

void VulkanExample::updateUniformBuffers()
{
	sceneUBO.values.projection = camera.matrices.perspective;
	sceneUBO.values.view = camera.matrices.view;
	sceneUBO.values.viewPos = camera.viewPos;
	sceneUBO.values.colorShadingRate = colorShadingRate;
	memcpy(sceneUBO.buffer.mapped, &sceneUBO.values, sizeof(sceneUBO.values));

	auto projView = camera.matrices.perspective * camera.matrices.view;
	computeUBO.values.reprojection = previousProjView * glm::inverse(projView);
	computeUBO.values.mode = mode;
	computeUBO.values.sensitivity = sensitivity;
	memcpy(computeUBO.buffer.mapped, &computeUBO.values, sizeof(computeUBO.values));

	previousProjView = projView;
}

void VulkanExample::prepareGraphics()
{
	// semaphore for compute & graphics sync
	VkSemaphoreCreateInfo semaphoreCI = vks::initializers::semaphoreCreateInfo();
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &main.semaphore));

	// signal the semaphore
	VkSubmitInfo submitInfo = vks::initializers::submitInfo();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &main.semaphore;
	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK_RESULT(vkQueueWaitIdle(queue));
}

void VulkanExample::prepareCompute()
{
	// get a compute queue from the device
	vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

	// create compute pipeline
	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		// input image (main framebuffer color image)
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0),
		// output image (shading rate texture)
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 2),
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 3),
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
	};

	VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

	VkPipelineLayoutCreateInfo pPipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCI, nullptr, &compute.pipelineLayout));

	VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

	std::array<VkDescriptorImageInfo, 2> inputImageInfo{};
	inputImageInfo[0] = main.frameBuffer.colorImage.descriptor;
	inputImageInfo[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	inputImageInfo[1] = main.frameBuffer.shadingRateVisualizeImage.descriptor;
	inputImageInfo[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
		vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &inputImageInfo[0]),
		vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &inputImageInfo[1]),
		vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, &shadingRateImage.descriptor),
		vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3, &compute.errorHistoryImage.descriptor),
		vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, &computeUBO.buffer.descriptor),
	};
	vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);

	// create compute shader pipeline
	VkComputePipelineCreateInfo computePipelineCI = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
	computePipelineCI.stage = loadShader(getHomeworkShadersPath() + "homework2/asr.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
	VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCI, nullptr, &compute.pipeline));

	// Separate command pool as queue family for compute may be different than graphics
	VkCommandPoolCreateInfo cmdPoolCI = {};
	cmdPoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolCI.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
	cmdPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &compute.commandPool));

	// create a command buffer for compute operations
	VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));

	// semaphore for compute & graphics sync
	VkSemaphoreCreateInfo semaphoreCI = vks::initializers::semaphoreCreateInfo();
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCI, nullptr, &compute.semaphore));
}

void VulkanExample::buildComputeCommandBuffer()
{
	vkQueueWaitIdle(compute.queue);

	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

	VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

	vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
	vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

	vkCmdDispatch(compute.commandBuffer, shadingRateImage.width, shadingRateImage.height, 1);

	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;
	{
		VkImageMemoryBarrier imageMemoryBarrier{};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.subresourceRange = subresourceRange;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV;
		imageMemoryBarrier.image = shadingRateImage.image;
		vkCmdPipelineBarrier(compute.commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.image = main.frameBuffer.colorImage.image;
		vkCmdPipelineBarrier(compute.commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageMemoryBarrier.image = main.frameBuffer.shadingRateVisualizeImage.image;
		vkCmdPipelineBarrier(compute.commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
	}

	vkEndCommandBuffer(compute.commandBuffer);
}

void VulkanExample::prepareMainRenderPass()
{
	// framebuffer color image
	{
		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = swapChain.colorFormat;
		imageCI.extent = VkExtent3D{ width, height, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &main.frameBuffer.colorImage.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device, main.frameBuffer.colorImage.image, &memReqs);

		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &main.frameBuffer.colorImage.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, main.frameBuffer.colorImage.image, main.frameBuffer.colorImage.deviceMemory, 0));

		VkImageViewCreateInfo imageViewCI{};
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = main.frameBuffer.colorImage.image;
		imageViewCI.format = swapChain.colorFormat;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &main.frameBuffer.colorImage.view));

		VkSamplerCreateInfo samplerCI = {};
		samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.mipLodBias = 0.0f;
		samplerCI.compareOp = VK_COMPARE_OP_NEVER;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = 0.0f;
		samplerCI.maxAnisotropy = 1.0f;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &main.frameBuffer.colorImage.sampler));

		main.frameBuffer.colorImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		main.frameBuffer.colorImage.descriptor.imageView = main.frameBuffer.colorImage.view;
		main.frameBuffer.colorImage.descriptor.sampler = main.frameBuffer.colorImage.sampler;
		main.frameBuffer.colorImage.device = vulkanDevice;

		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		{
			VkImageMemoryBarrier imageMemoryBarrier{};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = main.frameBuffer.colorImage.image;
			imageMemoryBarrier.subresourceRange = subresourceRange;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}
		vulkanDevice->flushCommandBuffer(commandBuffer, queue, true);
	}
	// framebuffer depth image
	{
		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = depthFormat;
		imageCI.extent = VkExtent3D{ width, height, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &main.frameBuffer.depthImage.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device, main.frameBuffer.depthImage.image, &memReqs);

		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &main.frameBuffer.depthImage.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, main.frameBuffer.depthImage.image, main.frameBuffer.depthImage.deviceMemory, 0));

		VkImageViewCreateInfo imageViewCI{};
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = main.frameBuffer.depthImage.image;
		imageViewCI.format = depthFormat;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &main.frameBuffer.depthImage.view));

		main.frameBuffer.depthImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		main.frameBuffer.depthImage.descriptor.imageView = main.frameBuffer.depthImage.view;
		main.frameBuffer.depthImage.descriptor.sampler = main.frameBuffer.depthImage.sampler = VK_NULL_HANDLE;
		main.frameBuffer.depthImage.device = vulkanDevice;
	}
	// framebuffer shading rate visualize image
	{
		auto const format = VK_FORMAT_R32G32B32A32_SFLOAT;

		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = VkExtent3D{ width, height, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &main.frameBuffer.shadingRateVisualizeImage.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device, main.frameBuffer.shadingRateVisualizeImage.image, &memReqs);

		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &main.frameBuffer.shadingRateVisualizeImage.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, main.frameBuffer.shadingRateVisualizeImage.image, main.frameBuffer.shadingRateVisualizeImage.deviceMemory, 0));

		VkImageViewCreateInfo imageViewCI{};
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = main.frameBuffer.shadingRateVisualizeImage.image;
		imageViewCI.format = format;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &main.frameBuffer.shadingRateVisualizeImage.view));

		VkSamplerCreateInfo samplerCI = {};
		samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.mipLodBias = 0.0f;
		samplerCI.compareOp = VK_COMPARE_OP_NEVER;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = 0.0f;
		samplerCI.maxAnisotropy = 1.0f;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &main.frameBuffer.shadingRateVisualizeImage.sampler));

		main.frameBuffer.shadingRateVisualizeImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		main.frameBuffer.shadingRateVisualizeImage.descriptor.imageView = main.frameBuffer.shadingRateVisualizeImage.view;
		main.frameBuffer.shadingRateVisualizeImage.descriptor.sampler = main.frameBuffer.shadingRateVisualizeImage.sampler;
		main.frameBuffer.shadingRateVisualizeImage.device = vulkanDevice;

		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;
		{
			VkImageMemoryBarrier imageMemoryBarrier{};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = main.frameBuffer.shadingRateVisualizeImage.image;
			imageMemoryBarrier.subresourceRange = subresourceRange;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}
		vulkanDevice->flushCommandBuffer(commandBuffer, queue, true);
	}
	// renderPass
	{
		std::array<VkAttachmentDescription, 3> attachments{};
		// color attachment
		attachments[0].format = swapChain.colorFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		// depth attachment
		attachments[1].format = depthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		// shading rate visualize attachment
		attachments[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		std::array<VkAttachmentReference, 2> colorReferences{};
		colorReferences[0].attachment = 0;
		colorReferences[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorReferences[1].attachment = 2;
		colorReferences[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 1;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
		subpassDescription.pColorAttachments = colorReferences.data();
		subpassDescription.pDepthStencilAttachment = &depthReference;
		subpassDescription.inputAttachmentCount = 0;
		subpassDescription.pInputAttachments = nullptr;
		subpassDescription.preserveAttachmentCount = 0;
		subpassDescription.pPreserveAttachments = nullptr;
		subpassDescription.pResolveAttachments = nullptr;

		// subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies{};

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

		VkRenderPassCreateInfo renderPassCI{};
		renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassCI.pAttachments = attachments.data();
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassCI.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &main.renderPass));
	}
	// framebuffer
	{
		std::array<VkImageView, 3> attachments = {
			main.frameBuffer.colorImage.view,
			main.frameBuffer.depthImage.view,
			main.frameBuffer.shadingRateVisualizeImage.view,
		};

		VkFramebufferCreateInfo frameBufferCI = {};
		frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferCI.pNext = NULL;
		frameBufferCI.renderPass = main.renderPass;
		frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
		frameBufferCI.pAttachments = attachments.data();
		frameBufferCI.width = width;
		frameBufferCI.height = height;
		frameBufferCI.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &main.frameBuffer.frameBuffer));
	}
}

void VulkanExample::preparePresentRenderPass()
{
	// use the swapchain renderPass
	present.renderPass = renderPass;

	// pass in main framebuffer image
	const std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
	};
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &present.descriptorSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &present.descriptorSetLayout, 1);
	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &present.descriptorSet));
	const std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
		vks::initializers::writeDescriptorSet(present.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &main.frameBuffer.colorImage.descriptor),
		vks::initializers::writeDescriptorSet(present.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &main.frameBuffer.shadingRateVisualizeImage.descriptor),
	};
	vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

	VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&present.descriptorSetLayout, 1);
	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &present.pipelineLayout));

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
	VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
	VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
	VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
	VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
	VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
	VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
	const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
	VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
	const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
		loadShader(getHomeworkShadersPath() + "homework2/present.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
		loadShader(getHomeworkShadersPath() + "homework2/present.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
	};

	VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(present.pipelineLayout, renderPass, 0);
	pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
	pipelineCI.pRasterizationState = &rasterizationStateCI;
	pipelineCI.pColorBlendState = &colorBlendStateCI;
	pipelineCI.pMultisampleState = &multisampleStateCI;
	pipelineCI.pViewportState = &viewportStateCI;
	pipelineCI.pDepthStencilState = &depthStencilStateCI;
	pipelineCI.pDynamicState = &dynamicStateCI;
	pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineCI.pStages = shaderStages.data();
	pipelineCI.pVertexInputState = &vertexInputStateCI;

	VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &present.pipeline));
}

void VulkanExample::prepare()
{
	VulkanExampleBase::prepare();
	loadAssets();

	// [POI]
	vkCmdBindShadingRateImageNV = reinterpret_cast<PFN_vkCmdBindShadingRateImageNV>(vkGetDeviceProcAddr(device, "vkCmdBindShadingRateImageNV"));
	physicalDeviceShadingRateImagePropertiesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES_NV;
	VkPhysicalDeviceProperties2 deviceProperties2{};
	deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties2.pNext = &physicalDeviceShadingRateImagePropertiesNV;
	vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);

	prepareShadingRateImage();
	prepareUniformBuffers();
	setupDescriptors();
	prepareMainRenderPass();
	preparePresentRenderPass();
	prepareGraphics();
	prepareCompute();
	preparePipelines();
	buildComputeCommandBuffer();
	buildCommandBuffers();
	prepared = true;
}

void VulkanExample::renderFrame()
{
	VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	// Submit compute commands
	VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
	computeSubmitInfo.commandBufferCount = 1;
	computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
	computeSubmitInfo.waitSemaphoreCount = 1;
	computeSubmitInfo.pWaitSemaphores = &main.semaphore;
	computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
	computeSubmitInfo.signalSemaphoreCount = 1;
	computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
	VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));

	VulkanExampleBase::prepareFrame();

	VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, semaphores.presentComplete };
	VkSemaphore graphicsSignalSemaphores[] = { main.semaphore, semaphores.renderComplete };

	// submit graphics commands
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
	submitInfo.waitSemaphoreCount = 2;
	submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
	submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
	submitInfo.signalSemaphoreCount = 2;
	submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

	VulkanExampleBase::submitFrame();
}

void VulkanExample::render()
{
	if (!prepared) {
		return;
	}
	renderFrame();
	if (camera.updated) {
		updateUniformBuffers();
	}
}

void VulkanExample::OnUpdateUIOverlay(vks::UIOverlay* overlay)
{
	if (overlay->checkBox("Enable shading rate", &enableShadingRate)) {
		buildCommandBuffers();
	}
	if (overlay->checkBox("Color shading rates", &colorShadingRate)) {
		updateUniformBuffers();
	}
	overlay->sliderFloat("Adaptive sensitivity", &sensitivity, 0.0f, 1.0f);
	if (overlay->radioButton("Content adaptive mode", mode == 0)) {
		mode = 0;
		updateUniformBuffers();
	}
	if (overlay->radioButton("Motion adaptive mode", mode == 1)) {
		mode = 1;
		updateUniformBuffers();
	}
}

VULKAN_EXAMPLE_MAIN()
