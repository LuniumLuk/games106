/*
* Vulkan Example - Variable rate shading
*
* Copyright (C) 2020 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

#define ENABLE_VALIDATION true

class VulkanExample : public VulkanExampleBase
{
public:
	vkglTF::Model scene;

	vks::Texture2D shadingRateImage;

	bool enableShadingRate = true;
	bool colorShadingRate = false;

	struct SceneUBO {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 view;
			glm::mat4 model = glm::mat4(1.0f);
			glm::vec4 lightPos = glm::vec4(0.0f, 2.5f, 0.0f, 1.0f);
			glm::vec4 viewPos;
			int32_t colorShadingRate;
		} values;
	} sceneUBO;

	glm::mat4 previousProjView = glm::mat4(1.0f);
	int mode = 0;
	bool duelThreshold = false; // first frame cannot have duel threshold
	float sensitivity = 0.1f;

	struct ComputeUBO {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 reprojection;
			int mode;
			int duelThreshold;
			float sensitivity;
		} values;
	} computeUBO;

	struct Pipelines {
		VkPipeline opaque;
		VkPipeline masked;
	};

	struct {
		Pipelines basePipelines;
		Pipelines shadingRatePipelines;
		VkSemaphore semaphore;
		VkPipelineLayout pipelineLayout;
		VkDescriptorSet descriptorSet;
		VkDescriptorSetLayout descriptorSetLayout;
		VkRenderPass renderPass;
		struct {
			VkFramebuffer frameBuffer;
			vks::Texture2D colorImage;
			vks::Texture2D depthImage;
			vks::Texture2D shadingRateVisualizeImage;
		} frameBuffer;
	} main;

	struct {
		VkPipeline pipeline;
		VkPipelineLayout pipelineLayout;
		VkDescriptorSet descriptorSet;
		VkDescriptorSetLayout descriptorSetLayout;
		VkRenderPass renderPass;
	} present;

	struct {
		VkQueue queue;
		VkCommandPool commandPool;
		VkCommandBuffer commandBuffer;
		VkSemaphore semaphore;
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		vks::Texture2D errorHistoryImage;
	} compute;

	VkPhysicalDeviceShadingRateImagePropertiesNV physicalDeviceShadingRateImagePropertiesNV{};
	VkPhysicalDeviceShadingRateImageFeaturesNV enabledPhysicalDeviceShadingRateImageFeaturesNV{};
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeatures{};
	PFN_vkCmdBindShadingRateImageNV vkCmdBindShadingRateImageNV;

	VulkanExample();
	~VulkanExample();
	virtual void getEnabledFeatures();
	void handleResize();
	void buildComputeCommandBuffer();
	void buildCommandBuffers();
	void loadglTFFile(std::string filename);
	void loadAssets();
	void prepareShadingRateImage();
	void setupDescriptors();
	void preparePipelines();
	void prepareGraphics();
	void prepareCompute();
	void prepareMainRenderPass();
	void preparePresentRenderPass();
	void prepareUniformBuffers();
	void updateUniformBuffers();
	void prepare();
	virtual void renderFrame();
	virtual void render();
	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay);
};