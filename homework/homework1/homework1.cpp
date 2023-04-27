/*
* Vulkan Example - glTF scene loading and rendering
*
* Copyright (C) 2020-2022 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are shown here
 * This means no complex materials, no animations, no skins, etc.
 * For details on how glTF 2.0 works, see the official spec at https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Other samples will load models using a dedicated model loader with more features (see base/VulkanglTFModel.hpp)
 *
 * If you are looking for a complete glTF implementation, check out https://github.com/SaschaWillems/Vulkan-glTF-PBR/
 */

// [TODO] Animation System
// [TODO] Handle Window Resize
// [TODO] Simplify Code

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif
#include "tiny_gltf.h"

#include "vulkanexamplebase.h"

#define ENABLE_VALIDATION true

// Index for unloaded texture, such texture will be replaced with a default (white | black) texture
#define TEXTURE_INDEX_EMPTY 0xffffffff

#define TEXTURE_DEFAULT_WHITE 2
#define TEXTURE_DEFAULT_BLACK 1

// We are using variance shadow map that store first and second moment of the depth value
#define SHADOW_MAP_COLOR_FORMAT VK_FORMAT_R32G32_SFLOAT
#define SHADOW_MAP_SIZE 1024

// First we render the whole scene to a separate framebuffer with a color attachment and a depth attachment
// Then in the postprocessing stage, we use the color attachment as input and render it to the screen
#define RENDER_TARGET_COLOR_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT

// Contains everything required to render a glTF model in Vulkan
// This class is heavily simplified (compared to glTF's feature set) but retains the basic glTF structure
class VulkanglTFModel
{
public:
	// The class requires some Vulkan objects so it can create it's own resources
	vks::VulkanDevice* vulkanDevice;
	VkQueue copyQueue;

	// The vertex layout for the samples' model
	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec3 color;
	};

	// Single vertex buffer for all primitives
	struct {
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertices;

	// Single index buffer for all primitives
	struct {
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;

	// The following structures roughly represent the glTF scene structure
	// To keep things simple, they only contain those properties that are required for this sample
	struct Node;

	// A primitive contains the data for a single draw call
	struct Primitive {
		uint32_t firstIndex;
		uint32_t indexCount;
		int32_t materialIndex;
	};

	// Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
	struct Mesh {
		std::vector<Primitive> primitives;
	};

	// A node represents an object in the glTF scene graph
	struct Node {
		Node* parent;
		std::vector<Node*> children;
		Mesh mesh;
		glm::mat4 matrix;
		~Node() {
			for (auto& child : children) {
				delete child;
			}
		}
	};

	struct PushConstant {
		glm::mat4 model;
		glm::vec4 baseColorFactor;
		float roughnessFactor;
		float metallicFactor;
	};

	// A glTF material stores information in e.g. the texture that is attached to it and colors
	struct Material {
		glm::vec4 baseColorFactor = glm::vec4(1.0f);
		float roughnessFactor = 1.0f;
		float metallicFactor = 1.0f;
		uint32_t baseColorTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t normalTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t metallicRoughnessTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t emissiveTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t occlusionTextureIndex = TEXTURE_INDEX_EMPTY;
		// We use one set for textures of each material
		VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
	};

	// Contains the texture for a single glTF image
	// Images may be reused by texture objects and are as such separated
	struct Image {
		vks::Texture2D texture;
		// We also store (and create) a descriptor set that's used to access this texture from the fragment shader
		VkDescriptorSet descriptorSet;
	};

	// A glTF texture stores a reference to the image and a sampler
	// In this sample, we are only interested in the image
	struct Texture {
		int32_t imageIndex;
	};

	/*
		Model data
	*/
	std::vector<Image> images;
	std::vector<Texture> textures;
	std::vector<Material> materials;
	std::vector<Node*> nodes;

	~VulkanglTFModel()
	{
		for (auto node : nodes) {
			delete node;
		}
		// Release all Vulkan resources allocated for the model
		vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
		vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
		for (Image image : images) {
			vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view, nullptr);
			vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
			vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler, nullptr);
			vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory, nullptr);
		}
	}

	/*
		glTF loading functions

		The following functions take a glTF input model loaded via tinyglTF and convert all required data into our own structure
	*/

	void loadImages(tinygltf::Model& input)
	{
		// Images can be stored inside the glTF (which is the case for the sample model), so instead of directly
		// loading them from disk, we fetch them from the glTF loader and upload the buffers
		// The last image is reserved for blank texture, which is used in case a texture is missing
		images.resize(input.images.size() + 2);
		for (size_t i = 0; i < input.images.size(); i++) {
			tinygltf::Image& glTFImage = input.images[i];
			// Get the image data from the glTF loader
			unsigned char* buffer = nullptr;
			VkDeviceSize bufferSize = 0;
			bool deleteBuffer = false;
			// We convert RGB-only images to RGBA, as most devices don't support RGB-formats in Vulkan
			if (glTFImage.component == 3) {
				bufferSize = glTFImage.width * glTFImage.height * 4;
				buffer = new unsigned char[bufferSize];
				unsigned char* rgba = buffer;
				unsigned char* rgb = &glTFImage.image[0];
				for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i) {
					memcpy(rgba, rgb, sizeof(unsigned char) * 3);
					rgba += 4;
					rgb += 3;
				}
				deleteBuffer = true;
			}
			else {
				buffer = &glTFImage.image[0];
				bufferSize = glTFImage.image.size();
			}
			// Load texture from image buffer
			images[i].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, glTFImage.width, glTFImage.height, vulkanDevice, copyQueue);
			if (deleteBuffer) {
				delete[] buffer;
			}
		}
		// Add a white and black texture for empty texture in model
		{
			unsigned char* buffer = new unsigned char[4]{ 255, 255, 255, 255 };
			VkDeviceSize bufferSize = 4 * sizeof(unsigned char);
			images[images.size() - 2].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, vulkanDevice, copyQueue);
		}
		{
			unsigned char* buffer = new unsigned char[4] { 0, 0, 0, 255 };
			VkDeviceSize bufferSize = 4 * sizeof(unsigned char);
			images[images.size() - 1].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, vulkanDevice, copyQueue);
		}
	}

	void loadTextures(tinygltf::Model& input)
	{
		textures.resize(input.textures.size());
		for (size_t i = 0; i < input.textures.size(); i++) {
			textures[i].imageIndex = input.textures[i].source;
		}
	}

	void loadMaterials(tinygltf::Model& input)
	{
		materials.resize(input.materials.size());
		for (size_t i = 0; i < input.materials.size(); i++) {
			// We only read the most basic properties required for our sample
			tinygltf::Material glTFMaterial = input.materials[i];
			// Get the base color factor
			if (glTFMaterial.values.find("baseColorFactor") != glTFMaterial.values.end()) {
				materials[i].baseColorFactor = glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
			}
			// Get base color texture index
			if (glTFMaterial.values.find("baseColorTexture") != glTFMaterial.values.end()) {
				materials[i].baseColorTextureIndex = glTFMaterial.values["baseColorTexture"].TextureIndex();
			}
			// Get metallic roughness texture index
			//  - Here we assume that all textures are using TEXCOORD_0
			if (glTFMaterial.values.find("metallicRoughnessTexture") != glTFMaterial.values.end()) {
				materials[i].metallicRoughnessTextureIndex = glTFMaterial.values["metallicRoughnessTexture"].TextureIndex();
			}
			// Get the roughness factor
			if (glTFMaterial.values.find("roughnessFactor") != glTFMaterial.values.end()) {
				materials[i].roughnessFactor = static_cast<float>(glTFMaterial.values["roughnessFactor"].Factor());
			}
			// Get the metallic factor
			if (glTFMaterial.values.find("metallicFactor") != glTFMaterial.values.end()) {
				materials[i].metallicFactor = static_cast<float>(glTFMaterial.values["metallicFactor"].Factor());
			}
			// Get normal texture index
			if (glTFMaterial.additionalValues.find("normalTexture") != glTFMaterial.additionalValues.end()) {
				materials[i].normalTextureIndex = glTFMaterial.additionalValues["normalTexture"].TextureIndex();
			}
			// Get emissive texture index
			if (glTFMaterial.additionalValues.find("emissiveTexture") != glTFMaterial.additionalValues.end()) {
				materials[i].emissiveTextureIndex = glTFMaterial.additionalValues["emissiveTexture"].TextureIndex();
			}
			// Get occlusion texture index
			if (glTFMaterial.additionalValues.find("occlusionTexture") != glTFMaterial.additionalValues.end()) {
				materials[i].occlusionTextureIndex = glTFMaterial.additionalValues["occlusionTexture"].TextureIndex();
			}
		}
	}

	void loadNode(const tinygltf::Node& inputNode, const tinygltf::Model& input, VulkanglTFModel::Node* parent, std::vector<uint32_t>& indexBuffer, std::vector<VulkanglTFModel::Vertex>& vertexBuffer)
	{
		VulkanglTFModel::Node* node = new VulkanglTFModel::Node{};
		node->matrix = glm::mat4(1.0f);
		node->parent = parent;

		// Get the local node matrix
		// It's either made up from translation, rotation, scale or a 4x4 matrix
		if (inputNode.translation.size() == 3) {
			node->matrix = glm::translate(node->matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
		}
		if (inputNode.rotation.size() == 4) {
			glm::quat q = glm::make_quat(inputNode.rotation.data());
			node->matrix *= glm::mat4(q);
		}
		if (inputNode.scale.size() == 3) {
			node->matrix = glm::scale(node->matrix, glm::vec3(glm::make_vec3(inputNode.scale.data())));
		}
		if (inputNode.matrix.size() == 16) {
			node->matrix = glm::make_mat4x4(inputNode.matrix.data());
		};

		// Load node's children
		if (inputNode.children.size() > 0) {
			for (size_t i = 0; i < inputNode.children.size(); i++) {
				loadNode(input.nodes[inputNode.children[i]], input , node, indexBuffer, vertexBuffer);
			}
		}

		// If the node contains mesh data, we load vertices and indices from the buffers
		// In glTF this is done via accessors and buffer views
		if (inputNode.mesh > -1) {
			const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
			// Iterate through all primitives of this node's mesh
			for (size_t i = 0; i < mesh.primitives.size(); i++) {
				const tinygltf::Primitive& glTFPrimitive = mesh.primitives[i];
				uint32_t firstIndex = static_cast<uint32_t>(indexBuffer.size());
				uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
				uint32_t indexCount = 0;
				// Vertices
				{
					const float* positionBuffer = nullptr;
					const float* normalsBuffer = nullptr;
					const float* texCoordsBuffer = nullptr;
					size_t vertexCount = 0;

					// Get buffer data for vertex positions
					if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						positionBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
						vertexCount = accessor.count;
					}
					// Get buffer data for vertex normals
					if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						normalsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}
					// Get buffer data for vertex texture coordinates
					// glTF supports multiple sets, we only load the first one
					if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						texCoordsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}

					// Append data to model's vertex buffer
					for (size_t v = 0; v < vertexCount; v++) {
						Vertex vert{};
						vert.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
						vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
						vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
						vert.color = glm::vec3(1.0f);
						vertexBuffer.push_back(vert);
					}
				}
				// Indices
				{
					const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.indices];
					const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
					const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];

					indexCount += static_cast<uint32_t>(accessor.count);

					// glTF supports different component types of indices
					switch (accessor.componentType) {
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
						const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
						const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
						const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					default:
						std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
						return;
					}
				}
				Primitive primitive{};
				primitive.firstIndex = firstIndex;
				primitive.indexCount = indexCount;
				primitive.materialIndex = glTFPrimitive.material;
				node->mesh.primitives.push_back(primitive);
			}
		}

		if (parent) {
			parent->children.push_back(node);
		}
		else {
			nodes.push_back(node);
		}
	}

	/*
		glTF rendering functions
	*/

	// Draw a single node including child nodes (if present)
	void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node* node)
	{
		if (node->mesh.primitives.size() > 0) {
			// Pass the node's matrix via push constants
			// Traverse the node hierarchy to the top-most parent to get the final matrix of the current node
			glm::mat4 nodeMatrix = node->matrix;
			VulkanglTFModel::Node* currentParent = node->parent;
			while (currentParent) {
				nodeMatrix = currentParent->matrix * nodeMatrix;
				currentParent = currentParent->parent;
			}

			for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives) {
				if (primitive.indexCount > 0) {
					PushConstant push{};
					push.model = nodeMatrix;
					// Pass material properties via push constants
					push.baseColorFactor = materials[primitive.materialIndex].baseColorFactor;
					push.roughnessFactor = materials[primitive.materialIndex].roughnessFactor;
					push.metallicFactor = materials[primitive.materialIndex].metallicFactor;
					// Pass the final matrix to the vertex shader using push constants
					vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant), &push);

					auto const& mat = materials[primitive.materialIndex];
					// Bind the descriptor for the current primitive's texture
					vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &materials[primitive.materialIndex].descriptorSet, 0, nullptr);
					vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
				}
			}
		}
		for (auto& child : node->children) {
			drawNode(commandBuffer, pipelineLayout, child);
		}
	}

	// Draw the glTF scene starting at the top-level-nodes
	void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout)
	{
		// All vertices and indices are stored in single buffers, so we only need to bind once
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		// Render all nodes at top-level
		for (auto& node : nodes) {
			drawNode(commandBuffer, pipelineLayout, node);
		}
	}

};

class VulkanExample : public VulkanExampleBase
{
public:
	bool wireframe = false;

	VulkanglTFModel glTFModel;

	struct ShaderData {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 model;
			// Change point light to directional light to simplify shadow mapping =)
			glm::vec4 lightDir = glm::vec4(1.0f, 1.0f, -1.0f, 1.0f);
			glm::vec4 viewPos;
			glm::mat4 lightMatrix;
			float frustumSize;
			float shadowMapSize;
		} values;
	} shaderData;

	struct Pipelines {
		VkPipeline solid;
		VkPipeline wireframe = VK_NULL_HANDLE;
		VkPipeline shadow = VK_NULL_HANDLE;
		VkPipeline filter = VK_NULL_HANDLE;
		VkPipeline postprocessing = VK_NULL_HANDLE;
	} pipelines;

	struct PipelineLayouts {
		VkPipelineLayout main;
		VkPipelineLayout filter;
		VkPipelineLayout postprocessing;
	} pipelineLayouts;

	VkDescriptorSet descriptorSet;

	struct DescriptorSetLayouts {
		VkDescriptorSetLayout matrices;
		VkDescriptorSetLayout textures;
		VkDescriptorSetLayout shadow;
		VkDescriptorSetLayout postprocessing;
	} descriptorSetLayouts;

	// Image resources
	struct VulkanImage {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
		VkSampler sampler;
	};
	VulkanImage shadowMapColorImage;
	VkDescriptorSet shadowMapColorDescriptorSet;
	VulkanImage shadowMapDepthImage;
	VulkanImage shadowMapFilteredImage;
	VkDescriptorSet shadowMapFilteredDescriptorSet;
	VulkanImage renderTargetColorImage;
	VulkanImage renderTargetDepthImage;
	VkDescriptorSet renderTargetDescriptorSet;

	// Use shadowRenderPass for rendering shadow
	// Use mainRenderPass for shading
	// Use renderPass for postprocessing
	VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
	VkRenderPass filterRenderPass = VK_NULL_HANDLE;
	VkRenderPass mainRenderPass = VK_NULL_HANDLE;
	VkFramebuffer shadowFrameBuffer = VK_NULL_HANDLE;
	VkFramebuffer filterFrameBuffer = VK_NULL_HANDLE;
	VkFramebuffer mainFrameBuffer = VK_NULL_HANDLE;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "homework1";
		camera.type = Camera::CameraType::lookat;
		camera.flipY = true;
		camera.setPosition(glm::vec3(0.0f, -0.1f, -1.0f));
		camera.setRotation(glm::vec3(0.0f, 45.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources
		// Note : Inherited destructor cleans up resources stored in base class
		vkDestroyPipeline(device, pipelines.solid, nullptr);
		if (pipelines.wireframe != VK_NULL_HANDLE) {
			vkDestroyPipeline(device, pipelines.wireframe, nullptr);
		}
		vkDestroyPipeline(device, pipelines.shadow, nullptr);
		vkDestroyPipeline(device, pipelines.filter, nullptr);
		vkDestroyPipeline(device, pipelines.postprocessing, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayouts.main, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.filter, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.postprocessing, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.textures, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.shadow, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.postprocessing, nullptr);

		vkDestroyFramebuffer(device, shadowFrameBuffer, nullptr);
		vkDestroyFramebuffer(device, filterFrameBuffer, nullptr);
		vkDestroyFramebuffer(device, mainFrameBuffer, nullptr);
		vkDestroyRenderPass(device, shadowRenderPass, nullptr);
		vkDestroyRenderPass(device, filterRenderPass, nullptr);
		vkDestroyRenderPass(device, mainRenderPass, nullptr);

		vkDestroyImageView(device, shadowMapColorImage.view, nullptr);
		vkDestroyImage(device, shadowMapColorImage.image, nullptr);
		vkDestroySampler(device, shadowMapColorImage.sampler, nullptr);
		vkFreeMemory(device, shadowMapColorImage.mem, nullptr);
		vkDestroyImageView(device, shadowMapFilteredImage.view, nullptr);
		vkDestroyImage(device, shadowMapFilteredImage.image, nullptr);
		vkDestroySampler(device, shadowMapFilteredImage.sampler, nullptr);
		vkFreeMemory(device, shadowMapFilteredImage.mem, nullptr);
		vkDestroyImageView(device, shadowMapDepthImage.view, nullptr);
		vkDestroyImage(device, shadowMapDepthImage.image, nullptr);
		// No sampler is created for shadowMapDepthImage
		vkFreeMemory(device, shadowMapDepthImage.mem, nullptr);
		vkDestroyImageView(device, renderTargetColorImage.view, nullptr);
		vkDestroyImage(device, renderTargetColorImage.image, nullptr);
		vkDestroySampler(device, renderTargetColorImage.sampler, nullptr);
		vkFreeMemory(device, renderTargetColorImage.mem, nullptr);
		vkDestroyImageView(device, renderTargetDepthImage.view, nullptr);
		vkDestroyImage(device, renderTargetDepthImage.image, nullptr);
		// No sampler is created for renderTargetDepthImage
		vkFreeMemory(device, renderTargetDepthImage.mem, nullptr);

		shaderData.buffer.destroy();
	}

	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// RenderPass #1: Shadow
			{
				const VkViewport viewport = vks::initializers::viewport((float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0.0f, 1.0f);
				const VkRect2D scissor = vks::initializers::rect2D(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, 0);

				std::array<VkClearValue, 2> clearValues = {
					VkClearValue{ { 1.0f, 1.0f, 1.0f, 1.0f } },
					VkClearValue{ 1.0f, 0 },
				};

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = shadowRenderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = SHADOW_MAP_SIZE;
				renderPassBeginInfo.renderArea.extent.height = SHADOW_MAP_SIZE;
				renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassBeginInfo.pClearValues = clearValues.data();
				renderPassBeginInfo.framebuffer = shadowFrameBuffer;
				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
				// Bind scene matrices descriptor to set 0
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.main, 0, 1, &descriptorSet, 0, nullptr);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.shadow);
				glTFModel.draw(drawCmdBuffers[i], pipelineLayouts.main);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}
			// RenderPass #2: Filter
			{
				const VkViewport viewport = vks::initializers::viewport((float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0.0f, 1.0f);
				const VkRect2D scissor = vks::initializers::rect2D(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, 0);

				std::array<VkClearValue, 1> clearValues = {
					VkClearValue{ { 0.0f, 0.0f, 0.0f, 1.0f } },
				};

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = filterRenderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = SHADOW_MAP_SIZE;
				renderPassBeginInfo.renderArea.extent.height = SHADOW_MAP_SIZE;
				renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassBeginInfo.pClearValues = clearValues.data();
				renderPassBeginInfo.framebuffer = filterFrameBuffer;
				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
				// Bind scene matrices descriptor to set 0
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.filter, 0, 1, &shadowMapColorDescriptorSet, 0, nullptr);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.filter);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}
			// RenderPass #3: Main
			{
				const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
				const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

				std::array<VkClearValue, 2> clearValues = {
					VkClearValue{ { 0.25f, 0.25f, 0.25f, 1.0f } },
					VkClearValue{ 1.0f, 0 },
				};

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = mainRenderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassBeginInfo.pClearValues = clearValues.data();
				renderPassBeginInfo.framebuffer = mainFrameBuffer;
				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
				// Bind scene matrices descriptor to set 0
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.main, 0, 1, &descriptorSet, 0, nullptr);
				// Bind shadow map descriptor to set 2
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.main, 2, 1, &shadowMapFilteredDescriptorSet, 0, nullptr);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);
				glTFModel.draw(drawCmdBuffers[i], pipelineLayouts.main);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}
			// RenderPass #4: Postprocessing
			{
				const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
				const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

				std::array<VkClearValue, 2> clearValues = {
					VkClearValue{ { 0.0f, 0.0f, 0.0f, 1.0f } },
					VkClearValue{ 1.0f, 0 },
				};

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = renderPass;
				renderPassBeginInfo.renderArea.offset.x = 0;
				renderPassBeginInfo.renderArea.offset.y = 0;
				renderPassBeginInfo.renderArea.extent.width = width;
				renderPassBeginInfo.renderArea.extent.height = height;
				renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassBeginInfo.pClearValues = clearValues.data();
				renderPassBeginInfo.framebuffer = frameBuffers[i];
				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
				// Bind render target descriptor to set 0
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.postprocessing, 0, 1, &renderTargetDescriptorSet, 0, nullptr);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.postprocessing);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				drawUI(drawCmdBuffers[i]);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadglTFFile(std::string filename)
	{
		tinygltf::Model glTFInput;
		tinygltf::TinyGLTF gltfContext;
		std::string error, warning;

		this->device = device;

#if defined(__ANDROID__)
		// On Android all assets are packed with the apk in a compressed form, so we need to open them using the asset manager
		// We let tinygltf handle this, by passing the asset manager of our app
		tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
		bool fileLoaded = gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

		// Pass some Vulkan resources required for setup and rendering to the glTF model loading class
		glTFModel.vulkanDevice = vulkanDevice;
		glTFModel.copyQueue = queue;

		std::vector<uint32_t> indexBuffer;
		std::vector<VulkanglTFModel::Vertex> vertexBuffer;

		if (fileLoaded) {
			glTFModel.loadImages(glTFInput);
			glTFModel.loadMaterials(glTFInput);
			glTFModel.loadTextures(glTFInput);
			const tinygltf::Scene& scene = glTFInput.scenes[0];
			for (size_t i = 0; i < scene.nodes.size(); i++) {
				const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
				glTFModel.loadNode(node, glTFInput, nullptr, indexBuffer, vertexBuffer);
			}
		}
		else {
			vks::tools::exitFatal("Could not open the glTF file.\n\nThe file is part of the additional asset pack.\n\nRun \"download_assets.py\" in the repository root to download the latest version.", -1);
			return;
		}

		// Create and upload vertex and index buffer
		// We will be using one single vertex buffer and one single index buffer for the whole glTF scene
		// Primitives (of the glTF model) will then index into these using index offsets

		size_t vertexBufferSize = vertexBuffer.size() * sizeof(VulkanglTFModel::Vertex);
		size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
		glTFModel.indices.count = static_cast<uint32_t>(indexBuffer.size());

		struct StagingBuffer {
			VkBuffer buffer;
			VkDeviceMemory memory;
		} vertexStaging, indexStaging;

		// Create host visible staging buffers (source)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			vertexBuffer.data()));
		// Index data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			indexBufferSize,
			&indexStaging.buffer,
			&indexStaging.memory,
			indexBuffer.data()));

		// Create device local buffers (target)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBufferSize,
			&glTFModel.vertices.buffer,
			&glTFModel.vertices.memory));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBufferSize,
			&glTFModel.indices.buffer,
			&glTFModel.indices.memory));

		// Copy data from staging buffers (host) do device local buffer (gpu)
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			vertexStaging.buffer,
			glTFModel.vertices.buffer,
			1,
			&copyRegion);

		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			indexStaging.buffer,
			glTFModel.indices.buffer,
			1,
			&copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		// Free staging resources
		vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
		vkFreeMemory(device, vertexStaging.memory, nullptr);
		vkDestroyBuffer(device, indexStaging.buffer, nullptr);
		vkFreeMemory(device, indexStaging.memory, nullptr);
	}

	void loadAssets()
	{
		loadglTFFile(getAssetPath() + "buster_drone/busterDrone.gltf");
	}

	void setupDescriptors()
	{
		/*
			This sample uses separate descriptor sets (and layouts) for the matrices and materials (textures)
		*/

		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
			// One combined image sampler per model image/texture + two for shadow map + one for filtered map + tow for postprocessing render target
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(glTFModel.images.size()) + 5),
		};
		// One set for matrices and one per model image/texture
		// Use Only three sets (personal favor), one for scene parameters and one for material parameters and one for textures
		// const uint32_t maxSetCount = static_cast<uint32_t>(glTFModel.images.size()) + 1;
		const uint32_t maxSetCount = static_cast<uint32_t>(glTFModel.materials.size()) + 4;
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Descriptor set layout for passing matrices
		VkDescriptorSetLayoutBinding setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));
		// Descriptor set layout for passing material textures
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures));
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.shadow));
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.postprocessing));

		// Pipeline layout using both descriptor sets (set 0 = matrices, set 1 = material, set 2 = shadow)
		std::array<VkDescriptorSetLayout, 3> setLayouts = { descriptorSetLayouts.matrices, descriptorSetLayouts.textures, descriptorSetLayouts.shadow };
		VkPipelineLayoutCreateInfo pipelineLayoutCI= vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
		// We will use push constants to push the local matrices of a primitive to the vertex shader
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(VulkanglTFModel::PushConstant), 0);
		// Push constant ranges are part of the pipeline layout
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayouts.main));

		pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.postprocessing, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayouts.postprocessing));

		pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.shadow, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayouts.filter));

		// Descriptor set for scene matrices
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		// Descriptor set for materials
		for (auto& mat : glTFModel.materials) {
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.textures, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &mat.descriptorSet));
			auto getImageDescriptor = [&](uint32_t index, uint32_t defaultTexture) {
				return (index == TEXTURE_INDEX_EMPTY) ? glTFModel.images[glTFModel.images.size() - defaultTexture].texture.descriptor : glTFModel.images[glTFModel.textures[index].imageIndex].texture.descriptor;
			};
			std::vector<VkDescriptorImageInfo> imageDescriptors = {
				getImageDescriptor(mat.baseColorTextureIndex, TEXTURE_DEFAULT_BLACK),
				getImageDescriptor(mat.normalTextureIndex, TEXTURE_DEFAULT_BLACK),
				getImageDescriptor(mat.metallicRoughnessTextureIndex, TEXTURE_DEFAULT_BLACK),
				getImageDescriptor(mat.emissiveTextureIndex, TEXTURE_DEFAULT_BLACK),
				getImageDescriptor(mat.occlusionTextureIndex, TEXTURE_DEFAULT_WHITE),
			};
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(mat.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0,
				imageDescriptors.data(), static_cast<uint32_t>(imageDescriptors.size()));
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
	}

	void prepareRenderPass()
	{
		{
			std::array<VkAttachmentDescription, 2> attachments = {};
			// Color attachment
			attachments[0].format = SHADOW_MAP_COLOR_FORMAT;
			attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			// Depth attachment
			attachments[1].format = depthFormat;
			attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

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
			subpassDescription.pDepthStencilAttachment = &depthReference;
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

			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &shadowRenderPass));
		}
		{
			std::array<VkAttachmentDescription, 1> attachments = {};
			// Color attachment
			attachments[0].format = SHADOW_MAP_COLOR_FORMAT;
			attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkAttachmentReference colorReference = {};
			colorReference.attachment = 0;
			colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpassDescription = {};
			subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpassDescription.colorAttachmentCount = 1;
			subpassDescription.pColorAttachments = &colorReference;
			subpassDescription.pDepthStencilAttachment = nullptr;
			subpassDescription.inputAttachmentCount = 0;
			subpassDescription.pInputAttachments = nullptr;
			subpassDescription.preserveAttachmentCount = 0;
			subpassDescription.pPreserveAttachments = nullptr;
			subpassDescription.pResolveAttachments = nullptr;

			// Subpass dependencies for layout transitions
			std::array<VkSubpassDependency, 1> dependencies;

			dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[0].dstSubpass = 0;
			dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
			dependencies[0].dependencyFlags = 0;

			VkRenderPassCreateInfo renderPassCI = {};
			renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			renderPassCI.pAttachments = attachments.data();
			renderPassCI.subpassCount = 1;
			renderPassCI.pSubpasses = &subpassDescription;
			renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
			renderPassCI.pDependencies = dependencies.data();

			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &filterRenderPass));
		}
		{
			std::array<VkAttachmentDescription, 2> attachments = {};
			// Color attachment
			attachments[0].format = RENDER_TARGET_COLOR_FORMAT;
			attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			// Depth attachment
			attachments[1].format = depthFormat;
			attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

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
			subpassDescription.pDepthStencilAttachment = &depthReference;
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

			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &mainRenderPass));
		}
	}

	void prepareFrameBuffer()
	{
		{
			std::array<VkImageView, 2> attachments = {
				shadowMapColorImage.view,
				shadowMapDepthImage.view,
			};

			VkFramebufferCreateInfo frameBufferCI = {};
			frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frameBufferCI.pNext = NULL;
			frameBufferCI.renderPass = shadowRenderPass;
			frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCI.pAttachments = attachments.data();
			frameBufferCI.width = SHADOW_MAP_SIZE;
			frameBufferCI.height = SHADOW_MAP_SIZE;
			frameBufferCI.layers = 1;

			VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &shadowFrameBuffer));
		}
		{
			std::array<VkImageView, 1> attachments = {
				shadowMapFilteredImage.view,
			};

			VkFramebufferCreateInfo frameBufferCI = {};
			frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frameBufferCI.pNext = NULL;
			frameBufferCI.renderPass = filterRenderPass;
			frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCI.pAttachments = attachments.data();
			frameBufferCI.width = SHADOW_MAP_SIZE;
			frameBufferCI.height = SHADOW_MAP_SIZE;
			frameBufferCI.layers = 1;

			VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &filterFrameBuffer));
		}
		{
			std::array<VkImageView, 2> attachments = {
				renderTargetColorImage.view,
				renderTargetDepthImage.view,
			};

			VkFramebufferCreateInfo frameBufferCI = {};
			frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frameBufferCI.pNext = NULL;
			frameBufferCI.renderPass = mainRenderPass;
			frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCI.pAttachments = attachments.data();
			frameBufferCI.width = width;
			frameBufferCI.height = height;
			frameBufferCI.layers = 1;

			VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &mainFrameBuffer));
		}
	}

	void prepareLightMatrix()
	{
		auto lightDir = glm::normalize(glm::vec3(shaderData.values.lightDir));

		auto proj = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.01f, 10.0f);
		auto view = glm::lookAt(lightDir * 2.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		shaderData.values.frustumSize = 20.0f;
		shaderData.values.shadowMapSize = SHADOW_MAP_SIZE;
		shaderData.values.lightMatrix = proj * view;
	}

	void prepareMainPipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		// Enable color blend
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_TRUE);
		blendAttachmentStateCI.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentStateCI.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendAttachmentStateCI.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentStateCI.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentStateCI.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendAttachmentStateCI.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		// Vertex input bindings and attributes
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)),	// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)),// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)),	// Location 2: Texture coordinates
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)),	// Location 3: Color
		};
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputStateCI.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getHomeworkShadersPath() + "homework1/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getHomeworkShadersPath() + "homework1/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts.main, mainRenderPass, 0);
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Solid rendering pipeline
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.solid));

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid) {
			rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
			rasterizationStateCI.lineWidth = 1.0f;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
		}

		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
		blendAttachmentStateCI.blendEnable = VK_FALSE;
		shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework1/depth.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getHomeworkShadersPath() + "homework1/depth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.renderPass = shadowRenderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.shadow));
	}

	void preparePostprocessingPipeline()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getHomeworkShadersPath() + "homework1/quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getHomeworkShadersPath() + "homework1/postprocessing.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts.postprocessing, renderPass, 0);
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.postprocessing));
	}

	void prepareFilterPipeline()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getHomeworkShadersPath() + "homework1/quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getHomeworkShadersPath() + "homework1/filter.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts.filter, filterRenderPass, 0);
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.filter));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&shaderData.buffer,
			sizeof(shaderData.values)));

		// Map persistent
		VK_CHECK_RESULT(shaderData.buffer.map());

		updateUniformBuffers();
	}

	void prepareResources() {
		{
			VkFormat format = SHADOW_MAP_COLOR_FORMAT;

			VkImageCreateInfo imageCI{};
			imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageCI.imageType = VK_IMAGE_TYPE_2D;
			imageCI.format = format;
			imageCI.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
			imageCI.mipLevels = 1;
			imageCI.arrayLayers = 1;
			imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

			VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &shadowMapColorImage.image));
			VkMemoryRequirements memReqs{};
			vkGetImageMemoryRequirements(device, shadowMapColorImage.image, &memReqs);

			VkMemoryAllocateInfo memAllloc{};
			memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			memAllloc.allocationSize = memReqs.size;
			memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &shadowMapColorImage.mem));
			VK_CHECK_RESULT(vkBindImageMemory(device, shadowMapColorImage.image, shadowMapColorImage.mem, 0));

			VkImageViewCreateInfo imageViewCI{};
			imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCI.image = shadowMapColorImage.image;
			imageViewCI.format = format;
			imageViewCI.subresourceRange.baseMipLevel = 0;
			imageViewCI.subresourceRange.levelCount = 1;
			imageViewCI.subresourceRange.baseArrayLayer = 0;
			imageViewCI.subresourceRange.layerCount = 1;
			imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &shadowMapColorImage.view));

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
			VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &shadowMapColorImage.sampler));

			VkDescriptorImageInfo descriptorImageInfo{};
			descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			descriptorImageInfo.imageView = shadowMapColorImage.view;
			descriptorImageInfo.sampler = shadowMapColorImage.sampler;

			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.shadow, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &shadowMapColorDescriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(shadowMapColorDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &descriptorImageInfo);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
		{
			VkFormat format = depthFormat;

			VkImageCreateInfo imageCI{};
			imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageCI.imageType = VK_IMAGE_TYPE_2D;
			imageCI.format = format;
			imageCI.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
			imageCI.mipLevels = 1;
			imageCI.arrayLayers = 1;
			imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &shadowMapDepthImage.image));
			VkMemoryRequirements memReqs{};
			vkGetImageMemoryRequirements(device, shadowMapDepthImage.image, &memReqs);

			VkMemoryAllocateInfo memAllloc{};
			memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			memAllloc.allocationSize = memReqs.size;
			memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &shadowMapDepthImage.mem));
			VK_CHECK_RESULT(vkBindImageMemory(device, shadowMapDepthImage.image, shadowMapDepthImage.mem, 0));

			VkImageViewCreateInfo imageViewCI{};
			imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCI.image = shadowMapDepthImage.image;
			imageViewCI.format = format;
			imageViewCI.subresourceRange.baseMipLevel = 0;
			imageViewCI.subresourceRange.levelCount = 1;
			imageViewCI.subresourceRange.baseArrayLayer = 0;
			imageViewCI.subresourceRange.layerCount = 1;
			imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
			VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &shadowMapDepthImage.view));
		}
		{
			VkFormat format = SHADOW_MAP_COLOR_FORMAT;

			VkImageCreateInfo imageCI{};
			imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageCI.imageType = VK_IMAGE_TYPE_2D;
			imageCI.format = format;
			imageCI.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
			imageCI.mipLevels = 1;
			imageCI.arrayLayers = 1;
			imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

			VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &shadowMapFilteredImage.image));
			VkMemoryRequirements memReqs{};
			vkGetImageMemoryRequirements(device, shadowMapFilteredImage.image, &memReqs);

			VkMemoryAllocateInfo memAllloc{};
			memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			memAllloc.allocationSize = memReqs.size;
			memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &shadowMapFilteredImage.mem));
			VK_CHECK_RESULT(vkBindImageMemory(device, shadowMapFilteredImage.image, shadowMapFilteredImage.mem, 0));

			VkImageViewCreateInfo imageViewCI{};
			imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCI.image = shadowMapFilteredImage.image;
			imageViewCI.format = format;
			imageViewCI.subresourceRange.baseMipLevel = 0;
			imageViewCI.subresourceRange.levelCount = 1;
			imageViewCI.subresourceRange.baseArrayLayer = 0;
			imageViewCI.subresourceRange.layerCount = 1;
			imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &shadowMapFilteredImage.view));

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
			VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &shadowMapFilteredImage.sampler));

			VkDescriptorImageInfo descriptorImageInfo{};
			descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			descriptorImageInfo.imageView = shadowMapFilteredImage.view;
			descriptorImageInfo.sampler = shadowMapFilteredImage.sampler;

			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.shadow, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &shadowMapFilteredDescriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(shadowMapFilteredDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &descriptorImageInfo);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
		{
			VkFormat format = RENDER_TARGET_COLOR_FORMAT;

			VkImageCreateInfo imageCI{};
			imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageCI.imageType = VK_IMAGE_TYPE_2D;
			imageCI.format = format;
			imageCI.extent = { width, height, 1 };
			imageCI.mipLevels = 1;
			imageCI.arrayLayers = 1;
			imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

			VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &renderTargetColorImage.image));
			VkMemoryRequirements memReqs{};
			vkGetImageMemoryRequirements(device, renderTargetColorImage.image, &memReqs);

			VkMemoryAllocateInfo memAllloc{};
			memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			memAllloc.allocationSize = memReqs.size;
			memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &renderTargetColorImage.mem));
			VK_CHECK_RESULT(vkBindImageMemory(device, renderTargetColorImage.image, renderTargetColorImage.mem, 0));

			VkImageViewCreateInfo imageViewCI{};
			imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCI.image = renderTargetColorImage.image;
			imageViewCI.format = format;
			imageViewCI.subresourceRange.baseMipLevel = 0;
			imageViewCI.subresourceRange.levelCount = 1;
			imageViewCI.subresourceRange.baseArrayLayer = 0;
			imageViewCI.subresourceRange.layerCount = 1;
			imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &renderTargetColorImage.view));

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
			VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &renderTargetColorImage.sampler));

			VkDescriptorImageInfo descriptorImageInfo{};
			descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			descriptorImageInfo.imageView = renderTargetColorImage.view;
			descriptorImageInfo.sampler = renderTargetColorImage.sampler;

			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.postprocessing, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &renderTargetDescriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(renderTargetDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &descriptorImageInfo);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
		{
			VkImageCreateInfo imageCI{};
			imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageCI.imageType = VK_IMAGE_TYPE_2D;
			imageCI.format = depthFormat;
			imageCI.extent = { width, height, 1 };
			imageCI.mipLevels = 1;
			imageCI.arrayLayers = 1;
			imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &renderTargetDepthImage.image));
			VkMemoryRequirements memReqs{};
			vkGetImageMemoryRequirements(device, renderTargetDepthImage.image, &memReqs);

			VkMemoryAllocateInfo memAllloc{};
			memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			memAllloc.allocationSize = memReqs.size;
			memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &renderTargetDepthImage.mem));
			VK_CHECK_RESULT(vkBindImageMemory(device, renderTargetDepthImage.image, renderTargetDepthImage.mem, 0));

			VkImageViewCreateInfo imageViewCI{};
			imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCI.image = renderTargetDepthImage.image;
			imageViewCI.format = depthFormat;
			imageViewCI.subresourceRange.baseMipLevel = 0;
			imageViewCI.subresourceRange.levelCount = 1;
			imageViewCI.subresourceRange.baseArrayLayer = 0;
			imageViewCI.subresourceRange.layerCount = 1;
			imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
			VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &renderTargetDepthImage.view));
		}
	}

	void updateUniformBuffers()
	{
		shaderData.values.projection = camera.matrices.perspective;
		shaderData.values.model = camera.matrices.view;
		shaderData.values.viewPos = camera.viewPos;
		memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareLightMatrix();
		prepareUniformBuffers();
		setupDescriptors();
		prepareResources();
		prepareRenderPass();
		prepareFrameBuffer();
		prepareMainPipelines();
		preparePostprocessingPipeline();
		prepareFilterPipeline();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		renderFrame();
		if (camera.updated) {
			updateUniformBuffers();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->checkBox("Wireframe", &wireframe)) {
				buildCommandBuffers();
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
