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

/*
 * Games106 Homework 1
 * - By Ziyi.Lu
 * 
 * I implemented the following features:
 *	1. Skeleton animation
 *	2. PBR material
 *		including: normal/roughness/metallic/emission/occlusion map
 *	3. Postprocessing
 *		includeing: tone mapping, vignette, grain and chromatic aberration
 *	4. Variance shadow map (for directional light)
 * 
 * It took four render passes to composite the final image:
 * 
 *	Pass #1: Shadow Pass: (Render variance shadow map)
 *	Pass #2: Filter Pass: (Filter variance shadow map)
 *	Pass #3: Main Pass: (Render model with lighting and shadow)
 *	Pass #4: Postprocessing Pass: (Postprocessing and present)
 * 
 * To change the loaded model, goto [line 1204]
 */

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif
#include "tiny_gltf.h"

#include "vulkanexamplebase.h"

#define ENABLE_VALIDATION true

#include "utility.h"
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/matrix_decompose.hpp>

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

#define MAX_NUM_JOINTS 32

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
		glm::vec4 joint;
		glm::vec4 weight;
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
		struct ShaderData {
			vks::Buffer buffer;
			struct Values {
				glm::mat4 matrix;
				glm::mat4 matrixInverseTransposed;
				glm::mat4 jointMatrices[MAX_NUM_JOINTS];
				int jointCount;
			} values;
		} shaderData;
		VkDescriptorSet descriptorSet;

		~Mesh() {
			shaderData.buffer.destroy();
		}
	};

	struct Node;

	struct Skin {
		Node* root = nullptr;
		std::vector<glm::mat4> inverseBindMatrices;
		std::vector<Node*> joints;
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

		glm::mat4 worldTransform() const {
			glm::mat4 mat = matrix;
			VulkanglTFModel::Node* p = parent;
			while (p) {
				mat = p->matrix * mat;
				p = p->parent;
			}
			return mat;
		}

		Skin* skin = nullptr;
		int32_t skinIndex = -1;

		void update() {
			glm::mat4 m = worldTransform();
			mesh.shaderData.values.matrix = m;
			mesh.shaderData.values.matrixInverseTransposed = glm::transpose(glm::inverse(m));
			if (skin) {
				glm::mat4 inverseTransform = glm::inverse(m);
				size_t jointCount = std::min(skin->joints.size(), static_cast<size_t>(MAX_NUM_JOINTS));
				for (size_t i = 0; i < jointCount; ++i) {
					Node* jointNode = skin->joints[i];
					glm::mat4 jointTransform = jointNode->worldTransform() * skin->inverseBindMatrices[i];
					jointTransform = inverseTransform * jointTransform;
					mesh.shaderData.values.jointMatrices[i] = jointTransform;
				}
				mesh.shaderData.values.jointCount = static_cast<int>(jointCount);
			}
			else {
				mesh.shaderData.values.jointCount = 0;
			}
			memcpy(mesh.shaderData.buffer.mapped, &mesh.shaderData.values, sizeof(mesh.shaderData.values));

			for (auto& child : children) {
				child->update();
			}
		}
	};

	struct AnimationChannel {
		enum struct PathType { Translation, Rotation, Scale };
		PathType path;
		Node* node;
		uint32_t samplerIndex;
	};

	struct AnimationSampler {
		enum struct InterpolationType { Linear, Step, CubicSpline };
		InterpolationType interpolation;
		std::vector<float> inputs;
		std::vector<glm::vec4> outputs;
	};

	struct Animation {
		std::string name;
		std::vector<AnimationSampler> samplers;
		std::vector<AnimationChannel> channels;
		float start = std::numeric_limits<float>::max();
		float end = std::numeric_limits<float>::min();
	};

	// A glTF material stores information in e.g. the texture that is attached to it and colors
	struct Material {
		uint32_t baseColorTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t normalTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t metallicRoughnessTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t emissiveTextureIndex = TEXTURE_INDEX_EMPTY;
		uint32_t occlusionTextureIndex = TEXTURE_INDEX_EMPTY;
		// We use one descriptor set for textures and parameters of each material
		struct ShaderData {
			vks::Buffer buffer;
			struct Values {
				glm::vec4 baseColorFactor = glm::vec4(1.0f);
				float roughnessFactor = 1.0f;
				float metallicFactor = 1.0f;
			} values;
		} shaderData;
		VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

		~Material() {
			shaderData.buffer.destroy();
		}
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
	std::vector<Node*> linearNodes; // Store all nodes linearly, in the same order as glTF
	std::vector<Animation> animations;
	std::vector<Skin> skins;

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
				materials[i].shaderData.values.baseColorFactor = glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
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
				materials[i].shaderData.values.roughnessFactor = static_cast<float>(glTFMaterial.values["roughnessFactor"].Factor());
			}
			// Get the metallic factor
			if (glTFMaterial.values.find("metallicFactor") != glTFMaterial.values.end()) {
				materials[i].shaderData.values.metallicFactor = static_cast<float>(glTFMaterial.values["metallicFactor"].Factor());
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

			VK_CHECK_RESULT(vulkanDevice->createBuffer(
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				&materials[i].shaderData.buffer,
				sizeof(materials[i].shaderData.values)));

			VK_CHECK_RESULT(materials[i].shaderData.buffer.map());

			memcpy(materials[i].shaderData.buffer.mapped, &materials[i].shaderData.values, sizeof(materials[i].shaderData.values));
		}
	}

	void loadSkins(const tinygltf::Model& input) {
		for (auto const& s : input.skins) {
			Skin skin{};

			if (s.skeleton > -1) {
				skin.root = linearNodes[s.skeleton];
			}

			for (auto jointIndex : s.joints) {
				if (jointIndex >= 0) {
					skin.joints.push_back(linearNodes[jointIndex]);
				}
			}

			if (s.inverseBindMatrices > -1) {
				auto const& accessor = input.accessors[s.inverseBindMatrices];
				auto const& bufferView = input.bufferViews[accessor.bufferView];
				auto const& buffer = input.buffers[bufferView.buffer];
				skin.inverseBindMatrices.resize(accessor.count);
				memcpy(skin.inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(glm::mat4));
			}

			skins.push_back(skin);
		}
	}

	void loadAnimations(const tinygltf::Model& input) {
		for (auto const& anim : input.animations) {
			Animation animation{};
			animation.name = anim.name;
			if (animation.name.empty()) {
				animation.name = (std::string("Animation_") + std::to_string(animations.size()));
			}

			for (auto const& samp : anim.samplers) {
				AnimationSampler sampler{};

				if (samp.interpolation == "LINEAR") sampler.interpolation = AnimationSampler::InterpolationType::Linear;
				if (samp.interpolation == "STEP") sampler.interpolation = AnimationSampler::InterpolationType::Step;
				if (samp.interpolation == "CUBICSPLINE") sampler.interpolation = AnimationSampler::InterpolationType::CubicSpline;

				{
					auto const& accessor = input.accessors[samp.input];
					auto const& bufferView = input.bufferViews[accessor.bufferView];
					auto const& buffer = input.buffers[bufferView.buffer];

					assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && "Animation input component type must be float");

					const void* ptr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
					auto buf = static_cast<const float*>(ptr);

					for (size_t i = 0; i < accessor.count; ++i) {
						sampler.inputs.push_back(buf[i]);
					}

					for (auto const& input : sampler.inputs) {
						if (input < animation.start) animation.start = input;
						if (input > animation.end) animation.end = input;
					}
				}

				{
					auto const& accessor = input.accessors[samp.output];
					auto const& bufferView = input.bufferViews[accessor.bufferView];
					auto const& buffer = input.buffers[bufferView.buffer];

					assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && "Animation output component type must be float");

					const void* ptr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
					switch (accessor.type) {
					case TINYGLTF_TYPE_VEC3: {
						auto buf = static_cast<const glm::vec3*>(ptr);
						for (size_t i = 0; i < accessor.count; ++i) {
							sampler.outputs.push_back(glm::vec4(buf[i], 0.0f));
						}
						break;
					}
					case TINYGLTF_TYPE_VEC4: {
						auto buf = static_cast<const glm::vec4*>(ptr);
						for (size_t i = 0; i < accessor.count; ++i) {
							sampler.outputs.push_back(buf[i]);
						}
						break;
					}
					default:
						std::cerr << "Unknown type for animation output" << std::endl;
						break;
					}
				}

				animation.samplers.push_back(sampler);
			}

			for (auto const& chan : anim.channels) {
				AnimationChannel channel{};

				if (chan.target_path == "rotation") channel.path = AnimationChannel::PathType::Rotation;
				if (chan.target_path == "translation") channel.path = AnimationChannel::PathType::Translation;
				if (chan.target_path == "scale") channel.path = AnimationChannel::PathType::Scale;
				if (chan.target_path == "weights") {
					std::clog << "weights not supported for current AnimationChannel\n";
					continue;
				}

				channel.samplerIndex = chan.sampler;
				channel.node = linearNodes[chan.target_node];
				if (!channel.node) {
					continue;
				}

				animation.channels.push_back(channel);
			}

			animations.push_back(animation);
		}
	}

	void loadNode(const tinygltf::Node& inputNode, uint32_t index, const tinygltf::Model& input, VulkanglTFModel::Node* parent, std::vector<uint32_t>& indexBuffer, std::vector<VulkanglTFModel::Vertex>& vertexBuffer)
	{
		VulkanglTFModel::Node* node = new VulkanglTFModel::Node{};
		node->matrix = glm::mat4(1.0f);
		node->parent = parent;
		node->skinIndex = inputNode.skin;

		// Setup uniform buffer for animation
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&node->mesh.shaderData.buffer,
			sizeof(node->mesh.shaderData.values)));

		VK_CHECK_RESULT(node->mesh.shaderData.buffer.map());

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
		linearNodes.resize(input.nodes.size());
		if (inputNode.children.size() > 0) {
			for (size_t i = 0; i < inputNode.children.size(); i++) {
				loadNode(input.nodes[inputNode.children[i]], inputNode.children[i], input, node, indexBuffer, vertexBuffer);
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
					const void* jointsBuffer = nullptr;
					const float* weightsBuffer = nullptr;
					size_t vertexCount = 0;

					int jointsComponentType;

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

					// Get buffer data for vertex joint indices
					if (glTFPrimitive.attributes.find("JOINTS_0") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("JOINTS_0")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						jointsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
						jointsComponentType = accessor.componentType;
					}
					// Get buffer data for vertex joint weights
					if (glTFPrimitive.attributes.find("WEIGHTS_0") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("WEIGHTS_0")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						weightsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}

					bool hasSkin = jointsBuffer && weightsBuffer;

					// Append data to model's vertex buffer
					for (size_t v = 0; v < vertexCount; v++) {
						Vertex vert{};
						vert.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
						vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
						vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
						vert.color = glm::vec3(1.0f);

						if (hasSkin) {
							switch (jointsComponentType) {
							case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
								const uint16_t* buf = static_cast<const uint16_t*>(jointsBuffer);
								vert.joint = glm::vec4(glm::make_vec4(&buf[v * 4]));
								break;
							}
							case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
								const uint8_t* buf = static_cast<const uint8_t*>(jointsBuffer);
								vert.joint = glm::vec4(glm::make_vec4(&buf[v * 4]));
								break;
							}
							default:
								std::cerr << "Joint component type " << jointsComponentType << " not supported!" << std::endl;
								break;
							}
							vert.weight = glm::make_vec4(&weightsBuffer[v * 4]);
						}
						else {
							vert.joint = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
							vert.weight = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
						}

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

		linearNodes[index] = node;
	}

	/*
		glTF rendering functions
	*/

	void update(uint32_t animationIndex, float time) {
		if (animationIndex >= static_cast<uint32_t>(animations.size())) {
			std::cerr << "No animation with index " << animationIndex << std::endl;
			return;
		}

		auto const& animation = animations[animationIndex];

		for (auto const& channel : animation.channels) {
			auto const& sampler = animation.samplers[channel.samplerIndex];
			if (sampler.inputs.size() > sampler.outputs.size()) {
				continue;
			}

			for (size_t i = 0; i < sampler.inputs.size() - 1; ++i) {
				if (time < sampler.inputs[i] || time > sampler.inputs[i + 1]) continue;
				float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
				if (u <= 1.0f) {
					glm::vec3 scale;
					glm::quat rotation;
					glm::vec3 translation;
					glm::vec3 skew;
					glm::vec4 perspective;
					glm::decompose(channel.node->matrix, scale, rotation, translation, skew, perspective);

					switch (channel.path) {
					case AnimationChannel::PathType::Translation: {
						auto t = glm::lerp(sampler.outputs[i], sampler.outputs[i + 1], u);
						translation = glm::vec3(t);
						break;
					}
					case AnimationChannel::PathType::Scale: {
						auto s = glm::lerp(sampler.outputs[i], sampler.outputs[i + 1], u);
						scale = glm::vec3(s);
						break;
					}
					case AnimationChannel::PathType::Rotation: {
						glm::quat q0 = glm::make_quat(&sampler.outputs[i].x);
						glm::quat q1 = glm::make_quat(&sampler.outputs[i + 1].x);
						rotation = glm::normalize(glm::slerp(q0, q1, u));
						break;
					}
					}

					channel.node->matrix = glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale);
				}
			}
		}

		for (auto& node : nodes) {
			node->update();
		}
	}

	// Draw a single node including child nodes (if present)
	void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node* node)
	{
		if (node->mesh.primitives.size() > 0) {
			// Pass the node's matrix via push constants
			// Traverse the node hierarchy to the top-most parent to get the final matrix of the current node
			glm::mat4 nodeMatrix = node->worldTransform();
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 3, 1, &node->mesh.descriptorSet, 0, nullptr);

			for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives) {
				if (primitive.indexCount > 0) {
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
	bool animationPlay = true;
	int32_t animationIndex = 0;
	float animationTime = 0.0f;
	std::vector<std::string> animationNames;
	hw1::Timer timer;
	float fixedUpdateDelta = 0.016f;
	float fixedUpdateTimer = 0.0f;
	bool enableToneMapping = true;
	bool enableVignette = true;
	bool enableGrain = true;
	bool enableChromaticAberration = true;
	float lightIntensity = 5.0f;
	float ambientIntensity = 1.0f;
	glm::vec3 lightDirection = glm::vec3(0.25f, 1.0f, 0.25f);

	VulkanglTFModel glTFModel;

	struct ShaderData {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 model;
			// Change point light to directional light to simplify shadow mapping =)
			glm::vec4 lightDir;
			glm::vec4 viewPos;
			glm::mat4 lightMatrix;
			float lightIntensity;
			float ambientIntensity;
		} values;
	} shaderData;

	struct PostprocessingData {
		vks::Buffer buffer;
		struct Values {
			int enableToneMapping;
			int enableVignette;
			int enableGrain;
			int enableChromaticAberration;
		} values;
	} postprocessingData;

	struct Pipelines {
		VkPipeline solid = VK_NULL_HANDLE;
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
	VkDescriptorSet postprocessingDataDescriptorSet;

	struct DescriptorSetLayouts {
		VkDescriptorSetLayout matrices;
		VkDescriptorSetLayout textures;
		VkDescriptorSetLayout shadow;
		VkDescriptorSetLayout postprocessing;
		VkDescriptorSetLayout animation;
		VkDescriptorSetLayout postprocessingData;
	} descriptorSetLayouts;

	// Image resources
	hw1::Image2D shadowMapColorImage;
	hw1::Image2D shadowMapDepthImage;
	hw1::Image2D shadowMapFilteredImage;
	hw1::Image2D renderTargetColorImage;
	hw1::Image2D renderTargetDepthImage;
	// Descriptor sets for images to be sampled in shader
	VkDescriptorSet shadowMapColorDescriptorSet = VK_NULL_HANDLE;
	VkDescriptorSet shadowMapFilteredDescriptorSet = VK_NULL_HANDLE;
	VkDescriptorSet renderTargetDescriptorSet = VK_NULL_HANDLE;

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
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.animation, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.postprocessingData, nullptr);

		vkDestroyFramebuffer(device, shadowFrameBuffer, nullptr);
		vkDestroyFramebuffer(device, filterFrameBuffer, nullptr);
		vkDestroyFramebuffer(device, mainFrameBuffer, nullptr);
		vkDestroyRenderPass(device, shadowRenderPass, nullptr);
		vkDestroyRenderPass(device, filterRenderPass, nullptr);
		vkDestroyRenderPass(device, mainRenderPass, nullptr);

		hw1::destroyImage2D(device, &shadowMapColorImage);
		hw1::destroyImage2D(device, &shadowMapDepthImage);
		hw1::destroyImage2D(device, &shadowMapFilteredImage);
		hw1::destroyImage2D(device, &renderTargetColorImage);
		hw1::destroyImage2D(device, &renderTargetDepthImage);

		shaderData.buffer.destroy();
		postprocessingData.buffer.destroy();
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
				// Bind parameter descriptor to set 0
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.postprocessing, 1, 1, &postprocessingDataDescriptorSet, 0, nullptr);
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
				glTFModel.loadNode(node, scene.nodes[i], glTFInput, nullptr, indexBuffer, vertexBuffer);
			}
			glTFModel.loadAnimations(glTFInput);
			glTFModel.loadSkins(glTFInput);
			for (auto& node : glTFModel.linearNodes) {
				if (node->skinIndex > -1) {
					node->skin = &glTFModel.skins[node->skinIndex];
				}
			}
			for (auto& node : glTFModel.nodes) {
				// [TODO] Only need to update nodes with mesh (!mesh.primitives.empty())
				node->update();
			}
			for (auto& anim : glTFModel.animations) {
				animationNames.push_back(anim.name);
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
		//loadglTFFile(getAssetPath() + "models/CesiumMan/glTF/CesiumMan.gltf");
	}

	void setupDescriptors()
	{
		/*
			This sample uses separate descriptor sets (and layouts) for the matrices and materials (textures)
		*/

		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(glTFModel.linearNodes.size()) + static_cast<uint32_t>(glTFModel.materials.size()) + 2),
			// One combined image sampler per model image/texture + two for shadow map + one for filtered map + tow for postprocessing render target
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(glTFModel.images.size()) + 5),
		};
		// One set for matrices + one per model image/texture + one per node animation + one for shadow map 
		//  + one for filtered shadow map + one for postprocessing input + one for postprocessing uniform buffer
		const uint32_t maxSetCount = static_cast<uint32_t>(glTFModel.materials.size()) + static_cast<uint32_t>(glTFModel.linearNodes.size()) + 5;
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
		descriptorPoolInfo.flags |= VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Descriptor set layout for passing matrices
		VkDescriptorSetLayoutBinding setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));
		// Descriptor set layout for passing material textures
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 5),
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures));
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.shadow));
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.postprocessing));
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.animation));
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.postprocessingData));

		// Pipeline layout using both descriptor sets (set 0 = matrices, set 1 = material, set 2 = shadow, set 3 = animation)
		std::vector<VkDescriptorSetLayout> setLayouts = { 
			descriptorSetLayouts.matrices,
			descriptorSetLayouts.textures,
			descriptorSetLayouts.shadow,
			descriptorSetLayouts.animation
		};
		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayouts.main));
		setLayouts = {
			descriptorSetLayouts.postprocessing,
			descriptorSetLayouts.postprocessingData,
		};
		pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayouts.postprocessing));
		pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.shadow, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayouts.filter));

		// Descriptor set for scene matrices
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

		// Descriptor set for postprocessing parameters
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.postprocessingData, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &postprocessingDataDescriptorSet));
		writeDescriptorSet = vks::initializers::writeDescriptorSet(postprocessingDataDescriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &postprocessingData.buffer.descriptor);
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
			writeDescriptorSet = vks::initializers::writeDescriptorSet(mat.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5, &mat.shaderData.buffer.descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}

		// Descriptor sets for animations
		for (auto& node : glTFModel.linearNodes) {
			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.animation, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &node->mesh.descriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(node->mesh.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &node->mesh.shaderData.buffer.descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
	}

	void recreateRenderPass()
	{
		if (shadowRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, shadowRenderPass, nullptr);
		if (filterRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, filterRenderPass, nullptr);
		if (mainRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, mainRenderPass, nullptr);

		hw1::createRenderPass(device, &shadowRenderPass, SHADOW_MAP_COLOR_FORMAT, depthFormat);
		hw1::createRenderPass(device, &filterRenderPass, SHADOW_MAP_COLOR_FORMAT);
		hw1::createRenderPass(device, &mainRenderPass, RENDER_TARGET_COLOR_FORMAT, depthFormat);
	}

	void recreateFrameBuffer()
	{
		if (shadowFrameBuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, shadowFrameBuffer, nullptr);
		if (filterFrameBuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, filterFrameBuffer, nullptr);
		if (mainFrameBuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, mainFrameBuffer, nullptr);

		hw1::createFrameBuffer(device, &shadowFrameBuffer, { SHADOW_MAP_SIZE , SHADOW_MAP_SIZE }, shadowRenderPass, shadowMapColorImage.view, shadowMapDepthImage.view);
		hw1::createFrameBuffer(device, &filterFrameBuffer, { SHADOW_MAP_SIZE , SHADOW_MAP_SIZE }, filterRenderPass, shadowMapFilteredImage.view);
		hw1::createFrameBuffer(device, &mainFrameBuffer, { width , height }, mainRenderPass, renderTargetColorImage.view, renderTargetDepthImage.view);
	}

	void updateLightInfo()
	{
		auto lightDir = glm::normalize(lightDirection);

		auto proj = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, 0.01f, 10.0f);
		// Flip Y
		proj[1][1] *= -1.0f;

		auto view = glm::lookAt(lightDir * 2.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		shaderData.values.lightMatrix = proj * view;
		shaderData.values.lightDir = glm::vec4(lightDir, 0.0);
		shaderData.values.lightIntensity = lightIntensity;
		shaderData.values.ambientIntensity = ambientIntensity;
	}

	void recreateMainPipelines()
	{
		if (pipelines.solid != VK_NULL_HANDLE) vkDestroyPipeline(device, pipelines.solid, nullptr);
		if (pipelines.wireframe != VK_NULL_HANDLE) vkDestroyPipeline(device, pipelines.wireframe, nullptr);
		if (pipelines.shadow != VK_NULL_HANDLE) vkDestroyPipeline(device, pipelines.shadow, nullptr);

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
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)),		// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)),	// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)),		// Location 2: Texture coordinates
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)),		// Location 3: Color
			vks::initializers::vertexInputAttributeDescription(0, 4, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, joint)),	// Location 4: Joint
			vks::initializers::vertexInputAttributeDescription(0, 5, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, weight)),	// Location 5: Weight
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

	void recreatePostprocessingPipeline()
	{
		if (pipelines.postprocessing != VK_NULL_HANDLE) vkDestroyPipeline(device, pipelines.postprocessing, nullptr);

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

	void recreateFilterPipeline()
	{
		if (pipelines.filter != VK_NULL_HANDLE) vkDestroyPipeline(device, pipelines.filter, nullptr);

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

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&postprocessingData.buffer,
			sizeof(postprocessingData.values)));

		VK_CHECK_RESULT(postprocessingData.buffer.map());

		updateUniformBuffers();
	}

	void recreateResources()
	{
		hw1::destroyImage2D(device, &shadowMapColorImage);
		hw1::destroyImage2D(device, &shadowMapDepthImage);
		hw1::destroyImage2D(device, &shadowMapFilteredImage);
		hw1::destroyImage2D(device, &renderTargetColorImage);
		hw1::destroyImage2D(device, &renderTargetDepthImage);

		hw1::createImage2D(*vulkanDevice, &shadowMapColorImage, SHADOW_MAP_COLOR_FORMAT, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 },
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

		hw1::createImage2D(*vulkanDevice, &shadowMapDepthImage, depthFormat, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 },
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

		hw1::createImage2D(*vulkanDevice, &shadowMapFilteredImage, SHADOW_MAP_COLOR_FORMAT, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 },
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

		hw1::createImage2D(*vulkanDevice, &renderTargetColorImage, RENDER_TARGET_COLOR_FORMAT, { width, height, 1 },
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

		hw1::createImage2D(*vulkanDevice, &renderTargetDepthImage, depthFormat, { width, height, 1 },
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

		if (shadowMapColorDescriptorSet != VK_NULL_HANDLE) vkFreeDescriptorSets(device, descriptorPool, 1, &shadowMapColorDescriptorSet);
		if (shadowMapFilteredDescriptorSet != VK_NULL_HANDLE) vkFreeDescriptorSets(device, descriptorPool, 1, &shadowMapFilteredDescriptorSet);
		if (renderTargetDescriptorSet != VK_NULL_HANDLE) vkFreeDescriptorSets(device, descriptorPool, 1, &renderTargetDescriptorSet);

		{
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.shadow, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &shadowMapColorDescriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(shadowMapColorDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &shadowMapColorImage.descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
		{
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.shadow, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &shadowMapFilteredDescriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(shadowMapFilteredDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &shadowMapFilteredImage.descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
		{
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.postprocessing, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &renderTargetDescriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(renderTargetDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &renderTargetColorImage.descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
	}

	void updateUniformBuffers()
	{
		updateLightInfo();

		shaderData.values.projection = camera.matrices.perspective;
		shaderData.values.model = camera.matrices.view;
		shaderData.values.viewPos = camera.viewPos;
		memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));

		postprocessingData.values.enableToneMapping = static_cast<int>(enableToneMapping);
		postprocessingData.values.enableVignette = static_cast<int>(enableVignette);
		postprocessingData.values.enableGrain = static_cast<int>(enableGrain);
		postprocessingData.values.enableChromaticAberration = static_cast<int>(enableChromaticAberration);
		memcpy(postprocessingData.buffer.mapped, &postprocessingData.values, sizeof(postprocessingData.values));
	}

	void prepare()
	{
		loadAssets();
		updateLightInfo();
		prepareUniformBuffers();
		setupDescriptors();
		VulkanExampleBase::prepare();
		/*
		 * The following 'recreate' functions will be called in setupFrameBuffer() in prepare() and especially in windowResize() for correctly resize
		 * 1) However, the descriptorPool is created in setupDescriptors() function and will be used in the following functions
		 *	  Therefore, VulkanExampleBase::prepare() must be called after setupDescriptors()
		 * 2) Also, the pipelineCache is created in VulkanExampleBase::prepare() which is required for creating pipelines
		 *	  Therefore, VulkanExampleBase::prepare() must be called before recreateMainPipelines()
		 */
		recreateResources();
		recreateRenderPass();
		recreateFrameBuffer();
		recreateMainPipelines();
		recreatePostprocessingPipeline();
		recreateFilterPipeline();
		buildCommandBuffers();
		timer.update();
		prepared = true;
	}

	void fixedUpdate()
	{
		if (animationPlay) {
			animationTime += fixedUpdateDelta;
			if (animationTime > glTFModel.animations[animationIndex].end) {
				animationTime = 0.0f;
			}
			glTFModel.update(animationIndex, animationTime);
		}
	}

	virtual void render()
	{
		renderFrame();
		updateUniformBuffers();
		fixedUpdateTimer += static_cast<float>(timer.deltaTime());
		timer.update();
		if (fixedUpdateTimer > fixedUpdateDelta) {
			fixedUpdateTimer -= fixedUpdateDelta;
			fixedUpdate();
		}
	}

	virtual void setupFrameBuffer()
	{
		VulkanExampleBase::setupFrameBuffer();
		recreateResources();
		recreateRenderPass();
		recreateFrameBuffer();
		recreateMainPipelines();
		recreatePostprocessingPipeline();
		recreateFilterPipeline();
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->checkBox("Wireframe", &wireframe)) {
				buildCommandBuffers();
			}
			if (overlay->header("Animation")) {
				overlay->comboBox("Animations", &animationIndex, animationNames);
				overlay->checkBox("Play", &animationPlay);
				overlay->sliderFloat("Time", &animationTime, glTFModel.animations[animationIndex].start, glTFModel.animations[animationIndex].end);
			}
			if (overlay->header("Postprocessing")) {
				overlay->checkBox("ToneMapping", &enableToneMapping);
				overlay->checkBox("Vignette", &enableVignette);
				overlay->checkBox("Grain", &enableGrain);
				overlay->checkBox("ChromaticAberration", &enableChromaticAberration);
			}
			if (overlay->header("Lighting")) {
				overlay->sliderFloat("Light Intensity", &lightIntensity, 0.0f, 10.0f);
				overlay->sliderFloat("Light Direction X", &lightDirection.x, -1.0f, 1.0f);
				overlay->sliderFloat("Light Direction Y", &lightDirection.y, -1.0f, 1.0f);
				overlay->sliderFloat("Light Direction Z", &lightDirection.z, -1.0f, 1.0f);
				overlay->sliderFloat("Ambient Intensity", &ambientIntensity, 0.0f, 10.0f);
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
