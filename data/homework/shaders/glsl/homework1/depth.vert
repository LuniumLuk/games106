#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;
layout (location = 4) in vec4 inJoint;
layout (location = 5) in vec4 inWeight;

layout (set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightDir;
	vec4 viewPos;
	mat4 lightMatrix;
	float lightIntensity;
} uboScene;

layout(push_constant) uniform PushConsts {
	mat4 model;
	vec4 baseColorFactor;
	float roughnessFactor;
	float metallicFactor;
} primitive;

#define MAX_NUM_JOINTS 32

layout (set = 3, binding = 0) uniform UBOAnimation
{
	mat4 matrix;
	mat4 matrixInverseTransposed;
	mat4 jointMatrices[MAX_NUM_JOINTS];
	int jointCount;
} uboAnimation;

void main() 
{
	vec4 pos;
	if (uboAnimation.jointCount == 0) {
		pos = uboAnimation.matrix * vec4(inPos, 1.0);
	}
	else {
		mat4 skinningMatrix =
			inWeight.x * uboAnimation.jointMatrices[min(int(inJoint.x), MAX_NUM_JOINTS)] +
			inWeight.y * uboAnimation.jointMatrices[min(int(inJoint.y), MAX_NUM_JOINTS)] +
			inWeight.z * uboAnimation.jointMatrices[min(int(inJoint.z), MAX_NUM_JOINTS)] +
			inWeight.w * uboAnimation.jointMatrices[min(int(inJoint.w), MAX_NUM_JOINTS)];
		
		pos = uboAnimation.matrix * skinningMatrix * vec4(inPos, 1.0);
	}

	gl_Position = uboScene.lightMatrix * pos;
}