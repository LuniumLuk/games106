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
	float ambientIntensity;
} uboScene;

#define MAX_NUM_JOINTS 32

layout (set = 3, binding = 0) uniform UBOAnimation
{
	mat4 matrix;
	mat4 matrixInverseTransposed;
	mat4 jointMatrices[MAX_NUM_JOINTS];
	int jointCount;
} uboAnimation;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;
layout (location = 5) out vec3 outWorldPos;
layout (location = 6) out vec4 outLightSpacePos;

void main() 
{
	outNormal = inNormal;
	outColor = inColor;
	outUV = inUV;

	vec4 pos;
	if (uboAnimation.jointCount == 0) {
		pos = uboAnimation.matrix * vec4(inPos, 1.0);
		outNormal = mat3(uboAnimation.matrixInverseTransposed) * inNormal;
	}
	else {
		mat4 skinningMatrix =
			inWeight.x * uboAnimation.jointMatrices[min(int(inJoint.x), MAX_NUM_JOINTS)] +
			inWeight.y * uboAnimation.jointMatrices[min(int(inJoint.y), MAX_NUM_JOINTS)] +
			inWeight.z * uboAnimation.jointMatrices[min(int(inJoint.z), MAX_NUM_JOINTS)] +
			inWeight.w * uboAnimation.jointMatrices[min(int(inJoint.w), MAX_NUM_JOINTS)];
		
		pos = uboAnimation.matrix * skinningMatrix * vec4(inPos, 1.0);
		outNormal = mat3(uboAnimation.matrixInverseTransposed) * inverse(transpose(mat3(skinningMatrix))) * inNormal;
	}

	gl_Position = uboScene.projection * uboScene.view * pos;
	outLightVec = uboScene.lightDir.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
	outWorldPos = pos.xyz;
	outLightSpacePos = uboScene.lightMatrix * pos;
}