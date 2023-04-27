#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;

layout (set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightDir;
	vec4 viewPos;
	mat4 lightMatrix;
} uboScene;

layout(push_constant) uniform PushConsts {
	mat4 model;
} primitive;

void main() 
{
	gl_Position = uboScene.lightMatrix * primitive.model * vec4(inPos.xyz, 1.0);
}