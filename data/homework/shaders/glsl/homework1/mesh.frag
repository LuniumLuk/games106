#version 450

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

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 1, binding = 1) uniform sampler2D samplerNormalMap;
layout (set = 1, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout (set = 1, binding = 3) uniform sampler2D samplerEmissiveMap;
layout (set = 1, binding = 4) uniform sampler2D samplerOcclusionMap;

layout (set = 1, binding = 5) uniform UBOMaterial
{
	vec4 baseColorFactor;
	float roughnessFactor;
	float metallicFactor;
} uboMaterial;

layout (set = 2, binding = 0) uniform sampler2D samplerShadowMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec3 inWorldPos;
layout (location = 6) in vec4 inLightSpacePos;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform PushConsts {
	mat4 model;
	vec4 baseColorFactor;
	float roughnessFactor;
	float metallicFactor;
} primitive;

#define PI 				3.141592653590

vec4 GammaTransform(vec4 color) {
	return vec4(pow(color.rgb, vec3(2.2)), color.a);
}

struct PBRState {
	float NoL;						// Cos angle between normal and light direction
	float NoV;						// Cos angle between normal and view direction
	float NoH;						// Cos angle between normal and half vector
	float LoH;						// Cos angle between light direction and half vector
	float VoH;						// Cos angle between view direction and half vector
	float alphaRoughness;			// Roughness value remapped linearly
	float perceptualRoughness;		// Roughness value originally
	float metallic;					// Metallic value
	vec3 R0;						// Reflectance color at normal incident angle
	vec3 R90;						// Reflectance color at grazing angle
	vec3 diffuseColor;				// Color contribution from diffuse lighting
	vec3 specularColor;				// Color contribution from specular lighting
};

float sqr(float x) {
	return x * x;
}

float pow5(float x) {
	float x2 = x * x;
	return x2 * x2 * x;
}

mat3 GetOrthoBasis() {
	vec3 q1 = dFdx(inWorldPos);
	vec3 q2 = dFdy(inWorldPos);
	vec3 N = inNormal;
	vec2 st1 = dFdx(inUV);
	vec2 st2 = dFdy(inUV);
	float s = float(gl_FrontFacing) * 2.0 - 1.0;
	vec3 T = s * normalize(q1 * st2.t - q2 * st1.t);
	// Left handedness
	vec3 B = -normalize(cross(N, T));
	return mat3(T, B, N);
}

vec3 LambertianDiffuse(PBRState state) {
	return state.diffuseColor / PI;
}

vec3 SpecularFresnel(PBRState state) {
	return state.R0 + (state.R90 - state.R0) * pow5(clamp(1.0 - state.VoH, 0.0, 1.0));
}

float GeometricOcclusion(PBRState state) {
	float NoL = state.NoL;
	float NoV = state.NoV;
	float r = state.alphaRoughness;

	float attenuationL = 2.0 * NoL / (NoL + sqrt(r * r + (1.0 - r * r) * (NoL * NoL)));
	float attenuationV = 2.0 * NoV / (NoV + sqrt(r * r + (1.0 - r * r) * (NoV * NoV)));
	return attenuationL * attenuationV;
}

float MicrofacetDistribution(PBRState state) {
	float a2 = state.alphaRoughness * state.alphaRoughness;
	float f = (state.NoH * a2 - state.NoH) * state.NoH + 1.0;
	return a2 / (PI * f * f);
}

vec3 GetLightSpaceCoords() {
	vec3 coords = inLightSpacePos.xyz / inLightSpacePos.w;
	coords.xy = coords.xy * 0.5 + 0.5;
	return coords;
}

float SampleShadowVSM(vec2 coords, float currentDepth) {
	vec2 moments = texture(samplerShadowMap, coords).xy;

	float p = step(currentDepth, moments.x);
	float variance = max(moments.y - sqr(moments.x), 0.00001);

	float d = currentDepth - moments.x;
	float pMax = smoothstep(0.2, 1.0, variance / (variance + sqr(d)));
	
	return 1.0 - min(max(p, pMax), 1.0);
}

float VarianceShadowMap() {
	vec3 coords = GetLightSpaceCoords();
	if (coords.z > 1.0) return 0.0;

	float currentDepth = coords.z;
	return SampleShadowVSM(coords.xy, currentDepth);
}

void main() 
{
	vec4 baseColor = GammaTransform(texture(samplerColorMap, inUV)) * vec4(inColor, 1.0) * uboMaterial.baseColorFactor;
	vec3 normal = texture(samplerNormalMap, inUV).rgb;
	vec3 pbr = texture(samplerMetallicRoughnessMap, inUV).rgb;
	vec3 emissive = texture(samplerEmissiveMap, inUV).rgb;
	float occlusion = texture(samplerOcclusionMap, inUV).r;
	float shadow = VarianceShadowMap();

	// Tangent Basis
	mat3 TBN = GetOrthoBasis();
	vec3 N = normalize(TBN * (normal * 2.0 - 1.0));
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 H = normalize(L + V);

	float NoL = clamp(dot(N, L), 0.001, 1.0);
	float NoV = clamp(abs(dot(N, V)), 0.001, 1.0);
	float NoH = clamp(dot(N, H), 0.0, 1.0);
	float LoH = clamp(dot(L, H), 0.0, 1.0);
	float VoH = clamp(dot(V, H), 0.0, 1.0);

	// PBR Parameters
	float perceptualRoughness = pbr.g * uboMaterial.roughnessFactor;
	float alphaRoughness = perceptualRoughness * perceptualRoughness;
	float metallic = pbr.b * uboMaterial.metallicFactor;

	vec3 F0 = vec3(0.04);
	vec3 diffuseColor = baseColor.rgb * (vec3(1.0) - F0) * (1.0 - metallic);
	vec3 specularColor = mix(F0, baseColor.rgb, metallic);

	float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
	float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
	vec3 R0 = specularColor.rgb;
	vec3 R90 = vec3(reflectance90);

	PBRState state = PBRState(
		NoL,
		NoV,
		NoH,
		LoH,
		VoH,
		alphaRoughness,
		perceptualRoughness,
		metallic,
		R0,
		R90,
		diffuseColor,
		specularColor
	);

	// BRDFs
	vec3 F = SpecularFresnel(state);
	float G = GeometricOcclusion(state);
	float D = MicrofacetDistribution(state);

	// Diffuse and Specular components
	vec3 diffuse = (1.0 - F) * LambertianDiffuse(state);
	vec3 specular = F * G * D / (4.0 * NoL * NoV);

	// Apply lighting (directional + ambient)
	float lightIntensity = uboScene.lightIntensity;
	vec3 color = (diffuse + specular) * lightIntensity * occlusion * (1 - shadow) + emissive;
	color += (diffuse + specular) * uboScene.ambientIntensity;

	// Output color in linear space
	outFragColor = vec4(color, baseColor.a);
}