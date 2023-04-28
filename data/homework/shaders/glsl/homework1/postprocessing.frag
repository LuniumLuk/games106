#version 450

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0) uniform sampler2D samplerInput;

layout (set = 1, binding = 0) uniform UBOParameter
{
	int enableToneMapping;
	int enableVignette;
	int enableGrain;
	int enableChromaticAberration;
} uboParameter;

const float vignetteFactor = 0.5;
const float grainFactor = 0.01;
const float chromaticAberrationFactor = 0.01;

vec3 Tonemap_ACES(const vec3 c) {
	// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
	// const float a = 2.51;
	// const float b = 0.03;
	// const float c = 2.43;
	// const float d = 0.59;
	// const float e = 0.14;
	// return saturate((x*(a*x+b))/(x*(c*x+d)+e));

	//ACES RRT/ODT curve fit courtesy of Stephen Hill
	vec3 a = c * (c + 0.0245786) - 0.000090537;
	vec3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
	return a / b;
}

float Vignette() {
	vec2 uv = inUV * 2.0 - 1.0;
	float vignette = sqrt(uv.x * uv.x + uv.y * uv.y);
	vignette = clamp(vignette, 0.0, 1.0);
	return 1.0 - vignette * vignetteFactor;
}

vec3 Grain() {
	vec2 uv = inUV * 2.0 - 1.0;
	float grain = fract(sin(dot(uv.xy, vec2(12.9898, 78.233))) * 43758.5453);
	return vec3(grain * grainFactor);
}

vec3 ChromaticAberration() {
	vec2 uv = inUV * 2.0 - 1.0;
	uv = vec2(
		sign(uv.x) * uv.x * uv.x,
		sign(uv.y) * uv.y * uv.y
	);
	vec2 offset = uv * chromaticAberrationFactor;
	float r = texture(samplerInput, clamp(inUV + offset, vec2(0.0), vec2(1.0))).r;
	float g = texture(samplerInput, inUV).g;
	float b = texture(samplerInput, clamp(inUV - offset, vec2(0.0), vec2(1.0))).b;
	return vec3(r, g, b);
}

void main() {
	vec3 color;
	if (uboParameter.enableChromaticAberration == 1) {
		color = ChromaticAberration();
	}
	else {
		color = texture(samplerInput, inUV).rgb;
	}
	if (uboParameter.enableToneMapping == 1) color = Tonemap_ACES(color);
	if (uboParameter.enableVignette == 1) color *= Vignette();
	if (uboParameter.enableGrain == 1) color += Grain();
	outFragColor = vec4(color, 1.0);
}

