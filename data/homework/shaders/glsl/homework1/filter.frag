#version 450

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0) uniform sampler2D samplerInput;

// 5x5 Gaussian kernel with sigma = (5 + 1) / 6
float K[5][5] = {
	{ 0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902 },
	{ 0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621 },
	{ 0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823 },
	{ 0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621 },
	{ 0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902 },
};

void main() {
	vec2 texelSize = vec2(1.0 / textureSize(samplerInput, 0));
	vec3 result = vec3(0.0);
	for (int i = -2; i <= 2; ++i) {
		for (int j = -2; j <= 2; ++j) {
			result += K[i + 2][j + 2] * texture(samplerInput, inUV + vec2(i, j) * texelSize).rgb;
		}
	}
	outFragColor = vec4(result, 1.0);
}

