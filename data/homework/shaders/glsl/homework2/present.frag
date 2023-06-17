#version 450

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_fragColor;

layout (set = 0, binding = 0) uniform sampler2D u_image;
layout (set = 0, binding = 1) uniform sampler2D u_shadingRateVisualize;

void main() {
	vec3 color = texture(u_image, in_uv).rgb;
	vec3 visualize = texture(u_shadingRateVisualize, in_uv).rgb;
	vec3 visualizeColor = vec3(1.0);

	if (int(visualize.r) == 1 && int(visualize.g) == 1) {
		visualizeColor = vec3(0.2, 0.4, 0.2);
	}
	if (int(visualize.r) == 2 && int(visualize.g) == 1) {
		visualizeColor = vec3(0.6, 0.2, 0.0);
	}
	if (int(visualize.r) == 1 && int(visualize.g) == 2) {
		visualizeColor = vec3(0.0, 0.2, 0.6);
	}
	if (int(visualize.r) == 2 && int(visualize.g) == 2) {
		visualizeColor = vec3(0.4, 0.6, 0.4);
	}
	if (int(visualize.r) == 4 && int(visualize.g) == 2) {
		visualizeColor = vec3(1.0, 0.4, 0.0);
	}
	if (int(visualize.r) == 2 && int(visualize.g) == 4) {
		visualizeColor = vec3(0.0, 0.4, 1.0);
	}
	if (int(visualize.r) == 4 && int(visualize.g) == 4) {
		visualizeColor = vec3(0.6, 1.0, 0.6);
	}

	out_fragColor = vec4(color * visualizeColor, 1.0);
}
