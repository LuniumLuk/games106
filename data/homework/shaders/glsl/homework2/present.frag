#version 450

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_fragColor;

layout (set = 0, binding = 0) uniform sampler2D u_image;
layout (set = 0, binding = 1) uniform sampler2D u_shadingRateVisualize;

void main() {
	vec3 color = texture(u_image, in_uv).rgb;
	vec3 visualize = texture(u_shadingRateVisualize, in_uv).rgb;
	out_fragColor = vec4(color * visualize, 1.0);
}
