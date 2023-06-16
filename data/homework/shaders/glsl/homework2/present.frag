#version 450

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_fragColor;

layout (set = 0, binding = 0) uniform sampler2D u_sampler;

void main() {
	vec3 color = texture(u_sampler, in_uv).rgb;
	out_fragColor = vec4(color, 1.0);
}
