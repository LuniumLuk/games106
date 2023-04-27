#version 450

layout (location = 0) out vec4 outFragColor;

void main() {
	float depth = gl_FragCoord.z;
	
	float dzdx = dFdx(depth);
	float dzdy = dFdy(depth);
	float moment2 = depth * depth + 0.25 * (dzdx * dzdx + dzdy * dzdy);

	outFragColor = vec4(depth, moment2, 0.0, 0.0);
	gl_FragDepth = depth;
}