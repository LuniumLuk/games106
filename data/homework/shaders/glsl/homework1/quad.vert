#version 450

layout (location = 0) out vec2 outUV;

// gl_VertexIndex: 0 -> Texcoord: (0.0, 0.0)
// gl_VertexIndex: 1 -> Texcoord: (2.0, 0.0)
// gl_VertexIndex: 2 -> Texcoord: (0.0, 2.0)

void main() {
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}