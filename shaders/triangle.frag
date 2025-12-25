#version 450

layout(binding = 1) uniform sampler2D texSampler;
// layout(binding = 0) uniform texture2D texImage;        // 只有图像数据
// layout(binding = 1) uniform sampler texSampler;        // 只有采样器
// vec4 color = texture(sampler2D(texImage, texSampler), texCoord); // // 需要手动组合图像和采样器

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord);
}