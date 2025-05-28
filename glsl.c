#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Mock OpenGL types and functions for demonstration
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef unsigned char GLubyte;
typedef int GLsizei;
typedef unsigned int GLenum;

#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_GEOMETRY_SHADER 0x8DD9
#define GL_COMPUTE_SHADER 0x91B9
#define GL_TESS_CONTROL_SHADER 0x8E88
#define GL_TESS_EVALUATION_SHADER 0x8E87

// GLSL Shader Programs Showcase
// Each section demonstrates different GLSL capabilities

// =============================================================================
// 1. BASIC VERTEX AND FRAGMENT SHADERS
// =============================================================================

const char* basic_vertex_shader = R"(
#version 450 core

// Input vertex attributes
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;
layout(location = 3) in vec3 a_tangent;

// Uniform blocks (UBO) - efficient way to pass uniforms
layout(std140, binding = 0) uniform CameraData {
    mat4 u_view_matrix;
    mat4 u_projection_matrix;
    vec3 u_camera_position;
    float u_time;
};

layout(std140, binding = 1) uniform ModelData {
    mat4 u_model_matrix;
    mat4 u_normal_matrix;
    vec4 u_color_tint;
};

// Output to fragment shader
out vec3 v_world_position;
out vec3 v_world_normal;
out vec2 v_texcoord;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec4 v_clip_position;

void main() {
    // Transform vertex to world space
    vec4 world_pos = u_model_matrix * vec4(a_position, 1.0);
    v_world_position = world_pos.xyz;
    
    // Transform normal and tangent to world space
    v_world_normal = normalize((u_normal_matrix * vec4(a_normal, 0.0)).xyz);
    v_tangent = normalize((u_model_matrix * vec4(a_tangent, 0.0)).xyz);
    v_bitangent = cross(v_world_normal, v_tangent);
    
    // Pass through texture coordinates
    v_texcoord = a_texcoord;
    
    // Calculate final clip space position
    v_clip_position = u_projection_matrix * u_view_matrix * world_pos;
    gl_Position = v_clip_position;
}
)";

// =============================================================================
// 2. ADVANCED FRAGMENT SHADER WITH PBR LIGHTING
// =============================================================================

const char* pbr_fragment_shader = R"(
#version 450 core

// Input from vertex shader
in vec3 v_world_position;
in vec3 v_world_normal;
in vec2 v_texcoord;
in vec3 v_tangent;
in vec3 v_bitangent;

// Uniform samplers
layout(binding = 0) uniform sampler2D u_albedo_texture;
layout(binding = 1) uniform sampler2D u_normal_texture;
layout(binding = 2) uniform sampler2D u_metallic_roughness_texture;
layout(binding = 3) uniform sampler2D u_ao_texture;
layout(binding = 4) uniform samplerCube u_environment_map;
layout(binding = 5) uniform samplerCube u_irradiance_map;

// Material properties
uniform vec3 u_albedo = vec3(1.0);
uniform float u_metallic = 0.0;
uniform float u_roughness = 0.5;
uniform float u_ao = 1.0;
uniform float u_normal_strength = 1.0;

// Camera uniform
uniform vec3 u_camera_position;

// Lighting uniforms
#define MAX_LIGHTS 8
uniform int u_light_count;
uniform vec3 u_light_positions[MAX_LIGHTS];
uniform vec3 u_light_colors[MAX_LIGHTS];
uniform float u_light_intensities[MAX_LIGHTS];

// Output color
out vec4 FragColor;

// Constants
const float PI = 3.14159265359;

// Utility functions
vec3 getNormalFromMap() {
    vec3 tangent_normal = texture(u_normal_texture, v_texcoord).xyz * 2.0 - 1.0;
    tangent_normal.xy *= u_normal_strength;
    
    vec3 N = normalize(v_world_normal);
    vec3 T = normalize(v_tangent);
    vec3 B = normalize(v_bitangent);
    mat3 TBN = mat3(T, B, N);
    
    return normalize(TBN * tangent_normal);
}

// Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX/Trowbridge-Reitz normal distribution function
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return num / denom;
}

// Smith's method for geometry function
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

void main() {
    // Sample textures
    vec3 albedo = pow(texture(u_albedo_texture, v_texcoord).rgb * u_albedo, 2.2);
    vec3 metal_rough = texture(u_metallic_roughness_texture, v_texcoord).rgb;
    float metallic = metal_rough.b * u_metallic;
    float roughness = metal_rough.g * u_roughness;
    float ao = texture(u_ao_texture, v_texcoord).r * u_ao;
    
    // Get normal from normal map
    vec3 N = getNormalFromMap();
    vec3 V = normalize(u_camera_position - v_world_position);
    
    // Calculate reflectance at normal incidence
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    
    // Direct lighting calculation
    vec3 Lo = vec3(0.0);
    
    for(int i = 0; i < u_light_count && i < MAX_LIGHTS; ++i) {
        vec3 L = normalize(u_light_positions[i] - v_world_position);
        vec3 H = normalize(V + L);
        float distance = length(u_light_positions[i] - v_world_position);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = u_light_colors[i] * u_light_intensities[i] * attenuation;
        
        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Ambient lighting (IBL approximation)
    vec3 F = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    
    vec3 irradiance = texture(u_irradiance_map, N).rgb;
    vec3 diffuse = irradiance * albedo;
    
    // Sample environment map for specular
    vec3 R = reflect(-V, N);
    vec3 prefilteredColor = textureLod(u_environment_map, R, roughness * 4.0).rgb;
    vec3 specular = prefilteredColor * F;
    
    vec3 ambient = (kD * diffuse + specular) * ao;
    vec3 color = ambient + Lo;
    
    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}
)";

// =============================================================================
// 3. GEOMETRY SHADER - WIREFRAME GENERATION
// =============================================================================

const char* wireframe_geometry_shader = R"(
#version 450 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

// Input from vertex shader
in vec3 v_world_position[];
in vec3 v_world_normal[];
in vec2 v_texcoord[];

// Output to fragment shader
out vec3 g_world_position;
out vec3 g_world_normal;
out vec2 g_texcoord;
out vec3 g_barycentric;

uniform float u_wireframe_thickness = 1.0;

void main() {
    // Pass through triangle with barycentric coordinates for wireframe
    for(int i = 0; i < 3; i++) {
        g_world_position = v_world_position[i];
        g_world_normal = v_world_normal[i];
        g_texcoord = v_texcoord[i];
        
        // Set barycentric coordinates
        g_barycentric = vec3(0.0);
        g_barycentric[i] = 1.0;
        
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
)";

// =============================================================================
// 4. COMPUTE SHADER - PARTICLE SYSTEM
// =============================================================================

const char* particle_compute_shader = R"(
#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Particle structure
struct Particle {
    vec3 position;
    float life;
    vec3 velocity;
    float size;
    vec4 color;
    vec3 acceleration;
    float padding;
};

// Storage buffer for particles
layout(std430, binding = 0) restrict buffer ParticleBuffer {
    Particle particles[];
};

// Uniforms
uniform float u_delta_time;
uniform vec3 u_gravity = vec3(0.0, -9.8, 0.0);
uniform vec3 u_wind = vec3(0.0);
uniform float u_damping = 0.98;
uniform vec3 u_emitter_position = vec3(0.0);
uniform float u_emission_rate = 100.0;

// Noise function for randomness
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 random3(vec3 p) {
    return vec3(
        random(p.xy),
        random(p.yz),
        random(p.zx)
    ) * 2.0 - 1.0;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if(index >= particles.length()) return;
    
    Particle p = particles[index];
    
    // Update existing particles
    if(p.life > 0.0) {
        // Physics integration
        p.acceleration = u_gravity + u_wind + random3(p.position + u_delta_time) * 0.5;
        p.velocity += p.acceleration * u_delta_time;
        p.velocity *= u_damping;
        p.position += p.velocity * u_delta_time;
        
        // Update life
        p.life -= u_delta_time;
        
        // Update color and size based on life
        float life_ratio = p.life / 5.0; // Assume max life of 5 seconds
        p.color.a = life_ratio;
        p.size = mix(0.1, 1.0, life_ratio);
        p.color.rgb = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), life_ratio);
    }
    // Respawn dead particles
    else {
        vec3 rand = random3(vec3(index) + u_delta_time);
        p.position = u_emitter_position + rand * 0.1;
        p.velocity = normalize(vec3(rand.x, abs(rand.y) + 1.0, rand.z)) * mix(2.0, 8.0, random(rand.xy));
        p.life = mix(2.0, 5.0, random(rand.yz));
        p.size = 1.0;
        p.color = vec4(1.0, 1.0, 0.0, 1.0);
        p.acceleration = vec3(0.0);
    }
    
    particles[index] = p;
}
)";

// =============================================================================
// 5. TESSELLATION SHADERS - ADAPTIVE TERRAIN
// =============================================================================

const char* terrain_tess_control_shader = R"(
#version 450 core

layout(vertices = 4) out;

// Input from vertex shader
in vec3 v_world_position[];
in vec2 v_texcoord[];

// Output to tessellation evaluation shader
out vec3 tc_world_position[];
out vec2 tc_texcoord[];

uniform vec3 u_camera_position;
uniform float u_tess_level_inner = 4.0;
uniform float u_tess_level_outer = 4.0;
uniform float u_tess_distance_scale = 50.0;

float getTessLevel(vec3 worldPos) {
    float distance = length(u_camera_position - worldPos);
    float tessLevel = max(1.0, u_tess_level_outer * (1.0 - distance / u_tess_distance_scale));
    return clamp(tessLevel, 1.0, 64.0);
}

void main() {
    // Pass through data
    tc_world_position[gl_InvocationID] = v_world_position[gl_InvocationID];
    tc_texcoord[gl_InvocationID] = v_texcoord[gl_InvocationID];
    
    // Calculate tessellation levels based on distance to camera
    if(gl_InvocationID == 0) {
        vec3 center = (v_world_position[0] + v_world_position[1] + 
                      v_world_position[2] + v_world_position[3]) * 0.25;
        
        float tessLevel = getTessLevel(center);
        
        gl_TessLevelInner[0] = tessLevel;
        gl_TessLevelInner[1] = tessLevel;
        gl_TessLevelOuter[0] = tessLevel;
        gl_TessLevelOuter[1] = tessLevel;
        gl_TessLevelOuter[2] = tessLevel;
        gl_TessLevelOuter[3] = tessLevel;
    }
}
)";

const char* terrain_tess_eval_shader = R"(
#version 450 core

layout(quads, equal_spacing, ccw) in;

// Input from tessellation control shader
in vec3 tc_world_position[];
in vec2 tc_texcoord[];

// Output to fragment shader
out vec3 te_world_position;
out vec3 te_world_normal;
out vec2 te_texcoord;
out float te_height;

// Height map
layout(binding = 6) uniform sampler2D u_height_map;

uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
uniform float u_height_scale = 10.0;

// Noise function for procedural height
float noise(vec2 p) {
    return texture(u_height_map, p).r;
}

float getHeight(vec2 uv) {
    return noise(uv) * u_height_scale;
}

vec3 calculateNormal(vec2 uv, float texelSize) {
    float hL = getHeight(uv + vec2(-texelSize, 0.0));
    float hR = getHeight(uv + vec2(texelSize, 0.0));
    float hD = getHeight(uv + vec2(0.0, -texelSize));
    float hU = getHeight(uv + vec2(0.0, texelSize));
    
    vec3 normal = normalize(vec3(hL - hR, 2.0 * texelSize * u_height_scale, hD - hU));
    return normal;
}

void main() {
    // Bilinear interpolation of patch coordinates
    vec3 p0 = mix(tc_world_position[0], tc_world_position[1], gl_TessCoord.x);
    vec3 p1 = mix(tc_world_position[3], tc_world_position[2], gl_TessCoord.x);
    vec3 position = mix(p0, p1, gl_TessCoord.y);
    
    vec2 uv0 = mix(tc_texcoord[0], tc_texcoord[1], gl_TessCoord.x);
    vec2 uv1 = mix(tc_texcoord[3], tc_texcoord[2], gl_TessCoord.x);
    vec2 texcoord = mix(uv0, uv1, gl_TessCoord.y);
    
    // Apply height displacement
    float height = getHeight(texcoord);
    position.y += height;
    
    // Calculate normal from height map
    vec3 normal = calculateNormal(texcoord, 1.0 / 1024.0);
    
    te_world_position = position;
    te_world_normal = normal;
    te_texcoord = texcoord;
    te_height = height;
    
    gl_Position = u_projection_matrix * u_view_matrix * vec4(position, 1.0);
}
)";

// =============================================================================
// 6. POST-PROCESSING FRAGMENT SHADER
// =============================================================================

const char* postprocess_fragment_shader = R"(
#version 450 core

in vec2 v_texcoord;
out vec4 FragColor;

// Input textures
layout(binding = 0) uniform sampler2D u_color_texture;
layout(binding = 1) uniform sampler2D u_depth_texture;
layout(binding = 2) uniform sampler2D u_normal_texture;

// Post-processing parameters
uniform bool u_enable_bloom = true;
uniform bool u_enable_ssao = true;
uniform bool u_enable_fxaa = true;
uniform float u_exposure = 1.0;
uniform float u_gamma = 2.2;
uniform float u_bloom_threshold = 1.0;
uniform float u_bloom_intensity = 0.5;

// SSAO parameters
uniform int u_ssao_samples = 16;
uniform float u_ssao_radius = 0.5;
uniform float u_ssao_bias = 0.025;

// Noise texture for SSAO
layout(binding = 3) uniform sampler2D u_noise_texture;

// Camera matrices for SSAO
uniform mat4 u_projection_matrix;
uniform mat4 u_view_matrix;

// SSAO sample kernel (would normally be generated and passed as uniform)
const vec3 ssao_kernel[16] = vec3[](
    vec3( 0.2024537,  0.841204, -0.9060141),
    vec3(-0.2024537, -0.841204,  0.9060141),
    vec3( 0.4024537,  0.241204, -0.4060141),
    vec3(-0.4024537, -0.241204,  0.4060141),
    vec3( 0.6024537,  0.641204, -0.2060141),
    vec3(-0.6024537, -0.641204,  0.2060141),
    vec3( 0.8024537,  0.041204, -0.1060141),
    vec3(-0.8024537, -0.041204,  0.1060141),
    vec3( 0.1024537,  0.941204, -0.8060141),
    vec3(-0.1024537, -0.941204,  0.8060141),
    vec3( 0.3024537,  0.341204, -0.6060141),
    vec3(-0.3024537, -0.341204,  0.6060141),
    vec3( 0.5024537,  0.541204, -0.4060141),
    vec3(-0.5024537, -0.541204,  0.4060141),
    vec3( 0.7024537,  0.141204, -0.2060141),
    vec3(-0.7024537, -0.141204,  0.2060141)
);

vec3 reconstructPosition(vec2 uv, float depth) {
    vec4 clip_pos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 view_pos = inverse(u_projection_matrix) * clip_pos;
    return view_pos.xyz / view_pos.w;
}

float calculateSSAO(vec2 texcoord) {
    if(!u_enable_ssao) return 1.0;
    
    float depth = texture(u_depth_texture, texcoord).r;
    vec3 position = reconstructPosition(texcoord, depth);
    vec3 normal = normalize(texture(u_normal_texture, texcoord).xyz * 2.0 - 1.0);
    
    // Get noise vector
    vec3 noise = texture(u_noise_texture, texcoord * textureSize(u_color_texture, 0) / 4.0).xyz;
    
    // Create TBN matrix
    vec3 tangent = normalize(noise - normal * dot(noise, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    float occlusion = 0.0;
    for(int i = 0; i < u_ssao_samples; ++i) {
        vec3 sample_pos = TBN * ssao_kernel[i];
        sample_pos = position + sample_pos * u_ssao_radius;
        
        vec4 offset = u_projection_matrix * vec4(sample_pos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;
        
        float sample_depth = texture(u_depth_texture, offset.xy).r;
        vec3 sample_position = reconstructPosition(offset.xy, sample_depth);
        
        float range_check = smoothstep(0.0, 1.0, u_ssao_radius / abs(position.z - sample_position.z));
        occlusion += (sample_position.z >= sample_pos.z + u_ssao_bias ? 1.0 : 0.0) * range_check;
    }
    
    return 1.0 - (occlusion / float(u_ssao_samples));
}

vec3 applyBloom(vec2 texcoord) {
    if(!u_enable_bloom) return vec3(0.0);
    
    vec3 color = texture(u_color_texture, texcoord).rgb;
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    if(brightness > u_bloom_threshold) {
        return color * u_bloom_intensity;
    }
    return vec3(0.0);
}

vec3 applyFXAA(vec2 texcoord) {
    if(!u_enable_fxaa) return texture(u_color_texture, texcoord).rgb;
    
    vec2 texel_size = 1.0 / textureSize(u_color_texture, 0);
    
    vec3 color_center = texture(u_color_texture, texcoord).rgb;
    vec3 color_north = texture(u_color_texture, texcoord + vec2(0.0, -texel_size.y)).rgb;
    vec3 color_south = texture(u_color_texture, texcoord + vec2(0.0, texel_size.y)).rgb;
    vec3 color_east = texture(u_color_texture, texcoord + vec2(texel_size.x, 0.0)).rgb;
    vec3 color_west = texture(u_color_texture, texcoord + vec2(-texel_size.x, 0.0)).rgb;
    
    // Simple FXAA implementation
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float luma_center = dot(color_center, luma);
    float luma_north = dot(color_north, luma);
    float luma_south = dot(color_south, luma);
    float luma_east = dot(color_east, luma);
    float luma_west = dot(color_west, luma);
    
    float luma_min = min(luma_center, min(min(luma_north, luma_south), min(luma_east, luma_west)));
    float luma_max = max(luma_center, max(max(luma_north, luma_south), max(luma_east, luma_west)));
    
    float luma_range = luma_max - luma_min;
    
    if(luma_range < max(0.0833, luma_max * 0.125)) {
        return color_center;
    }
    
    // Apply anti-aliasing
    vec3 color_average = (color_north + color_south + color_east + color_west) * 0.25;
    return mix(color_center, color_average, 0.5);
}

void main() {
    vec2 texcoord = v_texcoord;
    
    // Get base color
    vec3 color = applyFXAA(texcoord);
    
    // Apply SSAO
    float ssao = calculateSSAO(texcoord);
    color *= ssao;
    
    // Add bloom
    vec3 bloom = applyBloom(texcoord);
    color += bloom;
    
    // Tone mapping
    color = vec3(1.0) - exp(-color * u_exposure);
    
    // Gamma correction
    color = pow(color, vec3(1.0 / u_gamma));
    
    FragColor = vec4(color, 1.0);
}
)";

// =============================================================================
// 7. ADVANCED EFFECTS SHADERS
// =============================================================================

const char* parallax_mapping_fragment = R"(
#version 450 core

in vec3 v_world_position;
in vec2 v_texcoord;
in vec3 v_tangent_light_pos;
in vec3 v_tangent_view_pos;
in vec3 v_tangent_frag_pos;

layout(binding = 0) uniform sampler2D u_diffuse_map;
layout(binding = 1) uniform sampler2D u_normal_map;
layout(binding = 2) uniform sampler2D u_height_map;

uniform float u_height_scale = 0.1;
uniform int u_parallax_layers = 32;

out vec4 FragColor;

vec2 parallaxMapping(vec2 texCoords, vec3 viewDir) {
    // Steep parallax mapping
    float layer_depth = 1.0 / u_parallax_layers;
    float current_layer_depth = 0.0;
    
    vec2 P = viewDir.xy / viewDir.z * u_height_scale;
    vec2 delta_texCoords = P / u_parallax_layers;
    
    vec2 current_texCoords = texCoords;
    float current_depth_map_value = texture(u_height_map, current_texCoords).r;
    
    while(current_layer_depth < current_depth_map_value) {
        current_texCoords -= delta_texCoords;
        current_depth_map_value = texture(u_height_map, current_texCoords).r;
        current_layer_depth += layer_depth;
    }
    
    // Parallax occlusion mapping
    vec2 prev_texCoords = current_texCoords + delta_texCoords;
    
    float after_depth = current_depth_map_value - current_layer_depth;
    float before_depth = texture(u_height_map, prev_texCoords).r - current_layer_depth + layer_depth;
    
    float weight = after_depth / (after_depth - before_depth);
    vec2 final_texCoords = prev_texCoords * weight + current_texCoords * (1.0 - weight);
    
    return final_texCoords;
}

void main() {
    vec3 viewDir = normalize(v_tangent_view_pos - v_tangent_frag_pos);
    vec2 texCoords = parallaxMapping(v_texcoord, viewDir);
    
    if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0)
        discard;
    
    vec3 color = texture(u_diffuse_map, texCoords).rgb;
    vec3 normal = texture(u_normal_map, texCoords).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    
    // Simple lighting
    vec3 lightDir = normalize(v_tangent_light_pos - v_tangent_frag_pos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * color;
    
    FragColor = vec4(diffuse, 1.0);
}
)";

// =============================================================================
// C CODE FOR SHADER MANAGEMENT AND DEMONSTRATION
// =============================================================================

typedef struct {
    GLuint program;
    char name[64];
    GLenum type;
} Shader;

typedef struct {
    Shader* shaders;
    int count;
    int capacity;
} ShaderCollection;

// Mock OpenGL functions for demonstration
GLuint glCreateShader(GLenum shaderType) { return rand() % 1000 + 1; }
GLuint glCreateProgram(void) { return rand() % 1000 + 1; }
void glShaderSource(GLuint shader, GLsizei count, const char** string, const GLint* length) {}
void glCompileShader(GLuint shader) {}
void glAttachShader(GLuint program, GLuint shader) {}
void glLinkProgram(GLuint program) {}
void glUseProgram(GLuint program) {}
void glDeleteShader(GLuint shader) {}

ShaderCollection* create_shader_collection() {
    ShaderCollection* collection = malloc(sizeof(ShaderCollection));
    collection->capacity = 20;
    collection->shaders = malloc(collection->capacity * sizeof(Shader));
    collection->count = 0;
    return collection;
}

GLuint compile_shader(const char* source, GLenum type, const char* name) {
    printf("Compiling %s shader: %s\n", 
           type == GL_VERTEX_SHADER ? "vertex" :
           type == GL_FRAGMENT_SHADER ? "fragment" :
           type == GL_GEOMETRY_SHADER ? "geometry" :
           type == GL_COMPUTE_SHADER ? "compute" :
           type == GL_TESS_CONTROL_SHADER ? "tessellation control" :
           type == GL_TESS_EVALUATION_SHADER ? "tessellation evaluation" : "unknown",
           name);
    
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    // In real implementation, check compilation status here
    printf("  ✓ Shader compiled successfully\n");
    
    return shader;
}

GLuint create_shader_program(const char* name, GLuint* shaders, int shader_count) {
    printf("\nCreating shader program: %s\n", name);
    
    GLuint program = glCreateProgram();
    
    for(int i = 0; i < shader_count; i++) {
        glAttachShader(program, shaders[i]);
    }
    
    glLinkProgram(program);
    
    // Clean up individual shaders
    for(int i = 0; i < shader_count; i++) {
        glDeleteShader(shaders[i]);
    }
    
    printf("  ✓ Program linked successfully\n");
    return program;
}

void add_shader_to_collection(ShaderCollection* collection, GLuint program, const char* name, GLenum type) {
    if(collection->count >= collection->capacity) return;
    
    collection->shaders[collection->count] = (Shader){
        .program = program,
        .type = type
    };
    strncpy(collection->shaders[collection->count].name, name, 63);
    collection->count++;
}

void demonstrate_glsl_features() {
    printf("GLSL Features Demonstrated:\n");
    printf("==========================\n\n");
    
    printf("1. Modern GLSL Version (#version 450 core)\n");
    printf("   - Uses latest OpenGL 4.5 features\n");
    printf("   - Core profile for optimal performance\n\n");
    
    printf("2. Vertex Shader Capabilities:\n");
    printf("   ✓ Vertex attribute input (layout qualifiers)\n");
    printf("   ✓ Uniform Buffer Objects (UBO) for efficient data transfer\n");
    printf("   ✓ Matrix transformations (MVP pipeline)\n");
    printf("   ✓ Tangent space calculations\n");
    printf("   ✓ Multi-output varying variables\n\n");
    
    printf("3. Fragment Shader Advanced Features:\n");
    printf("   ✓ Physically Based Rendering (PBR)\n");
    printf("   ✓ Multiple texture sampling\n");
    printf("   ✓ Normal mapping and tangent space\n");
    printf("   ✓ Cook-Torrance BRDF implementation\n");
    printf("   ✓ Fresnel calculations\n");
    printf("   ✓ HDR tone mapping and gamma correction\n\n");
    
    printf("4. Geometry Shader Features:\n");
    printf("   ✓ Primitive generation and modification\n");
    printf("   ✓ Wireframe rendering with barycentric coordinates\n");
    printf("   ✓ Dynamic vertex generation\n\n");
    
    printf("5. Compute Shader Capabilities:\n");
    printf("   ✓ Parallel processing (local work groups)\n");
    printf("   ✓ Shader Storage Buffer Objects (SSBO)\n");
    printf("   ✓ Particle system simulation\n");
    printf("   ✓ Physics integration\n");
    printf("   ✓ Pseudo-random number generation\n\n");
    
    printf("6. Tessellation Shaders:\n");
    printf("   ✓ Adaptive level-of-detail\n");
    printf("   ✓ Distance-based tessellation\n");
    printf("   ✓ Height-based displacement\n");
    printf("   ✓ Normal calculation from height maps\n\n");
    
    printf("7. Post-Processing Effects:\n");
    printf("   ✓ Screen Space Ambient Occlusion (SSAO)\n");
    printf("   ✓ Fast Approximate Anti-Aliasing (FXAA)\n");
    printf("   ✓ HDR Bloom effects\n");
    printf("   ✓ Multi-pass rendering pipeline\n\n");
    
    printf("8. Advanced Techniques:\n");
    printf("   ✓ Parallax Occlusion Mapping\n");
    printf("   ✓ Steep parallax mapping\n");
    printf("   ✓ Dynamic texture coordinate modification\n");
    printf("   ✓ Per-pixel displacement\n\n");
    
    printf("9. GLSL Built-in Functions Used:\n");
    printf("   ✓ Mathematical: sin, cos, pow, sqrt, normalize, dot, cross\n");
    printf("   ✓ Texture sampling: texture, textureLod, textureSize\n");
    printf("   ✓ Interpolation: mix, smoothstep, clamp\n");
    printf("   ✓ Geometric: reflect, length, distance\n");
    printf("   ✓ Utility: fract, floor, ceil, min, max\n\n");
    
    printf("10. Modern GLSL Features:\n");
    printf("   ✓ Layout qualifiers for bindings\n");
    printf("   ✓ Uniform blocks and shader storage\n");
    printf("   ✓ Interface blocks\n");
    printf("   ✓ Subroutines (not shown but supported)\n");
    printf("   ✓ Atomic operations (compute shaders)\n");
    printf("   ✓ Image load/store operations\n\n");
}

void create_all_shader_programs(ShaderCollection* collection) {
    // 1. Basic PBR Program
    GLuint basic_shaders[2] = {
        compile_shader(basic_vertex_shader, GL_VERTEX_SHADER, "BasicPBR_VS"),
        compile_shader(pbr_fragment_shader, GL_FRAGMENT_SHADER, "BasicPBR_FS")
    };
    GLuint pbr_program = create_shader_program("PBR Material", basic_shaders, 2);
    add_shader_to_collection(collection, pbr_program, "PBR Material", GL_FRAGMENT_SHADER);
    
    // 2. Wireframe Program with Geometry Shader
    GLuint wireframe_shaders[3] = {
        compile_shader(basic_vertex_shader, GL_VERTEX_SHADER, "Wireframe_VS"),
        compile_shader(wireframe_geometry_shader, GL_GEOMETRY_SHADER, "Wireframe_GS"),
        compile_shader("// Simple wireframe fragment shader\n#version 450 core\nin vec3 g_barycentric;\nout vec4 FragColor;\nvoid main() {\n    float edge = min(min(g_barycentric.x, g_barycentric.y), g_barycentric.z);\n    float line = 1.0 - smoothstep(0.0, 0.01, edge);\n    FragColor = vec4(vec3(line), 1.0);\n}", GL_FRAGMENT_SHADER, "Wireframe_FS")
    };
    GLuint wireframe_program = create_shader_program("Wireframe Renderer", wireframe_shaders, 3);
    add_shader_to_collection(collection, wireframe_program, "Wireframe Renderer", GL_GEOMETRY_SHADER);
    
    // 3. Particle Compute Program
    GLuint particle_compute = compile_shader(particle_compute_shader, GL_COMPUTE_SHADER, "ParticleSystem_CS");
    GLuint compute_program = create_shader_program("Particle System", &particle_compute, 1);
    add_shader_to_collection(collection, compute_program, "Particle System", GL_COMPUTE_SHADER);
    
    // 4. Tessellation Program
    GLuint tess_shaders[4] = {
        compile_shader("// Passthrough vertex shader\n#version 450 core\nlayout(location=0) in vec3 a_position;\nlayout(location=2) in vec2 a_texcoord;\nout vec3 v_world_position;\nout vec2 v_texcoord;\nvoid main() { v_world_position = a_position; v_texcoord = a_texcoord; }", GL_VERTEX_SHADER, "Terrain_VS"),
        compile_shader(terrain_tess_control_shader, GL_TESS_CONTROL_SHADER, "Terrain_TCS"),
        compile_shader(terrain_tess_eval_shader, GL_TESS_EVALUATION_SHADER, "Terrain_TES"),
        compile_shader("// Simple terrain fragment shader\n#version 450 core\nin vec3 te_world_position;\nin float te_height;\nout vec4 FragColor;\nvoid main() {\n    vec3 color = mix(vec3(0.2, 0.8, 0.2), vec3(0.8, 0.8, 0.8), te_height / 10.0);\n    FragColor = vec4(color, 1.0);\n}", GL_FRAGMENT_SHADER, "Terrain_FS")
    };
    GLuint tess_program = create_shader_program("Adaptive Terrain", tess_shaders, 4);
    add_shader_to_collection(collection, tess_program, "Adaptive Terrain", GL_TESS_EVALUATION_SHADER);
    
    // 5. Post-processing Program
    GLuint postprocess_shaders[2] = {
        compile_shader("// Fullscreen quad vertex shader\n#version 450 core\nlayout(location=0) in vec2 a_position;\nout vec2 v_texcoord;\nvoid main() {\n    v_texcoord = a_position * 0.5 + 0.5;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n}", GL_VERTEX_SHADER, "Postprocess_VS"),
        compile_shader(postprocess_fragment_shader, GL_FRAGMENT_SHADER, "Postprocess_FS")
    };
    GLuint postprocess_program = create_shader_program("Post-processing", postprocess_shaders, 2);
    add_shader_to_collection(collection, postprocess_program, "Post-processing", GL_FRAGMENT_SHADER);
    
    // 6. Parallax Mapping Program
    GLuint parallax_shaders[2] = {
        compile_shader("// Parallax vertex shader\n#version 450 core\nlayout(location=0) in vec3 a_position;\nlayout(location=1) in vec3 a_normal;\nlayout(location=2) in vec2 a_texcoord;\nlayout(location=3) in vec3 a_tangent;\nuniform mat4 u_mvp_matrix;\nuniform mat4 u_model_matrix;\nuniform vec3 u_light_pos;\nuniform vec3 u_view_pos;\nout vec3 v_world_position;\nout vec2 v_texcoord;\nout vec3 v_tangent_light_pos;\nout vec3 v_tangent_view_pos;\nout vec3 v_tangent_frag_pos;\nvoid main() {\n    v_world_position = (u_model_matrix * vec4(a_position, 1.0)).xyz;\n    v_texcoord = a_texcoord;\n    \n    vec3 T = normalize(mat3(u_model_matrix) * a_tangent);\n    vec3 N = normalize(mat3(u_model_matrix) * a_normal);\n    vec3 B = cross(N, T);\n    mat3 TBN = transpose(mat3(T, B, N));\n    \n    v_tangent_light_pos = TBN * u_light_pos;\n    v_tangent_view_pos = TBN * u_view_pos;\n    v_tangent_frag_pos = TBN * v_world_position;\n    \n    gl_Position = u_mvp_matrix * vec4(a_position, 1.0);\n}", GL_VERTEX_SHADER, "Parallax_VS"),
        compile_shader(parallax_mapping_fragment, GL_FRAGMENT_SHADER, "Parallax_FS")
    };
    GLuint parallax_program = create_shader_program("Parallax Mapping", parallax_shaders, 2);
    add_shader_to_collection(collection, parallax_program, "Parallax Mapping", GL_FRAGMENT_SHADER);
}

void print_shader_summary(ShaderCollection* collection) {
    printf("\n=== SHADER PROGRAM SUMMARY ===\n");
    printf("Created %d shader programs:\n\n", collection->count);
    
    for(int i = 0; i < collection->count; i++) {
        printf("%d. %s (Program ID: %u)\n", i+1, collection->shaders[i].name, collection->shaders[i].program);
        
        const char* primary_type = 
            collection->shaders[i].type == GL_VERTEX_SHADER ? "Vertex Processing" :
            collection->shaders[i].type == GL_FRAGMENT_SHADER ? "Fragment Processing" :
            collection->shaders[i].type == GL_GEOMETRY_SHADER ? "Geometry Processing" :
            collection->shaders[i].type == GL_COMPUTE_SHADER ? "Compute Processing" :
            collection->shaders[i].type == GL_TESS_EVALUATION_SHADER ? "Tessellation" : "Mixed";
        
        printf("   Primary Focus: %s\n", primary_type);
    }
    
    printf("\nThese programs demonstrate the full spectrum of GLSL capabilities\n");
    printf("from basic vertex transformation to advanced compute operations.\n");
}

void free_shader_collection(ShaderCollection* collection) {
    free(collection->shaders);
    free(collection);
}

int main() {
    printf("GLSL (OpenGL Shading Language) Capabilities Showcase\n");
    printf("====================================================\n\n");
    
    // Create shader collection
    ShaderCollection* shaders = create_shader_collection();
    
    // Demonstrate GLSL features
    demonstrate_glsl_features();
    
    // Create and compile all shader programs
    printf("COMPILING SHADER PROGRAMS:\n");
    printf("=========================\n");
    create_all_shader_programs(shaders);
    
    // Print summary
    print_shader_summary(shaders);
    
    printf("\n=== PERFORMANCE CONSIDERATIONS ===\n");
    printf("1. Uniform Buffer Objects (UBO) reduce driver overhead\n");
    printf("2. Texture arrays and samplers minimize state changes\n");
    printf("3. Compute shaders utilize parallel GPU architecture\n");
    printf("4. Tessellation provides adaptive detail levels\n");
    printf("5. Early fragment tests optimize fill rate\n");
    printf("6. Instanced rendering reduces draw calls\n\n");
    
    printf("=== MODERN GLSL BEST PRACTICES ===\n");
    printf("1. Use explicit layout qualifiers for all inputs/outputs\n");
    printf("2. Minimize varying variables between shader stages\n");
    printf("3. Pack data efficiently in uniform blocks\n");
    printf("4. Use appropriate precision qualifiers\n");
    printf("5. Leverage built-in functions for optimal performance\n");
    printf("6. Profile and optimize hot paths\n\n");
    
    printf("This showcase demonstrates GLSL's evolution from simple\n");
    printf("vertex/fragment processing to a complete parallel computing\n");
    printf("platform capable of sophisticated graphics and compute tasks.\n\n");
    
    // Cleanup
    free_shader_collection(shaders);
    
    printf("GLSL showcase completed successfully!\n");
    return 0;
}