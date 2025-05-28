#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define MAX_VERTEX_ATTRIBUTES 16
#define MAX_UNIFORMS 32
#define MAX_VERTICES 10000
#define PI 3.14159265359f

// Attribute types
typedef enum {
    ATTR_FLOAT,
    ATTR_VEC2,
    ATTR_VEC3,
    ATTR_VEC4,
    ATTR_MAT4
} AttributeType;

// Uniform types
typedef enum {
    UNIFORM_FLOAT,
    UNIFORM_VEC2,
    UNIFORM_VEC3,
    UNIFORM_VEC4,
    UNIFORM_MAT4,
    UNIFORM_INT,
    UNIFORM_BOOL
} UniformType;

// Vector and matrix structures
typedef struct { float x, y; } Vec2;
typedef struct { float x, y, z; } Vec3;
typedef struct { float x, y, z, w; } Vec4;
typedef struct { float m[16]; } Mat4;

// Color structure
typedef struct { uint8_t r, g, b, a; } Color;

// Vertex attribute structure
typedef struct {
    AttributeType type;
    char name[32];
    union {
        float f;
        Vec2 vec2;
        Vec3 vec3;
        Vec4 vec4;
        Mat4 mat4;
    } value;
} VertexAttribute;

// Uniform structure
typedef struct {
    UniformType type;
    char name[32];
    union {
        float f;
        Vec2 vec2;
        Vec3 vec3;
        Vec4 vec4;
        Mat4 mat4;
        int i;
        bool b;
    } value;
} Uniform;

// Input vertex data
typedef struct {
    VertexAttribute attributes[MAX_VERTEX_ATTRIBUTES];
    int attribute_count;
} InputVertex;

// Output vertex data (after vertex shader)
typedef struct {
    Vec4 position;          // gl_Position equivalent
    VertexAttribute varyings[MAX_VERTEX_ATTRIBUTES]; // Interpolated attributes
    int varying_count;
} OutputVertex;

// Shader context (uniforms and built-in variables)
typedef struct {
    Uniform uniforms[MAX_UNIFORMS];
    int uniform_count;
    
    // Built-in uniforms
    Mat4 model_matrix;
    Mat4 view_matrix;
    Mat4 projection_matrix;
    Mat4 normal_matrix;
    Vec3 camera_position;
    float time;
    
    // Built-in vertex shader outputs
    Vec4 gl_Position;
    float gl_PointSize;
} ShaderContext;

// Vertex shader function pointer
typedef void (*VertexShaderFunc)(const InputVertex* input, OutputVertex* output, ShaderContext* context);

// Vertex shader program
typedef struct {
    char name[64];
    VertexShaderFunc shader_func;
    char* source_code; // For debugging/display
} VertexShaderProgram;

// Mesh data
typedef struct {
    InputVertex* vertices;
    uint32_t* indices;
    int vertex_count;
    int index_count;
} Mesh;

// Rendering pipeline state
typedef struct {
    VertexShaderProgram* current_shader;
    ShaderContext context;
    OutputVertex* transformed_vertices;
    int transformed_vertex_count;
} Pipeline;

// Vector operations
Vec2 vec2_add(Vec2 a, Vec2 b) { return (Vec2){a.x + b.x, a.y + b.y}; }
Vec3 vec3_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec4 vec4_add(Vec4 a, Vec4 b) { return (Vec4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }

Vec2 vec2_mul(Vec2 v, float s) { return (Vec2){v.x * s, v.y * s}; }
Vec3 vec3_mul(Vec3 v, float s) { return (Vec3){v.x * s, v.y * s, v.z * s}; }
Vec4 vec4_mul(Vec4 v, float s) { return (Vec4){v.x * s, v.y * s, v.z * s, v.w * s}; }

float vec3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float vec3_length(Vec3 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    return len > 0.0001f ? vec3_mul(v, 1.0f / len) : (Vec3){0, 0, 0};
}

// Matrix operations
Mat4 mat4_identity() {
    Mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

Mat4 mat4_translate(Vec3 translation) {
    Mat4 m = mat4_identity();
    m.m[12] = translation.x;
    m.m[13] = translation.y;
    m.m[14] = translation.z;
    return m;
}

Mat4 mat4_rotate_y(float angle) {
    Mat4 m = mat4_identity();
    float c = cosf(angle), s = sinf(angle);
    m.m[0] = c;   m.m[2] = s;
    m.m[8] = -s;  m.m[10] = c;
    return m;
}

Mat4 mat4_scale(Vec3 scale) {
    Mat4 m = mat4_identity();
    m.m[0] = scale.x;
    m.m[5] = scale.y;
    m.m[10] = scale.z;
    return m;
}

Mat4 mat4_perspective(float fovy, float aspect, float near_plane, float far_plane) {
    float f = 1.0f / tanf(fovy * 0.5f);
    Mat4 m = {0};
    m.m[0] = f / aspect;
    m.m[5] = f;
    m.m[10] = (far_plane + near_plane) / (near_plane - far_plane);
    m.m[11] = -1.0f;
    m.m[14] = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    return m;
}

Mat4 mat4_look_at(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = vec3_normalize((Vec3){target.x - eye.x, target.y - eye.y, target.z - eye.z});
    Vec3 s = vec3_normalize(vec3_cross(f, up));
    Vec3 u = vec3_cross(s, f);
    
    Mat4 m = mat4_identity();
    m.m[0] = s.x;   m.m[4] = s.y;   m.m[8] = s.z;    m.m[12] = -vec3_dot(s, eye);
    m.m[1] = u.x;   m.m[5] = u.y;   m.m[9] = u.z;    m.m[13] = -vec3_dot(u, eye);
    m.m[2] = -f.x;  m.m[6] = -f.y;  m.m[10] = -f.z;  m.m[14] = vec3_dot(f, eye);
    return m;
}

Vec4 mat4_mul_vec4(Mat4 m, Vec4 v) {
    return (Vec4){
        m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w
    };
}

Mat4 mat4_mul(Mat4 a, Mat4 b) {
    Mat4 result = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                result.m[i * 4 + j] += a.m[i * 4 + k] * b.m[k * 4 + j];
            }
        }
    }
    return result;
}

// Helper functions for shader context
Vec3 get_uniform_vec3(ShaderContext* ctx, const char* name) {
    for (int i = 0; i < ctx->uniform_count; i++) {
        if (strcmp(ctx->uniforms[i].name, name) == 0 && ctx->uniforms[i].type == UNIFORM_VEC3) {
            return ctx->uniforms[i].value.vec3;
        }
    }
    return (Vec3){0, 0, 0};
}

float get_uniform_float(ShaderContext* ctx, const char* name) {
    for (int i = 0; i < ctx->uniform_count; i++) {
        if (strcmp(ctx->uniforms[i].name, name) == 0 && ctx->uniforms[i].type == UNIFORM_FLOAT) {
            return ctx->uniforms[i].value.f;
        }
    }
    return 0.0f;
}

Vec3 get_attribute_vec3(const InputVertex* vertex, const char* name) {
    for (int i = 0; i < vertex->attribute_count; i++) {
        if (strcmp(vertex->attributes[i].name, name) == 0 && vertex->attributes[i].type == ATTR_VEC3) {
            return vertex->attributes[i].value.vec3;
        }
    }
    return (Vec3){0, 0, 0};
}

Vec2 get_attribute_vec2(const InputVertex* vertex, const char* name) {
    for (int i = 0; i < vertex->attribute_count; i++) {
        if (strcmp(vertex->attributes[i].name, name) == 0 && vertex->attributes[i].type == ATTR_VEC2) {
            return vertex->attributes[i].value.vec2;
        }
    }
    return (Vec2){0, 0};
}

void set_varying_vec3(OutputVertex* output, const char* name, Vec3 value) {
    output->varyings[output->varying_count] = (VertexAttribute){
        .type = ATTR_VEC3,
        .value.vec3 = value
    };
    strcpy(output->varyings[output->varying_count].name, name);
    output->varying_count++;
}

void set_varying_vec2(OutputVertex* output, const char* name, Vec2 value) {
    output->varyings[output->varying_count] = (VertexAttribute){
        .type = ATTR_VEC2,
        .value.vec2 = value
    };
    strcpy(output->varyings[output->varying_count].name, name);
    output->varying_count++;
}

// Basic vertex shader - MVP transform only
void basic_vertex_shader(const InputVertex* input, OutputVertex* output, ShaderContext* context) {
    // Get vertex position
    Vec3 position = get_attribute_vec3(input, "position");
    
    // Transform through MVP matrix
    Mat4 mvp = mat4_mul(context->projection_matrix, 
               mat4_mul(context->view_matrix, context->model_matrix));
    
    Vec4 world_pos = {position.x, position.y, position.z, 1.0f};
    context->gl_Position = mat4_mul_vec4(mvp, world_pos);
    output->position = context->gl_Position;
    
    // Pass through texture coordinates if present
    Vec2 texcoord = get_attribute_vec2(input, "texcoord");
    set_varying_vec2(output, "v_texcoord", texcoord);
    
    output->varying_count = 1;
}

// Phong lighting vertex shader
void phong_vertex_shader(const InputVertex* input, OutputVertex* output, ShaderContext* context) {
    // Get vertex attributes
    Vec3 position = get_attribute_vec3(input, "position");
    Vec3 normal = get_attribute_vec3(input, "normal");
    Vec2 texcoord = get_attribute_vec2(input, "texcoord");
    
    // Transform position
    Vec4 world_pos_4 = mat4_mul_vec4(context->model_matrix, (Vec4){position.x, position.y, position.z, 1.0f});
    Vec3 world_pos = {world_pos_4.x, world_pos_4.y, world_pos_4.z};
    
    Mat4 mvp = mat4_mul(context->projection_matrix, 
               mat4_mul(context->view_matrix, context->model_matrix));
    context->gl_Position = mat4_mul_vec4(mvp, (Vec4){position.x, position.y, position.z, 1.0f});
    output->position = context->gl_Position;
    
    // Transform normal (should use normal matrix in practice)
    Vec4 world_normal_4 = mat4_mul_vec4(context->model_matrix, (Vec4){normal.x, normal.y, normal.z, 0.0f});
    Vec3 world_normal = vec3_normalize((Vec3){world_normal_4.x, world_normal_4.y, world_normal_4.z});
    
    // Calculate lighting vectors
    Vec3 light_pos = get_uniform_vec3(context, "light_position");
    Vec3 light_dir = vec3_normalize((Vec3){
        light_pos.x - world_pos.x,
        light_pos.y - world_pos.y,
        light_pos.z - world_pos.z
    });
    
    Vec3 view_dir = vec3_normalize((Vec3){
        context->camera_position.x - world_pos.x,
        context->camera_position.y - world_pos.y,
        context->camera_position.z - world_pos.z
    });
    
    // Calculate diffuse lighting
    float diffuse = fmaxf(0.0f, vec3_dot(world_normal, light_dir));
    
    // Calculate specular lighting (Blinn-Phong)
    Vec3 half_dir = vec3_normalize((Vec3){
        light_dir.x + view_dir.x,
        light_dir.y + view_dir.y,
        light_dir.z + view_dir.z
    });
    float spec_power = get_uniform_float(context, "specular_power");
    float specular = powf(fmaxf(0.0f, vec3_dot(world_normal, half_dir)), spec_power);
    
    // Set varyings
    set_varying_vec3(output, "v_world_pos", world_pos);
    set_varying_vec3(output, "v_normal", world_normal);
    set_varying_vec2(output, "v_texcoord", texcoord);
    set_varying_vec3(output, "v_light_dir", light_dir);
    set_varying_vec3(output, "v_view_dir", view_dir);
    
    // Pre-calculated lighting (Gouraud shading alternative)
    Vec3 light_color = get_uniform_vec3(context, "light_color");
    Vec3 ambient = vec3_mul(light_color, 0.1f);
    Vec3 diffuse_color = vec3_mul(light_color, diffuse);
    Vec3 specular_color = vec3_mul(light_color, specular);
    
    Vec3 final_color = vec3_add(vec3_add(ambient, diffuse_color), specular_color);
    set_varying_vec3(output, "v_color", final_color);
}

// Animated vertex shader with wave deformation
void wave_vertex_shader(const InputVertex* input, OutputVertex* output, ShaderContext* context) {
    Vec3 position = get_attribute_vec3(input, "position");
    Vec2 texcoord = get_attribute_vec2(input, "texcoord");
    
    // Apply wave deformation
    float wave_amplitude = get_uniform_float(context, "wave_amplitude");
    float wave_frequency = get_uniform_float(context, "wave_frequency");
    float wave_speed = get_uniform_float(context, "wave_speed");
    
    float wave_offset = sinf(position.x * wave_frequency + context->time * wave_speed) * wave_amplitude;
    position.y += wave_offset;
    
    // Standard transformation
    Mat4 mvp = mat4_mul(context->projection_matrix, 
               mat4_mul(context->view_matrix, context->model_matrix));
    context->gl_Position = mat4_mul_vec4(mvp, (Vec4){position.x, position.y, position.z, 1.0f});
    output->position = context->gl_Position;
    
    // Pass modified position and original texcoord
    set_varying_vec3(output, "v_world_pos", position);
    set_varying_vec2(output, "v_texcoord", texcoord);
    set_varying_vec3(output, "v_wave_offset", (Vec3){0, wave_offset, 0});
}

// Displacement mapping vertex shader
void displacement_vertex_shader(const InputVertex* input, OutputVertex* output, ShaderContext* context) {
    Vec3 position = get_attribute_vec3(input, "position");
    Vec3 normal = get_attribute_vec3(input, "normal");
    Vec2 texcoord = get_attribute_vec2(input, "texcoord");
    
    // Simple procedural displacement (in practice, you'd sample a displacement texture)
    float displacement_scale = get_uniform_float(context, "displacement_scale");
    float noise = sinf(position.x * 5.0f) * cosf(position.z * 5.0f) * sinf(context->time);
    Vec3 displaced_pos = vec3_add(position, vec3_mul(normal, noise * displacement_scale));
    
    // Transform displaced position
    Mat4 mvp = mat4_mul(context->projection_matrix, 
               mat4_mul(context->view_matrix, context->model_matrix));
    context->gl_Position = mat4_mul_vec4(mvp, (Vec4){displaced_pos.x, displaced_pos.y, displaced_pos.z, 1.0f});
    output->position = context->gl_Position;
    
    // Set varyings
    set_varying_vec3(output, "v_world_pos", displaced_pos);
    set_varying_vec3(output, "v_normal", normal);
    set_varying_vec2(output, "v_texcoord", texcoord);
}

// Skeletal animation vertex shader (simplified)
void skinned_vertex_shader(const InputVertex* input, OutputVertex* output, ShaderContext* context) {
    Vec3 position = get_attribute_vec3(input, "position");
    Vec3 normal = get_attribute_vec3(input, "normal");
    
    // In a real implementation, you'd have bone indices and weights as vertex attributes
    // For this demo, we'll simulate simple bone transformation
    float bone_rotation = context->time * 2.0f;
    Mat4 bone_transform = mat4_rotate_y(bone_rotation);
    
    // Apply bone transformation
    Vec4 transformed_pos = mat4_mul_vec4(bone_transform, (Vec4){position.x, position.y, position.z, 1.0f});
    Vec4 transformed_normal = mat4_mul_vec4(bone_transform, (Vec4){normal.x, normal.y, normal.z, 0.0f});
    
    // Standard MVP transformation
    Mat4 mvp = mat4_mul(context->projection_matrix, 
               mat4_mul(context->view_matrix, context->model_matrix));
    context->gl_Position = mat4_mul_vec4(mvp, transformed_pos);
    output->position = context->gl_Position;
    
    // Set varyings
    set_varying_vec3(output, "v_world_pos", (Vec3){transformed_pos.x, transformed_pos.y, transformed_pos.z});
    set_varying_vec3(output, "v_normal", vec3_normalize((Vec3){transformed_normal.x, transformed_normal.y, transformed_normal.z}));
}

// Create vertex shader programs
VertexShaderProgram create_shader_program(const char* name, VertexShaderFunc func, const char* source) {
    VertexShaderProgram program;
    strcpy(program.name, name);
    program.shader_func = func;
    program.source_code = malloc(strlen(source) + 1);
    strcpy(program.source_code, source);
    return program;
}

// Create test mesh (cube)
Mesh create_cube_mesh() {
    Mesh mesh = {0};
    mesh.vertex_count = 8;
    mesh.vertices = malloc(mesh.vertex_count * sizeof(InputVertex));
    
    // Cube vertices
    Vec3 positions[] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
    };
    
    Vec3 normals[] = {
        {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
        {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
    };
    
    Vec2 texcoords[] = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1},
        {0, 0}, {1, 0}, {1, 1}, {0, 1}
    };
    
    for (int i = 0; i < mesh.vertex_count; i++) {
        mesh.vertices[i].attribute_count = 3;
        
        // Position
        mesh.vertices[i].attributes[0] = (VertexAttribute){
            .type = ATTR_VEC3,
            .value.vec3 = vec3_normalize(positions[i]) // Normalize for unit sphere
        };
        strcpy(mesh.vertices[i].attributes[0].name, "position");
        
        // Normal
        mesh.vertices[i].attributes[1] = (VertexAttribute){
            .type = ATTR_VEC3,
            .value.vec3 = vec3_normalize(normals[i])
        };
        strcpy(mesh.vertices[i].attributes[1].name, "normal");
        
        // Texture coordinate
        mesh.vertices[i].attributes[2] = (VertexAttribute){
            .type = ATTR_VEC2,
            .value.vec2 = texcoords[i]
        };
        strcpy(mesh.vertices[i].attributes[2].name, "texcoord");
    }
    
    // Cube indices (triangulated faces)
    uint32_t cube_indices[] = {
        0,1,2, 2,3,0,  // Front
        4,7,6, 6,5,4,  // Back
        0,4,5, 5,1,0,  // Bottom
        2,6,7, 7,3,2,  // Top
        0,3,7, 7,4,0,  // Left
        1,5,6, 6,2,1   // Right
    };
    
    mesh.index_count = 36;
    mesh.indices = malloc(mesh.index_count * sizeof(uint32_t));
    memcpy(mesh.indices, cube_indices, mesh.index_count * sizeof(uint32_t));
    
    return mesh;
}

// Setup shader context with uniforms
void setup_shader_context(ShaderContext* context, float time) {
    // Setup matrices
    context->model_matrix = mat4_mul(mat4_rotate_y(time), mat4_scale((Vec3){1.5f, 1.5f, 1.5f}));
    context->view_matrix = mat4_look_at((Vec3){0, 0, 5}, (Vec3){0, 0, 0}, (Vec3){0, 1, 0});
    context->projection_matrix = mat4_perspective(60.0f * PI / 180.0f, 16.0f/9.0f, 0.1f, 100.0f);
    context->camera_position = (Vec3){0, 0, 5};
    context->time = time;
    
    // Setup uniforms
    context->uniform_count = 6;
    
    context->uniforms[0] = (Uniform){
        .type = UNIFORM_VEC3,
        .value.vec3 = {2, 2, 2}
    };
    strcpy(context->uniforms[0].name, "light_position");
    
    context->uniforms[1] = (Uniform){
        .type = UNIFORM_VEC3,
        .value.vec3 = {1, 1, 1}
    };
    strcpy(context->uniforms[1].name, "light_color");
    
    context->uniforms[2] = (Uniform){
        .type = UNIFORM_FLOAT,
        .value.f = 32.0f
    };
    strcpy(context->uniforms[2].name, "specular_power");
    
    context->uniforms[3] = (Uniform){
        .type = UNIFORM_FLOAT,
        .value.f = 0.2f
    };
    strcpy(context->uniforms[3].name, "wave_amplitude");
    
    context->uniforms[4] = (Uniform){
        .type = UNIFORM_FLOAT,
        .value.f = 3.0f
    };
    strcpy(context->uniforms[4].name, "wave_frequency");
    
    context->uniforms[5] = (Uniform){
        .type = UNIFORM_FLOAT,
        .value.f = 2.0f
    };
    strcpy(context->uniforms[5].name, "wave_speed");
}

// Execute vertex shader on mesh
void process_vertices(Pipeline* pipeline, const Mesh* mesh) {
    pipeline->transformed_vertex_count = mesh->vertex_count;
    pipeline->transformed_vertices = malloc(mesh->vertex_count * sizeof(OutputVertex));
    
    for (int i = 0; i < mesh->vertex_count; i++) {
        pipeline->transformed_vertices[i].varying_count = 0;
        pipeline->current_shader->shader_func(
            &mesh->vertices[i],
            &pipeline->transformed_vertices[i],
            &pipeline->context
        );
    }
}

// Print vertex shader output for debugging
void print_vertex_output(const OutputVertex* vertex, int index) {
    printf("Vertex %d:\n", index);
    printf("  gl_Position: (%.3f, %.3f, %.3f, %.3f)\n", 
           vertex->position.x, vertex->position.y, vertex->position.z, vertex->position.w);
    
    for (int i = 0; i < vertex->varying_count; i++) {
        printf("  %s: ", vertex->varyings[i].name);
        switch (vertex->varyings[i].type) {
            case ATTR_VEC2:
                printf("(%.3f, %.3f)\n", vertex->varyings[i].value.vec2.x, vertex->varyings[i].value.vec2.y);
                break;
            case ATTR_VEC3:
                printf("(%.3f, %.3f, %.3f)\n", 
                       vertex->varyings[i].value.vec3.x, vertex->varyings[i].value.vec3.y, vertex->varyings[i].value.vec3.z);
                break;
            default:
                printf("(unsupported type)\n");
                break;
        }
    }
    printf("\n");
}

// Save transformed vertices to file for analysis
void save_vertex_data(const OutputVertex* vertices, int count, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "# Vertex Shader Output Data\n");
    fprintf(f, "# Format: x y z w (clip space position)\n");
    
    for (int i = 0; i < count; i++) {
        fprintf(f, "%.6f %.6f %.6f %.6f\n", 
                vertices[i].position.x, vertices[i].position.y, 
                vertices[i].position.z, vertices[i].position.w);
    }
    
    fclose(f);
    printf("Saved vertex data to %s\n", filename);
}

// Free resources
void free_mesh(Mesh* mesh) {
    free(mesh->vertices);
    free(mesh->indices);
}

void free_pipeline(Pipeline* pipeline) {
    free(pipeline->transformed_vertices);
}

int main() {
    printf("Vertex Shader Implementation Demo\n");
    printf("================================\n");
    
    // Create shader programs
    VertexShaderProgram shaders[] = {
        create_shader_program("Basic", basic_vertex_shader,
            "// Basic vertex shader\n"
            "in vec3 position;\n"
            "in vec2 texcoord;\n"
            "uniform mat4 mvp_matrix;\n"
            "out vec2 v_texcoord;\n"
            "void main() {\n"
            "    gl_Position = mvp_matrix * vec4(position, 1.0);\n"
            "    v_texcoord = texcoord;\n"
            "}"),
        
        create_shader_program("Phong", phong_vertex_shader,
            "// Phong lighting vertex shader\n"
            "in vec3 position;\n"
            "in vec3 normal;\n"
            "in vec2 texcoord;\n"
            "uniform mat4 model_matrix;\n"
            "uniform mat4 view_matrix;\n"
            "uniform mat4 projection_matrix;\n"
            "uniform vec3 light_position;\n"
            "uniform vec3 camera_position;\n"
            "out vec3 v_world_pos;\n"
            "out vec3 v_normal;\n"
            "out vec2 v_texcoord;\n"
            "void main() {\n"
            "    vec4 world_pos = model_matrix * vec4(position, 1.0);\n"
            "    gl_Position = projection_matrix * view_matrix * world_pos;\n"
            "    v_world_pos = world_pos.xyz;\n"
            "    v_normal = (model_matrix * vec4(normal, 0.0)).xyz;\n"
            "    v_texcoord = texcoord;\n"
            "}"),
        
        create_shader_program("Wave", wave_vertex_shader,
            "// Animated wave vertex shader\n"
            "in vec3 position;\n"
            "in vec2 texcoord;\n"
            "uniform mat4 mvp_matrix;\n"
            "uniform float time;\n"
            "uniform float wave_amplitude;\n"
            "uniform float wave_frequency;\n"
            "uniform float wave_speed;\n"
            "out vec2 v_texcoord;\n"
            "out vec3 v_wave_offset;\n"
            "void main() {\n"
            "    vec3 pos = position;\n"
            "    float wave = sin(pos.x * wave_frequency + time * wave_speed);\n"
            "    pos.y += wave * wave_amplitude;\n"
            "    gl_Position = mvp_matrix * vec4(pos, 1.0);\n"
            "    v_texcoord = texcoord;\n"
            "    v_wave_offset = vec3(0, wave * wave_amplitude, 0);\n"
            "}")
    };
    
    int shader_count = 3;
    
    // Create test mesh
    Mesh cube_mesh = create_cube_mesh();
    printf("Created cube mesh with %d vertices\n", cube_mesh.vertex_count);
    
    // Create pipeline
    Pipeline pipeline = {0};
    
    // Test each shader
    for (int s = 0; s < shader_count; s++) {
        printf("\n--- Testing %s Vertex Shader ---\n", shaders[s].name);
        printf("Source code:\n%s\n", shaders[s].source_code);
        
        pipeline.current_shader = &shaders[s];
        
        // Test with different time values
        for (int frame = 0; frame < 3; frame++) {
            float time = frame * 0.5f;
            setup_shader_context(&pipeline.context, time);
            
            printf("\nFrame %d (time=%.1f):\n", frame, time);
            
            // Process vertices
            process_vertices(&pipeline, &cube_mesh);
            
            // Print first few vertices
            for (int i = 0; i < 3 && i < pipeline.transformed_vertex_count; i++) {
                print_vertex_output(&pipeline.transformed_vertices[i], i);
            }
            
            // Save vertex data for this frame
            char filename[64];
            snprintf(filename, sizeof(filename), "vertices_%s_frame%d.txt", shaders[s].name, frame);
            save_vertex_data(pipeline.transformed_vertices, pipeline.transformed_vertex_count, filename);
            
            free(pipeline.transformed_vertices);
        }
    }
    
    // Performance test
    printf("\n--- Performance Test ---\n");
    pipeline.current_shader = &shaders[1]; // Use Phong shader
    setup_shader_context(&pipeline.context, 0.0f);
    
    // Create larger mesh for performance testing
    const int perf_vertex_count = 10000;
    printf("Testing performance with %d vertices...\n", perf_vertex_count);
    
    // Simulate large mesh by replicating cube data
    Mesh large_mesh = {0};
    large_mesh.vertex_count = perf_vertex_count;
    large_mesh.vertices = malloc(large_mesh.vertex_count * sizeof(InputVertex));
    
    for (int i = 0; i < large_mesh.vertex_count; i++) {
        large_mesh.vertices[i] = cube_mesh.vertices[i % cube_mesh.vertex_count];
        // Add some variation
        large_mesh.vertices[i].attributes[0].value.vec3.x += (i % 100) * 0.1f;
        large_mesh.vertices[i].attributes[0].value.vec3.y += (i % 50) * 0.1f;
    }
    
    // Time the vertex processing
    clock_t start = clock();
    process_vertices(&pipeline, &large_mesh);
    clock_t end = clock();
    
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Processed %d vertices in %.3f seconds\n", perf_vertex_count, cpu_time);
    printf("Rate: %.1f vertices/second\n", perf_vertex_count / cpu_time);
    
    // Cleanup
    for (int i = 0; i < shader_count; i++) {
        free(shaders[i].source_code);
    }
    free_mesh(&cube_mesh);
    free_mesh(&large_mesh);
    free_pipeline(&pipeline);
    
    printf("\nVertex shader demo completed!\n");
    printf("Generated output files with vertex transformation data.\n");
    
    return 0;
}