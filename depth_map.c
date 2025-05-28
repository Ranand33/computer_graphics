#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define PI 3.14159265359f
#define SHADOW_MAP_SIZE 1024
#define MAX_CASCADES 4

// Basic math structures
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float x, y;
} Vec2;

typedef struct {
    float m[16];
} Mat4;

typedef struct {
    float left, right, bottom, top, near, far;
} Frustum;

// Vertex structure
typedef struct {
    Vec3 position;
    Vec3 normal;
    Vec2 texcoord;
} Vertex;

// Triangle structure
typedef struct {
    Vertex v[3];
} Triangle;

// Shadow map types
typedef enum {
    SHADOW_MAP_2D,      // For directional and spot lights
    SHADOW_MAP_CUBE,    // For point lights
    SHADOW_MAP_CASCADE  // For cascaded shadow maps
} ShadowMapType;

// Shadow map structure
typedef struct {
    ShadowMapType type;
    int width, height;
    float* depthBuffer;
    Mat4 lightMatrix;      // View-projection matrix from light's perspective
    float near, far;       // Near/far planes for the light
    int face;             // For cube maps (0-5)
} ShadowMap;

// Cube map shadow structure for point lights
typedef struct {
    ShadowMap faces[6];    // Six faces of the cube
    Vec3 position;         // Light position
    float near, far;       // Near/far planes
} CubeShadowMap;

// Cascaded shadow map for large scenes
typedef struct {
    ShadowMap cascades[MAX_CASCADES];
    float splitDistances[MAX_CASCADES + 1];
    int numCascades;
} CascadedShadowMap;

// Light structure with shadow mapping
typedef struct {
    Vec3 position;
    Vec3 direction;
    Vec3 color;
    float intensity;
    int castsShadows;
    void* shadowMap;    // Points to appropriate shadow map type
} Light;

// Camera structure
typedef struct {
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    Vec3 right;
    float fov;
    float aspect;
    float near;
    float far;
} Camera;

// Vector math functions
Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 vec3_mul(Vec3 v, float s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float vec3_length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len > 0.0001f) {
        return vec3_mul(v, 1.0f / len);
    }
    return v;
}

// Matrix operations
Mat4 mat4_identity() {
    Mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

Mat4 mat4_multiply(Mat4 a, Mat4 b) {
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

Mat4 mat4_perspective(float fov, float aspect, float near, float far) {
    Mat4 m = {0};
    float tanHalfFov = tanf(fov * 0.5f * PI / 180.0f);
    
    m.m[0] = 1.0f / (aspect * tanHalfFov);
    m.m[5] = 1.0f / tanHalfFov;
    m.m[10] = -(far + near) / (far - near);
    m.m[11] = -1.0f;
    m.m[14] = -(2.0f * far * near) / (far - near);
    
    return m;
}

Mat4 mat4_orthographic(float left, float right, float bottom, float top, float near, float far) {
    Mat4 m = mat4_identity();
    
    m.m[0] = 2.0f / (right - left);
    m.m[5] = 2.0f / (top - bottom);
    m.m[10] = -2.0f / (far - near);
    m.m[12] = -(right + left) / (right - left);
    m.m[13] = -(top + bottom) / (top - bottom);
    m.m[14] = -(far + near) / (far - near);
    
    return m;
}

Mat4 mat4_lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = vec3_normalize(vec3_sub(center, eye));
    Vec3 s = vec3_normalize(vec3_cross(f, up));
    Vec3 u = vec3_cross(s, f);
    
    Mat4 m = mat4_identity();
    m.m[0] = s.x;
    m.m[4] = s.y;
    m.m[8] = s.z;
    m.m[1] = u.x;
    m.m[5] = u.y;
    m.m[9] = u.z;
    m.m[2] = -f.x;
    m.m[6] = -f.y;
    m.m[10] = -f.z;
    m.m[12] = -vec3_dot(s, eye);
    m.m[13] = -vec3_dot(u, eye);
    m.m[14] = vec3_dot(f, eye);
    
    return m;
}

Vec3 mat4_transform_point(Mat4 m, Vec3 p) {
    float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];
    return (Vec3){
        (m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12]) / w,
        (m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13]) / w,
        (m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14]) / w
    };
}

// Create shadow map
ShadowMap* create_shadow_map(int width, int height, ShadowMapType type) {
    ShadowMap* sm = (ShadowMap*)malloc(sizeof(ShadowMap));
    sm->type = type;
    sm->width = width;
    sm->height = height;
    sm->depthBuffer = (float*)malloc(width * height * sizeof(float));
    sm->near = 0.1f;
    sm->far = 100.0f;
    sm->face = 0;
    
    // Initialize depth buffer to max depth
    for (int i = 0; i < width * height; i++) {
        sm->depthBuffer[i] = 1.0f;
    }
    
    return sm;
}

void destroy_shadow_map(ShadowMap* sm) {
    free(sm->depthBuffer);
    free(sm);
}

// Clear shadow map
void clear_shadow_map(ShadowMap* sm) {
    for (int i = 0; i < sm->width * sm->height; i++) {
        sm->depthBuffer[i] = 1.0f;
    }
}

// Create cube shadow map for point lights
CubeShadowMap* create_cube_shadow_map(int size) {
    CubeShadowMap* csm = (CubeShadowMap*)malloc(sizeof(CubeShadowMap));
    
    // Create six shadow maps for cube faces
    for (int i = 0; i < 6; i++) {
        csm->faces[i] = *create_shadow_map(size, size, SHADOW_MAP_CUBE);
        csm->faces[i].face = i;
    }
    
    csm->near = 0.1f;
    csm->far = 25.0f;
    
    return csm;
}

void destroy_cube_shadow_map(CubeShadowMap* csm) {
    for (int i = 0; i < 6; i++) {
        free(csm->faces[i].depthBuffer);
    }
    free(csm);
}

// Get cube face view matrix
Mat4 get_cube_face_view_matrix(Vec3 position, int face) {
    Vec3 targets[6] = {
        {1, 0, 0},   // +X
        {-1, 0, 0},  // -X
        {0, 1, 0},   // +Y
        {0, -1, 0},  // -Y
        {0, 0, 1},   // +Z
        {0, 0, -1}   // -Z
    };
    
    Vec3 ups[6] = {
        {0, -1, 0},  // +X
        {0, -1, 0},  // -X
        {0, 0, 1},   // +Y
        {0, 0, -1},  // -Y
        {0, -1, 0},  // +Z
        {0, -1, 0}   // -Z
    };
    
    Vec3 target = vec3_add(position, targets[face]);
    return mat4_lookAt(position, target, ups[face]);
}

// Barycentric coordinates
void barycentric(Vec2 p, Vec2 a, Vec2 b, Vec2 c, float* u, float* v, float* w) {
    Vec2 v0 = {b.x - a.x, b.y - a.y};
    Vec2 v1 = {c.x - a.x, c.y - a.y};
    Vec2 v2 = {p.x - a.x, p.y - a.y};
    
    float d00 = v0.x * v0.x + v0.y * v0.y;
    float d01 = v0.x * v1.x + v0.y * v1.y;
    float d11 = v1.x * v1.x + v1.y * v1.y;
    float d20 = v2.x * v0.x + v2.y * v0.y;
    float d21 = v2.x * v1.x + v2.y * v1.y;
    
    float denom = d00 * d11 - d01 * d01;
    *v = (d11 * d20 - d01 * d21) / denom;
    *w = (d00 * d21 - d01 * d20) / denom;
    *u = 1.0f - *v - *w;
}

// Render triangle to shadow map
void render_triangle_to_shadow_map(ShadowMap* sm, Triangle tri, Mat4 lightMatrix) {
    // Transform vertices to light space
    Vec3 v0 = mat4_transform_point(lightMatrix, tri.v[0].position);
    Vec3 v1 = mat4_transform_point(lightMatrix, tri.v[1].position);
    Vec3 v2 = mat4_transform_point(lightMatrix, tri.v[2].position);
    
    // Convert to screen coordinates
    Vec2 v0_screen = {
        (v0.x + 1.0f) * 0.5f * sm->width,
        (v0.y + 1.0f) * 0.5f * sm->height
    };
    Vec2 v1_screen = {
        (v1.x + 1.0f) * 0.5f * sm->width,
        (v1.y + 1.0f) * 0.5f * sm->height
    };
    Vec2 v2_screen = {
        (v2.x + 1.0f) * 0.5f * sm->width,
        (v2.y + 1.0f) * 0.5f * sm->height
    };
    
    // Find bounding box
    float minX = fminf(v0_screen.x, fminf(v1_screen.x, v2_screen.x));
    float minY = fminf(v0_screen.y, fminf(v1_screen.y, v2_screen.y));
    float maxX = fmaxf(v0_screen.x, fmaxf(v1_screen.x, v2_screen.x));
    float maxY = fmaxf(v0_screen.y, fmaxf(v1_screen.y, v2_screen.y));
    
    // Clip to screen
    minX = fmaxf(0, minX);
    minY = fmaxf(0, minY);
    maxX = fminf(sm->width - 1, maxX);
    maxY = fminf(sm->height - 1, maxY);
    
    // Rasterize
    for (int y = (int)minY; y <= (int)maxY; y++) {
        for (int x = (int)minX; x <= (int)maxX; x++) {
            Vec2 p = {x + 0.5f, y + 0.5f};
            
            float u, v, w;
            barycentric(p, v0_screen, v1_screen, v2_screen, &u, &v, &w);
            
            if (u >= 0 && v >= 0 && w >= 0) {
                // Interpolate depth
                float depth = u * v0.z + v * v1.z + w * v2.z;
                
                // Map to [0, 1] range
                depth = (depth + 1.0f) * 0.5f;
                
                int idx = y * sm->width + x;
                if (depth < sm->depthBuffer[idx]) {
                    sm->depthBuffer[idx] = depth;
                }
            }
        }
    }
}

// Generate shadow map for directional light
void generate_directional_shadow_map(ShadowMap* sm, Light* light, Triangle* triangles, 
                                   int numTriangles, Vec3 sceneCenter, float sceneRadius) {
    clear_shadow_map(sm);
    
    // Calculate light view matrix
    Vec3 lightPos = vec3_sub(sceneCenter, vec3_mul(light->direction, sceneRadius * 2));
    Mat4 lightView = mat4_lookAt(lightPos, sceneCenter, (Vec3){0, 1, 0});
    
    // Calculate orthographic projection for directional light
    float orthoSize = sceneRadius * 2;
    Mat4 lightProj = mat4_orthographic(-orthoSize, orthoSize, -orthoSize, orthoSize, 
                                      0.1f, sceneRadius * 4);
    
    // Combine matrices
    sm->lightMatrix = mat4_multiply(lightProj, lightView);
    
    // Render all triangles
    for (int i = 0; i < numTriangles; i++) {
        render_triangle_to_shadow_map(sm, triangles[i], sm->lightMatrix);
    }
}

// Generate shadow map for spot light
void generate_spot_shadow_map(ShadowMap* sm, Light* light, Triangle* triangles, 
                            int numTriangles, float fov) {
    clear_shadow_map(sm);
    
    // Calculate light view matrix
    Vec3 target = vec3_add(light->position, light->direction);
    Mat4 lightView = mat4_lookAt(light->position, target, (Vec3){0, 1, 0});
    
    // Perspective projection for spot light
    Mat4 lightProj = mat4_perspective(fov, 1.0f, sm->near, sm->far);
    
    // Combine matrices
    sm->lightMatrix = mat4_multiply(lightProj, lightView);
    
    // Render all triangles
    for (int i = 0; i < numTriangles; i++) {
        render_triangle_to_shadow_map(sm, triangles[i], sm->lightMatrix);
    }
}

// Generate cube shadow map for point light
void generate_point_shadow_map(CubeShadowMap* csm, Light* light, Triangle* triangles, 
                             int numTriangles) {
    csm->position = light->position;
    
    // Render to each cube face
    for (int face = 0; face < 6; face++) {
        clear_shadow_map(&csm->faces[face]);
        
        // Get view matrix for this face
        Mat4 faceView = get_cube_face_view_matrix(light->position, face);
        
        // 90 degree FOV for cube map faces
        Mat4 faceProj = mat4_perspective(90.0f, 1.0f, csm->near, csm->far);
        
        csm->faces[face].lightMatrix = mat4_multiply(faceProj, faceView);
        
        // Render triangles to this face
        for (int i = 0; i < numTriangles; i++) {
            render_triangle_to_shadow_map(&csm->faces[face], triangles[i], 
                                        csm->faces[face].lightMatrix);
        }
    }
}

// Sample shadow map with PCF (Percentage Closer Filtering)
float sample_shadow_map_pcf(ShadowMap* sm, Vec3 worldPos, int samples) {
    // Transform world position to light space
    Vec3 lightSpacePos = mat4_transform_point(sm->lightMatrix, worldPos);
    
    // Convert to texture coordinates
    float u = (lightSpacePos.x + 1.0f) * 0.5f;
    float v = (lightSpacePos.y + 1.0f) * 0.5f;
    float depth = (lightSpacePos.z + 1.0f) * 0.5f;
    
    // Check bounds
    if (u < 0 || u > 1 || v < 0 || v > 1) {
        return 1.0f; // Not in shadow
    }
    
    float shadow = 0.0f;
    float texelSize = 1.0f / sm->width;
    float bias = 0.005f;
    
    // PCF kernel
    int halfSamples = samples / 2;
    for (int y = -halfSamples; y <= halfSamples; y++) {
        for (int x = -halfSamples; x <= halfSamples; x++) {
            float su = u + x * texelSize;
            float sv = v + y * texelSize;
            
            // Clamp to valid range
            su = fmaxf(0, fminf(1, su));
            sv = fmaxf(0, fminf(1, sv));
            
            int sx = (int)(su * sm->width);
            int sy = (int)(sv * sm->height);
            
            float shadowDepth = sm->depthBuffer[sy * sm->width + sx];
            shadow += (depth - bias > shadowDepth) ? 0.0f : 1.0f;
        }
    }
    
    return shadow / (float)(samples * samples);
}

// Sample cube shadow map
float sample_cube_shadow_map(CubeShadowMap* csm, Vec3 worldPos, Vec3 lightPos) {
    Vec3 lightToFrag = vec3_sub(worldPos, lightPos);
    
    // Determine which cube face to sample
    float absX = fabsf(lightToFrag.x);
    float absY = fabsf(lightToFrag.y);
    float absZ = fabsf(lightToFrag.z);
    
    int face;
    if (absX > absY && absX > absZ) {
        face = (lightToFrag.x > 0) ? 0 : 1;
    } else if (absY > absZ) {
        face = (lightToFrag.y > 0) ? 2 : 3;
    } else {
        face = (lightToFrag.z > 0) ? 4 : 5;
    }
    
    // Sample the appropriate face
    return sample_shadow_map_pcf(&csm->faces[face], worldPos, 3);
}

// Calculate cascaded shadow map splits
void calculate_cascade_splits(float near, float far, int numCascades, float* splits) {
    float lambda = 0.75f; // Blend between linear and logarithmic
    
    for (int i = 0; i <= numCascades; i++) {
        float p = (float)i / numCascades;
        
        // Linear split
        float linear = near + (far - near) * p;
        
        // Logarithmic split
        float log = near * powf(far / near, p);
        
        // Blend
        splits[i] = lambda * log + (1.0f - lambda) * linear;
    }
    
    splits[0] = near;
    splits[numCascades] = far;
}

// Save shadow map as image
void save_shadow_map_ppm(const char* filename, ShadowMap* sm) {
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    fprintf(file, "P6\n%d %d\n255\n", sm->width, sm->height);
    
    for (int i = 0; i < sm->width * sm->height; i++) {
        unsigned char val = (unsigned char)(sm->depthBuffer[i] * 255);
        fputc(val, file);
        fputc(val, file);
        fputc(val, file);
    }
    
    fclose(file);
}

// Save cube shadow map faces
void save_cube_shadow_map_ppm(const char* prefix, CubeShadowMap* csm) {
    char filename[256];
    const char* faceNames[] = {"posX", "negX", "posY", "negY", "posZ", "negZ"};
    
    for (int i = 0; i < 6; i++) {
        sprintf(filename, "%s_%s.ppm", prefix, faceNames[i]);
        save_shadow_map_ppm(filename, &csm->faces[i]);
    }
}

// Example scene rendering with shadows
typedef struct {
    Vec3 color;
    float shadow;
} PixelColor;

// Simple scene with shadow rendering
void render_scene_with_shadows(const char* filename, int width, int height,
                             Triangle* triangles, int numTriangles,
                             Light* lights, int numLights) {
    // Create framebuffer
    PixelColor* framebuffer = (PixelColor*)calloc(width * height, sizeof(PixelColor));
    float* depthBuffer = (float*)malloc(width * height * sizeof(float));
    
    // Initialize depth buffer
    for (int i = 0; i < width * height; i++) {
        depthBuffer[i] = FLT_MAX;
    }
    
    // Setup camera
    Camera camera = {
        .position = {5, 5, 5},
        .forward = vec3_normalize((Vec3){-1, -1, -1}),
        .up = {0, 1, 0},
        .fov = 60.0f,
        .aspect = (float)width / height,
        .near = 0.1f,
        .far = 100.0f
    };
    
    Vec3 target = vec3_add(camera.position, camera.forward);
    Mat4 view = mat4_lookAt(camera.position, target, camera.up);
    Mat4 proj = mat4_perspective(camera.fov, camera.aspect, camera.near, camera.far);
    Mat4 viewProj = mat4_multiply(proj, view);
    
    // Render scene
    for (int i = 0; i < numTriangles; i++) {
        Triangle tri = triangles[i];
        
        // Transform vertices
        Vec3 v0 = mat4_transform_point(viewProj, tri.v[0].position);
        Vec3 v1 = mat4_transform_point(viewProj, tri.v[1].position);
        Vec3 v2 = mat4_transform_point(viewProj, tri.v[2].position);
        
        // Convert to screen coordinates
        Vec2 v0_screen = {(v0.x + 1.0f) * 0.5f * width, (1.0f - v0.y) * 0.5f * height};
        Vec2 v1_screen = {(v1.x + 1.0f) * 0.5f * width, (1.0f - v1.y) * 0.5f * height};
        Vec2 v2_screen = {(v2.x + 1.0f) * 0.5f * width, (1.0f - v2.y) * 0.5f * height};
        
        // Bounding box
        float minX = fmaxf(0, fminf(v0_screen.x, fminf(v1_screen.x, v2_screen.x)));
        float minY = fmaxf(0, fminf(v0_screen.y, fminf(v1_screen.y, v2_screen.y)));
        float maxX = fminf(width - 1, fmaxf(v0_screen.x, fmaxf(v1_screen.x, v2_screen.x)));
        float maxY = fminf(height - 1, fmaxf(v0_screen.y, fmaxf(v1_screen.y, v2_screen.y)));
        
        // Rasterize
        for (int y = (int)minY; y <= (int)maxY; y++) {
            for (int x = (int)minX; x <= (int)maxX; x++) {
                Vec2 p = {x + 0.5f, y + 0.5f};
                
                float u, v, w;
                barycentric(p, v0_screen, v1_screen, v2_screen, &u, &v, &w);
                
                if (u >= 0 && v >= 0 && w >= 0) {
                    float depth = u * v0.z + v * v1.z + w * v2.z;
                    int idx = y * width + x;
                    
                    if (depth < depthBuffer[idx]) {
                        depthBuffer[idx] = depth;
                        
                        // Interpolate world position
                        Vec3 worldPos = {
                            u * tri.v[0].position.x + v * tri.v[1].position.x + w * tri.v[2].position.x,
                            u * tri.v[0].position.y + v * tri.v[1].position.y + w * tri.v[2].position.y,
                            u * tri.v[0].position.z + v * tri.v[1].position.z + w * tri.v[2].position.z
                        };
                        
                        // Calculate shadows from each light
                        float totalShadow = 1.0f;
                        for (int l = 0; l < numLights; l++) {
                            if (lights[l].castsShadows && lights[l].shadowMap) {
                                float shadow = sample_shadow_map_pcf(
                                    (ShadowMap*)lights[l].shadowMap, worldPos, 5
                                );
                                totalShadow *= shadow;
                            }
                        }
                        
                        // Simple shading
                        Vec3 normal = vec3_normalize((Vec3){
                            u * tri.v[0].normal.x + v * tri.v[1].normal.x + w * tri.v[2].normal.x,
                            u * tri.v[0].normal.y + v * tri.v[1].normal.y + w * tri.v[2].normal.y,
                            u * tri.v[0].normal.z + v * tri.v[1].normal.z + w * tri.v[2].normal.z
                        });
                        
                        float lighting = fmaxf(0.2f, vec3_dot(normal, vec3_normalize((Vec3){1, 1, 1})));
                        
                        framebuffer[idx].color = (Vec3){
                            lighting * totalShadow,
                            lighting * totalShadow,
                            lighting * totalShadow
                        };
                        framebuffer[idx].shadow = totalShadow;
                    }
                }
            }
        }
    }
    
    // Save result
    FILE* file = fopen(filename, "wb");
    if (file) {
        fprintf(file, "P6\n%d %d\n255\n", width, height);
        for (int i = 0; i < width * height; i++) {
            fputc((unsigned char)(framebuffer[i].color.x * 255), file);
            fputc((unsigned char)(framebuffer[i].color.y * 255), file);
            fputc((unsigned char)(framebuffer[i].color.z * 255), file);
        }
        fclose(file);
    }
    
    free(framebuffer);
    free(depthBuffer);
}

// Create ground plane
void create_ground_plane(Triangle* triangles, float size, float y) {
    Vertex v0 = {{-size, y, -size}, {0, 1, 0}, {0, 0}};
    Vertex v1 = {{ size, y, -size}, {0, 1, 0}, {1, 0}};
    Vertex v2 = {{ size, y,  size}, {0, 1, 0}, {1, 1}};
    Vertex v3 = {{-size, y,  size}, {0, 1, 0}, {0, 1}};
    
    triangles[0] = (Triangle){v0, v1, v2};
    triangles[1] = (Triangle){v0, v2, v3};
}

// Create cube
void create_cube(Triangle* triangles, Vec3 center, float size) {
    float h = size * 0.5f;
    
    // Define vertices
    Vertex vertices[8] = {
        {{center.x - h, center.y - h, center.z - h}, {0, 0, 0}, {0, 0}},
        {{center.x + h, center.y - h, center.z - h}, {0, 0, 0}, {1, 0}},
        {{center.x + h, center.y + h, center.z - h}, {0, 0, 0}, {1, 1}},
        {{center.x - h, center.y + h, center.z - h}, {0, 0, 0}, {0, 1}},
        {{center.x - h, center.y - h, center.z + h}, {0, 0, 0}, {0, 0}},
        {{center.x + h, center.y - h, center.z + h}, {0, 0, 0}, {1, 0}},
        {{center.x + h, center.y + h, center.z + h}, {0, 0, 0}, {1, 1}},
        {{center.x - h, center.y + h, center.z + h}, {0, 0, 0}, {0, 1}}
    };
    
    int faces[12][3] = {
        {0, 1, 2}, {0, 2, 3}, // Front
        {5, 4, 7}, {5, 7, 6}, // Back
        {4, 0, 3}, {4, 3, 7}, // Left
        {1, 5, 6}, {1, 6, 2}, // Right
        {3, 2, 6}, {3, 6, 7}, // Top
        {4, 5, 1}, {4, 1, 0}  // Bottom
    };
    
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 3; j++) {
            triangles[i].v[j] = vertices[faces[i][j]];
        }
        
        // Calculate normal
        Vec3 e1 = vec3_sub(triangles[i].v[1].position, triangles[i].v[0].position);
        Vec3 e2 = vec3_sub(triangles[i].v[2].position, triangles[i].v[0].position);
        Vec3 normal = vec3_normalize(vec3_cross(e1, e2));
        
        for (int j = 0; j < 3; j++) {
            triangles[i].v[j].normal = normal;
        }
    }
}

int main() {
    printf("Shadow Mapping Demo\n");
    printf("===================\n\n");
    
    // Create scene geometry
    Triangle* triangles = (Triangle*)malloc(sizeof(Triangle) * 50);
    int numTriangles = 0;
    
    // Ground plane
    create_ground_plane(&triangles[numTriangles], 10.0f, 0.0f);
    numTriangles += 2;
    
    // Add some cubes
    create_cube(&triangles[numTriangles], (Vec3){0, 1, 0}, 2.0f);
    numTriangles += 12;
    
    create_cube(&triangles[numTriangles], (Vec3){3, 0.5f, -2}, 1.0f);
    numTriangles += 12;
    
    create_cube(&triangles[numTriangles], (Vec3){-2, 0.75f, 2}, 1.5f);
    numTriangles += 12;
    
    // Setup lights
    Light lights[3];
    
    // Directional light (sun)
    printf("Generating directional light shadow map...\n");
    lights[0] = (Light){
        .direction = vec3_normalize((Vec3){-1, -2, -1}),
        .color = {1.0f, 0.9f, 0.8f},
        .intensity = 1.0f,
        .castsShadows = 1
    };
    
    ShadowMap* dirShadowMap = create_shadow_map(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, SHADOW_MAP_2D);
    lights[0].shadowMap = dirShadowMap;
    generate_directional_shadow_map(dirShadowMap, &lights[0], triangles, numTriangles, 
                                  (Vec3){0, 0, 0}, 5.0f);
    save_shadow_map_ppm("shadow_directional.ppm", dirShadowMap);
    
    // Spot light
    printf("Generating spot light shadow map...\n");
    lights[1] = (Light){
        .position = {4, 5, 4},
        .direction = vec3_normalize((Vec3){-1, -1, -1}),
        .color = {1.0f, 1.0f, 0.8f},
        .intensity = 10.0f,
        .castsShadows = 1
    };
    
    ShadowMap* spotShadowMap = create_shadow_map(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, SHADOW_MAP_2D);
    lights[1].shadowMap = spotShadowMap;
    generate_spot_shadow_map(spotShadowMap, &lights[1], triangles, numTriangles, 45.0f);
    save_shadow_map_ppm("shadow_spot.ppm", spotShadowMap);
    
    // Point light
    printf("Generating point light shadow map...\n");
    lights[2] = (Light){
        .position = {-3, 3, 0},
        .color = {0.8f, 0.8f, 1.0f},
        .intensity = 5.0f,
        .castsShadows = 1
    };
    
    CubeShadowMap* pointShadowMap = create_cube_shadow_map(512);
    lights[2].shadowMap = pointShadowMap;
    generate_point_shadow_map(pointShadowMap, &lights[2], triangles, numTriangles);
    save_cube_shadow_map_ppm("shadow_cube", pointShadowMap);
    
    // Render scene with shadows
    printf("Rendering scene with shadows...\n");
    render_scene_with_shadows("scene_with_shadows.ppm", 800, 600, triangles, numTriangles, 
                            lights, 1); // Using only directional light for main scene
    
    // Cleanup
    free(triangles);
    destroy_shadow_map(dirShadowMap);
    destroy_shadow_map(spotShadowMap);
    destroy_cube_shadow_map(pointShadowMap);
    
    printf("\nGenerated files:\n");
    printf("  - shadow_directional.ppm (directional light depth map)\n");
    printf("  - shadow_spot.ppm (spot light depth map)\n");
    printf("  - shadow_cube_*.ppm (point light cube map faces)\n");
    printf("  - scene_with_shadows.ppm (final rendered scene)\n");
    
    return 0;
}