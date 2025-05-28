#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define PI 3.14159265359f
#define FRAMEBUFFER_WIDTH 800
#define FRAMEBUFFER_HEIGHT 600
#define TEXTURE_SIZE 256
#define MAX_MIP_LEVELS 8
#define MAX_TRIANGLES 100
#define POM_MAX_STEPS 32
#define POM_MIN_STEPS 4
#define POM_HEIGHT_SCALE 0.1f

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

// Framebuffer pixel
typedef struct {
    Vec3 color;
    float depth;
} Pixel;

// Framebuffer structure
typedef struct {
    int width, height;
    Pixel* data;
} Framebuffer;

// Texture structure
typedef enum {
    WRAP_REPEAT,
    WRAP_CLAMP,
    WRAP_MIRROR
} WrapMode;

typedef struct {
    int width, height;
    int mipLevels;
    Vec3* data[MAX_MIP_LEVELS]; // RGB for diffuse/normal, single-channel for height
    WrapMode wrapU, wrapV;
    int channels; // 3 for RGB, 1 for height
} Texture;

// Vertex structure
typedef struct {
    Vec3 position;
    Vec2 texcoord;
    Vec3 normal;
    Vec3 tangent;
} Vertex;

// Triangle structure
typedef struct {
    Vertex v[3];
} Triangle;

// Camera structure
typedef struct {
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    float fov;
    float aspect;
    float near;
    float far;
} Camera;

// Light structure
typedef struct {
    Vec3 position;
    Vec3 color;
    float intensity;
} Light;

// Vector math functions
static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 vec3_mul(Vec3 v, float s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

static inline float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline float vec3_length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static inline Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    return len > 0.0001f ? vec3_mul(v, 1.0f / len) : v;
}

static inline Vec3 vec3_clamp(Vec3 v, float min, float max) {
    return (Vec3){
        fmaxf(min, fminf(max, v.x)),
        fmaxf(min, fminf(max, v.y)),
        fmaxf(min, fminf(max, v.z))
    };
}

// Matrix operations
static Mat4 mat4_identity(void) {
    Mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static Mat4 mat4_multiply(Mat4 a, Mat4 b) {
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

static Mat4 mat4_perspective(float fov, float aspect, float near, float far) {
    Mat4 m = {0};
    float tanHalfFov = tanf(fov * 0.5f * PI / 180.0f);
    
    m.m[0] = 1.0f / (aspect * tanHalfFov);
    m.m[5] = 1.0f / tanHalfFov;
    m.m[10] = -(far + near) / (far - near);
    m.m[11] = -1.0f;
    m.m[14] = -(2.0f * far * near) / (far - near);
    
    return m;
}

static Mat4 mat4_lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = vec3_normalize(vec3_sub(center, eye));
    Vec3 s = vec3_normalize(vec3_cross(f, up));
    Vec3 u = vec3_cross(s, f);
    
    Mat4 m = mat4_identity();
    m.m[0] = s.x; m.m[4] = s.y; m.m[8] = s.z;
    m.m[1] = u.x; m.m[5] = u.y; m.m[9] = u.z;
    m.m[2] = -f.x; m.m[6] = -f.y; m.m[10] = -f.z;
    m.m[12] = -vec3_dot(s, eye);
    m.m[13] = -vec3_dot(u, eye);
    m.m[14] = vec3_dot(f, eye);
    
    return m;
}

static Vec3 mat4_transform_point(Mat4 m, Vec3 p) {
    float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];
    return (Vec3){
        (m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12]) / w,
        (m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13]) / w,
        (m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14]) / w
    };
}

// Framebuffer operations
static Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = (Framebuffer*)malloc(sizeof(Framebuffer));
    if (!fb) return NULL;
    fb->width = width;
    fb->height = height;
    fb->data = (Pixel*)calloc(width * height, sizeof(Pixel));
    if (!fb->data) {
        free(fb);
        return NULL;
    }
    for (int i = 0; i < width * height; i++) {
        fb->data[i].depth = FLT_MAX;
    }
    return fb;
}

static void destroy_framebuffer(Framebuffer* fb) {
    if (fb) {
        free(fb->data);
        free(fb);
    }
}

static void clear_framebuffer(Framebuffer* fb) {
    for (int i = 0; i < fb->width * fb->height; i++) {
        fb->data[i].color = (Vec3){0.1f, 0.1f, 0.2f};
        fb->data[i].depth = FLT_MAX;
    }
}

// Texture operations
static Texture* create_texture(int width, int height, int channels, WrapMode wrapU, WrapMode wrapV) {
    Texture* tex = (Texture*)malloc(sizeof(Texture));
    if (!tex) return NULL;
    tex->width = width;
    tex->height = height;
    tex->channels = channels;
    tex->wrapU = wrapU;
    tex->wrapV = wrapV;
    tex->mipLevels = 0;
    
    int w = width, h = height;
    while (w >= 1 && h >= 1 && tex->mipLevels < MAX_MIP_LEVELS) {
        tex->data[tex->mipLevels] = (Vec3*)malloc(w * h * sizeof(Vec3));
        if (!tex->data[tex->mipLevels]) {
            while (tex->mipLevels > 0) {
                free(tex->data[--tex->mipLevels]);
            }
            free(tex);
            return NULL;
        }
        tex->mipLevels++;
        w /= 2;
        h /= 2;
    }
    
    return tex;
}

static void destroy_texture(Texture* tex) {
    if (tex) {
        for (int i = 0; i < tex->mipLevels; i++) {
            free(tex->data[i]);
        }
        free(tex);
    }
}

// Generate procedural textures
static void generate_checkerboard_texture(Texture* tex) {
    const int checkSize = 32;
    for (int y = 0; y < tex->height; y++) {
        for (int x = 0; x < tex->width; x++) {
            int cx = (x / checkSize) % 2;
            int cy = (y / checkSize) % 2;
            Vec3 color = (cx == cy) ? (Vec3){0.8f, 0.8f, 0.8f} : (Vec3){0.2f, 0.2f, 0.2f};
            tex->data[0][y * tex->width + x] = color;
        }
    }
    
    // Generate mipmaps
    for (int level = 1; level < tex->mipLevels; level++) {
        int srcW = tex->width >> (level - 1);
        int srcH = tex->height >> (level - 1);
        int dstW = srcW >> 1;
        int dstH = srcH >> 1;
        
        for (int y = 0; y < dstH; y++) {
            for (int x = 0; x < dstW; x++) {
                Vec3 sum = {0, 0, 0};
                for (int sy = 0; sy < 2; sy++) {
                    for (int sx = 0; sx < 2; sx++) {
                        sum = vec3_add(sum, tex->data[level-1][(y*2+sy) * srcW + (x*2+sx)]);
                    }
                }
                tex->data[level][y * dstW + x] = vec3_mul(sum, 0.25f);
            }
        }
    }
}

static void generate_normal_map(Texture* tex) {
    for (int y = 0; y < tex->height; y++) {
        for (int x = 0; x < tex->width; x++) {
            float u = (float)x / tex->width;
            float v = (float)y / tex->height;
            Vec3 normal = vec3_normalize((Vec3){
                sinf(u * 2.0f * PI) * cosf(v * 2.0f * PI),
                cosf(u * 2.0f * PI) * cosf(v * 2.0f * PI),
                sinf(v * 2.0f * PI)
            });
            tex->data[0][y * tex->width + x] = vec3_mul(vec3_add(normal, (Vec3){1, 1, 1}), 0.5f);
        }
    }
    
    // Generate mipmaps
    for (int level = 1; level < tex->mipLevels; level++) {
        int srcW = tex->width >> (level - 1);
        int srcH = tex->height >> (level - 1);
        int dstW = srcW >> 1;
        int dstH = srcH >> 1;
        
        for (int y = 0; y < dstH; y++) {
            for (int x = 0; x < dstW; x++) {
                Vec3 sum = {0, 0, 0};
                for (int sy = 0; sy < 2; sy++) {
                    for (int sx = 0; sx < 2; sx++) {
                        sum = vec3_add(sum, tex->data[level-1][(y*2+sy) * srcW + (x*2+sx)]);
                    }
                }
                tex->data[level][y * dstW + x] = vec3_mul(sum, 0.25f);
            }
        }
    }
}

static void generate_height_map(Texture* tex) {
    for (int y = 0; y < tex->height; y++) {
        for (int x = 0; x < tex->width; x++) {
            float u = (float)x / tex->width;
            float v = (float)y / tex->height;
            float height = 0.5f * (sinf(u * 4.0f * PI) * cosf(v * 4.0f * PI) + 1.0f);
            tex->data[0][y * tex->width + x] = (Vec3){height, height, height};
        }
    }
    
    // Generate mipmaps
    for (int level = 1; level < tex->mipLevels; level++) {
        int srcW = tex->width >> (level - 1);
        int srcH = tex->height >> (level - 1);
        int dstW = srcW >> 1;
        int dstH = srcH >> 1;
        
        for (int y = 0; y < dstH; y++) {
            for (int x = 0; x < dstW; x++) {
                Vec3 sum = {0, 0, 0};
                for (int sy = 0; sy < 2; sy++) {
                    for (int sx = 0; sx < 2; sx++) {
                        sum = vec3_add(sum, tex->data[level-1][(y*2+sy) * srcW + (x*2+sx)]);
                    }
                }
                tex->data[level][y * dstW + x] = vec3_mul(sum, 0.25f);
            }
        }
    }
}

// Texture sampling
static float wrap_coord(float t, WrapMode mode) {
    switch (mode) {
        case WRAP_REPEAT: return t - floorf(t);
        case WRAP_CLAMP: return fmaxf(0.0f, fminf(1.0f, t));
        case WRAP_MIRROR: {
            float f = t - floorf(t);
            return (int)floorf(t) % 2 == 0 ? f : 1.0f - f;
        }
    }
    return t;
}

static Vec3 sample_texture(Texture* tex, Vec2 uv, float lod) {
    uv.x = wrap_coord(uv.x, tex->wrapU);
    uv.y = wrap_coord(uv.y, tex->wrapV);
    
    int level = (int)lod;
    level = fmaxf(0, fminf(tex->mipLevels - 1, level));
    
    int w = tex->width >> level;
    int h = tex->height >> level;
    
    // Bilinear filtering
    float u = uv.x * w - 0.5f;
    float v = uv.y * h - 0.5f;
    int u0 = (int)u;
    int v0 = (int)v;
    int u1 = u0 + 1;
    int v1 = v0 + 1;
    
    u0 = fmaxf(0, fminf(w - 1, u0));
    v0 = fmaxf(0, fminf(h - 1, v0));
    u1 = fmaxf(0, fminf(w - 1, u1));
    v1 = fmaxf(0, fminf(h - 1, v1));
    
    float fu = u - u0;
    float fv = v - v0;
    
    Vec3 c00 = tex->data[level][v0 * w + u0];
    Vec3 c10 = tex->data[level][v0 * w + u1];
    Vec3 c01 = tex->data[level][v1 * w + u0];
    Vec3 c11 = tex->data[level][v1 * w + u1];
    
    Vec3 c0 = vec3_add(vec3_mul(c00, 1.0f - fu), vec3_mul(c10, fu));
    Vec3 c1 = vec3_add(vec3_mul(c01, 1.0f - fu), vec3_mul(c11, fu));
    
    return vec3_add(vec3_mul(c0, 1.0f - fv), vec3_mul(c1, fv));
}

// Parallax occlusion mapping
static Vec2 parallax_occlusion_mapping(Texture* heightTex, Vec2 uv, Vec3 viewDir, Vec3 tangent, Vec3 bitangent, float heightScale) {
    // Transform view direction to tangent space
    Vec3 tangentViewDir = {
        vec3_dot(viewDir, tangent),
        vec3_dot(viewDir, bitangent),
        vec3_dot(viewDir, vec3_cross(tangent, bitangent))
    };
    tangentViewDir = vec3_normalize(tangentViewDir);
    
    // Calculate number of steps based on view angle
    float viewAngle = fmaxf(0.0f, tangentViewDir.z);
    int numSteps = (int)(POM_MIN_STEPS + (POM_MAX_STEPS - POM_MIN_STEPS) * (1.0f - viewAngle));
    
    float layerDepth = 1.0f / numSteps;
    float currentDepth = 0.0f;
    Vec2 deltaUV = vec3_mul((Vec3){tangentViewDir.x, tangentViewDir.y, 0}, heightScale / tangentViewDir.z / numSteps).xy;
    Vec2 currentUV = uv;
    float prevHeight = 1.0f;
    Vec2 prevUV = uv;
    
    // Ray marching
    for (int i = 0; i < numSteps; i++) {
        float height = sample_texture(heightTex, currentUV, 0.0f).x;
        if (height < currentDepth) {
            // Interpolate between current and previous
            float t = (prevHeight - currentDepth) / (prevHeight - height + currentDepth - currentDepth + layerDepth);
            return (Vec2){
                prevUV.x + (currentUV.x - prevUV.x) * t,
                prevUV.y + (currentUV.y - prevUV.y) * t
            };
        }
        prevHeight = height;
        prevUV = currentUV;
        currentUV = vec2_add(currentUV, deltaUV);
        currentDepth += layerDepth;
    }
    
    return uv; // No intersection found
}

// Barycentric coordinates
static void barycentric(Vec2 p, Vec2 a, Vec2 b, Vec2 c, float* u, float* v, float* w) {
    Vec2 v0 = {b.x - a.x, b.y - a.y};
    Vec2 v1 = {c.x - a.x, c.y - a.y};
    Vec2 v2 = {p.x - a.x, p.y - a.y};
    
    float d00 = v0.x * v0.x + v0.y * v0.y;
    float d01 = v0.x * v1.x + v0.y * v1.y;
    float d11 = v1.x * v1.x + v1.y * v1.y;
    float d20 = v2.x * v0.x + v2.y * v0.y;
    float d21 = v2.x * v1.x + v2.y * v1.y;
    
    float denom = d00 * d11 - d01 * d01;
    *v = denom != 0 ? (d11 * d20 - d01 * d21) / denom : 0;
    *w = denom != 0 ? (d00 * d21 - d01 * d20) / denom : 0;
    *u = 1.0f - *v - *w;
}

// Render triangle with parallax mapping
static void render_triangle(Framebuffer* fb, Triangle tri, Mat4 viewProj, Texture* diffuseTex, Texture* normalTex, Texture* heightTex, Light* light, Camera* camera) {
    Vec3 v0 = mat4_transform_point(viewProj, tri.v[0].position);
    Vec3 v1 = mat4_transform_point(viewProj, tri.v[1].position);
    Vec3 v2 = mat4_transform_point(viewProj, tri.v[2].position);
    
    Vec2 v0_screen = {(v0.x + 1.0f) * 0.5f * fb->width, (1.0f - v0.y) * 0.5f * fb->height};
    Vec2 v1_screen = {(v1.x + 1.0f) * 0.5f * fb->width, (1.0f - v1.y) * 0.5f * fb->height};
    Vec2 v2_screen = {(v2.x + 1.0f) * 0.5f * fb->width, (1.0f - v2.y) * 0.5f * fb->height};
    
    float minX = fmaxf(0, fminf(v0_screen.x, fminf(v1_screen.x, v2_screen.x)));
    float minY = fmaxf(0, fminf(v0_screen.y, fminf(v1_screen.y, v2_screen.y)));
    float maxX = fminf(fb->width - 1, fmaxf(v0_screen.x, fmaxf(v1_screen.x, v2_screen.x)));
    float maxY = fminf(fb->height - 1, fmaxf(v0_screen.y, fmaxf(v1_screen.y, v2_screen.y)));
    
    for (int y = (int)minY; y <= (int)maxY; y++) {
        for (int x = (int)minX; x <= (int)maxX; x++) {
            Vec2 p = {x + 0.5f, y + 0.5f};
            float u, v, w;
            barycentric(p, v0_screen, v1_screen, v2_screen, &u, &v, &w);
            
            if (u >= 0 && v >= 0 && w >= 0) {
                // Perspective-correct interpolation
                float z0 = 1.0f / v0.z;
                float z1 = 1.0f / v1.z;
                float z2 = 1.0f / v2.z;
                float u_persp = (u * z0 + v * z1 + w * z2);
                float u_corr = u * z0 / u_persp;
                float v_corr = v * z1 / u_persp;
                float w_corr = w * z2 / u_persp;
                float depth = 1.0f / (u_corr * z0 + v_corr * z1 + w_corr * z2);
                
                int idx = y * fb->width + x;
                if (depth < fb->data[idx].depth) {
                    fb->data[idx].depth = depth;
                    
                    // Interpolate attributes
                    Vec2 texcoord = {
                        u_corr * tri.v[0].texcoord.x + v_corr * tri.v[1].texcoord.x + w_corr * tri.v[2].texcoord.x,
                        u_corr * tri.v[0].texcoord.y + v_corr * tri.v[1].texcoord.y + w_corr * tri.v[2].texcoord.y
                    };
                    Vec3 tangent = vec3_normalize(vec3_add(vec3_add(
                        vec3_mul(tri.v[0].tangent, u_corr),
                        vec3_mul(tri.v[1].tangent, v_corr)),
                        vec3_mul(tri.v[2].tangent, w_corr)));
                    Vec3 normal = vec3_normalize(vec3_add(vec3_add(
                        vec3_mul(tri.v[0].normal, u_corr),
                        vec3_mul(tri.v[1].normal, v_corr)),
                        vec3_mul(tri.v[2].normal, w_corr)));
                    Vec3 bitangent = vec3_cross(normal, tangent);
                    
                    // World position
                    Vec3 worldPos = {
                        u_corr * tri.v[0].position.x + v_corr * tri.v[1].position.x + w_corr * tri.v[2].position.x,
                        u_corr * tri.v[0].position.y + v_corr * tri.v[1].position.y + w_corr * tri.v[2].position.y,
                        u_corr * tri.v[0].position.z + v_corr * tri.v[1].position.z + w_corr * tri.v[2].position.z
                    };
                    
                    // Parallax mapping
                    Vec3 viewDir = vec3_normalize(vec3_sub(camera->position, worldPos));
                    Vec2 parallaxUV = parallax_occlusion_mapping(heightTex, texcoord, viewDir, tangent, bitangent, POM_HEIGHT_SCALE);
                    
                    // Sample textures with parallax-adjusted UVs
                    Vec3 diffuseColor = sample_texture(diffuseTex, parallaxUV, 0.0f);
                    Vec3 normalMap = sample_texture(normalTex, parallaxUV, 0.0f);
                    normalMap = vec3_normalize(vec3_sub(vec3_mul(normalMap, 2.0f), (Vec3){1, 1, 1}));
                    
                    // TBN matrix for normal mapping
                    Vec3 tbn[3] = {tangent, bitangent, normal};
                    Vec3 worldNormal = {
                        tbn[0].x * normalMap.x + tbn[1].x * normalMap.y + tbn[2].x * normalMap.z,
                        tbn[0].y * normalMap.x + tbn[1].y * normalMap.y + tbn[2].y * normalMap.z,
                        tbn[0].z * normalMap.x + tbn[1].z * normalMap.y + tbn[2].z * normalMap.z
                    };
                    worldNormal = vec3_normalize(worldNormal);
                    
                    // Lighting
                    Vec3 lightDir = vec3_normalize(vec3_sub(light->position, worldPos));
                    float diffuse = fmaxf(0.0f, vec3_dot(worldNormal, lightDir));
                    Vec3 ambient = (Vec3){0.2f, 0.2f, 0.2f};
                    Vec3 color = vec3_add(vec3_mul(diffuseColor, diffuse * light->intensity), ambient);
                    
                    fb->data[idx].color = vec3_clamp(color, 0.0f, 1.0f);
                }
            }
        }
    }
}

// Save texture as PPM
static void save_texture_ppm(const char* filename, Texture* tex, int level) {
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    int w = tex->width >> level;
    int h = tex->height >> level;
    fprintf(file, "P6\n%d %d\n255\n", w, h);
    
    for (int i = 0; i < w * h; i++) {
        Vec3 color = vec3_clamp(tex->data[level][i], 0.0f, 1.0f);
        if (tex->channels == 1) color = (Vec3){color.x, color.x, color.x}; // Height map to grayscale
        fputc((unsigned char)(color.x * 255), file);
        fputc((unsigned char)(color.y * 255), file);
        fputc((unsigned char)(color.z * 255), file);
    }
    
    fclose(file);
}

// Create ground plane
static void create_ground_plane(Triangle* triangles, int* numTriangles, float size, float y) {
    Vertex v0 = {{-size, y, -size}, {0, 0}, {0, 1, 0}, {1, 0, 0}};
    Vertex v1 = {{ size, y, -size}, {2, 0}, {0, 1, 0}, {1, 0, 0}};
    Vertex v2 = {{ size, y,  size}, {2, 2}, {0, 1, 0}, {1, 0, 0}};
    Vertex v3 = {{-size, y,  size}, {0, 2}, {0, 1, 0}, {1, 0, 0}};
    
    triangles[0] = (Triangle){v0, v1, v2};
    triangles[1] = (Triangle){v0, v2, v3};
    *numTriangles = 2;
}

// Create cube
static void create_cube(Triangle* triangles, int* numTriangles, Vec3 center, float size) {
    float h = size * 0.5f;
    
    Vertex vertices[8] = {
        {{center.x - h, center.y - h, center.z - h}, {0, 0}, {0, 0, -1}, {1, 0, 0}},
        {{center.x + h, center.y - h, center.z - h}, {1, 0}, {0, 0, -1}, {1, 0, 0}},
        {{center.x + h, center.y + h, center.z - h}, {1, 1}, {0, 0, -1}, {1, 0, 0}},
        {{center.x - h, center.y + h, center.z - h}, {0, 1}, {0, 0, -1}, {1, 0, 0}},
        {{center.x - h, center.y - h, center.z + h}, {0, 0}, {0, 0, 1}, {1, 0, 0}},
        {{center.x + h, center.y - h, center.z + h}, {1, 0}, {0, 0, 1}, {1, 0, 0}},
        {{center.x + h, center.y + h, center.z + h}, {1, 1}, {0, 0, 1}, {1, 0, 0}},
        {{center.x - h, center.y + h, center.z + h}, {0, 1}, {0, 0, 1}, {1, 0, 0}}
    };
    
    int faces[12][3] = {
        {0, 1, 2}, {0, 2, 3}, // Front
        {5, 4, 7}, {5, 7, 6}, // Back
        {4, 0, 3}, {4, 3, 7}, // Left
        {1, 5, 6}, {1, 6, 2}, // Right
        {3, 2, 6}, {3, 6, 7}, // Top
        {4, 5, 1}, {4, 1, 0}  // Bottom
    };
    
    Vec3 faceNormals[6] = {
        {0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}
    };
    
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 3; j++) {
            triangles[i].v[j] = vertices[faces[i][j]];
            triangles[i].v[j].normal = faceNormals[i / 2];
        }
    }
    
    *numTriangles = 12;
}

int main(void) {
    printf("Parallax Mapping Demo\n");
    printf("====================\n\n");
    
    // Create framebuffer
    printf("Initializing framebuffer...\n");
    Framebuffer* fb = create_framebuffer(FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT);
    if (!fb) {
        fprintf(stderr, "Error: Failed to create framebuffer\n");
        return 1;
    }
    clear_framebuffer(fb);
    
    // Create textures
    printf("Generating textures...\n");
    Texture* diffuseTex = create_texture(TEXTURE_SIZE, TEXTURE_SIZE, 3, WRAP_REPEAT, WRAP_REPEAT);
    Texture* normalTex = create_texture(TEXTURE_SIZE, TEXTURE_SIZE, 3, WRAP_CLAMP, WRAP_CLAMP);
    Texture* heightTex = create_texture(TEXTURE_SIZE, TEXTURE_SIZE, 1, WRAP_CLAMP, WRAP_CLAMP);
    if (!diffuseTex || !normalTex || !heightTex) {
        fprintf(stderr, "Error: Failed to create textures\n");
        destroy_framebuffer(fb);
        destroy_texture(diffuseTex);
        destroy_texture(normalTex);
        destroy_texture(heightTex);
        return 1;
    }
    
    generate_checkerboard_texture(diffuseTex);
    generate_normal_map(normalTex);
    generate_height_map(heightTex);
    
    save_texture_ppm("texture_diffuse.ppm", diffuseTex, 0);
    save_texture_ppm("texture_normal.ppm", normalTex, 0);
    save_texture_ppm("texture_height.ppm", heightTex, 0);
    
    // Setup camera
    Camera camera = {
        .position = {5, 5, 5},
        .forward = vec3_normalize((Vec3){-1, -1, -1}),
        .up = {0, 1, 0},
        .fov = 60.0f,
        .aspect = (float)FRAMEBUFFER_WIDTH / FRAMEBUFFER_HEIGHT,
        .near = 0.1f,
        .far = 100.0f
    };
    
    Vec3 target = vec3_add(camera.position, camera.forward);
    Mat4 view = mat4_lookAt(camera.position, target, camera.up);
    Mat4 proj = mat4_perspective(camera.fov, camera.aspect, camera.near, camera.far);
    Mat4 viewProj = mat4_multiply(proj, view);
    
    // Setup light
    Light light = {
        .position = {5, 5, 5},
        .color = {1, 1, 1},
        .intensity = 1.0f
    };
    
    // Create scene
    printf("Creating scene...\n");
    Triangle* triangles = (Triangle*)calloc(MAX_TRIANGLES, sizeof(Triangle));
    if (!triangles) {
        fprintf(stderr, "Error: Failed to allocate triangles\n");
        destroy_framebuffer(fb);
        destroy_texture(diffuseTex);
        destroy_texture(normalTex);
        destroy_texture(heightTex);
        return 1;
    }
    
    int numTriangles = 0;
    create_ground_plane(triangles, &numTriangles, 10.0f, 0.0f);
    create_cube(&triangles[numTriangles], &numTriangles, (Vec3){0, 1, 0}, 2.0f);
    
    // Render scene
    printf("Rendering scene with parallax mapping...\n");
    for (int i = 0; i < numTriangles; i++) {
        render_triangle(fb, triangles[i], viewProj, diffuseTex, normalTex, heightTex, &light, &camera);
    }
    
    // Save result
    printf("Saving rendered scene...\n");
    save_texture_ppm("scene_parallax.ppm", (Texture*)fb, 0); // Reuse texture save for framebuffer
    
    // Cleanup
    free(triangles);
    destroy_texture(diffuseTex);
    destroy_texture(normalTex);
    destroy_texture(heightTex);
    destroy_framebuffer(fb);
    
    printf("\nGenerated files:\n");
    printf("  - texture_diffuse.ppm (checkerboard diffuse texture)\n");
    printf("  - texture_normal.ppm (procedural normal map)\n");
    printf("  - texture_height.ppm (procedural height map)\n");
    printf("  - scene_parallax.ppm (rendered scene with parallax mapping)\n");
    
    return 0;
}