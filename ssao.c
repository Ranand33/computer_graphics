#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265359f
#define MAX_KERNEL_SIZE 64
#define NOISE_SIZE 4

// Vector structures
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float x, y;
} Vec2;

typedef struct {
    float m[16];
} Mat4;

// G-Buffer data
typedef struct {
    int width, height;
    Vec3* positions;      // View-space positions
    Vec3* normals;        // View-space normals
    float* depth;         // Linear depth values
    Vec3* worldPositions; // World-space positions (for visualization)
} GBuffer;

// SSAO parameters
typedef struct {
    int kernelSize;           // Number of samples (16-64)
    float radius;             // Sampling radius in world space
    float bias;               // Depth bias to prevent self-occlusion
    float intensity;          // AO intensity multiplier
    float power;              // Power function for contrast
    int noiseSize;            // Noise texture size (typically 4x4)
    int blurSize;             // Blur kernel size
    float rangeCheckScale;    // Scale for range check
} SSAOParams;

// SSAO data
typedef struct {
    Vec3* kernel;             // Sampling kernel
    Vec3* noise;              // Random rotation vectors
    float* aoBuffer;          // Raw AO values
    float* blurredAO;         // Blurred AO values
    SSAOParams params;
} SSAOData;

// Camera matrices
typedef struct {
    Mat4 view;
    Mat4 projection;
    Mat4 viewProjection;
    Mat4 invProjection;
    float near;
    float far;
} CameraData;

// Framebuffer
typedef struct {
    int width, height;
    Vec3* pixels;
} Framebuffer;

// Vector operations
float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 vec3_mul(Vec3 v, float s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
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

Vec3 vec3_reflect(Vec3 v, Vec3 n) {
    return vec3_sub(v, vec3_mul(n, 2.0f * vec3_dot(v, n)));
}

// Random float between 0 and 1
float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

// Random float between min and max
float random_range(float min, float max) {
    return min + random_float() * (max - min);
}

// Matrix operations
Mat4 mat4_identity() {
    Mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
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

Mat4 mat4_inverse_projection(Mat4 proj) {
    Mat4 inv = {0};
    
    inv.m[0] = 1.0f / proj.m[0];
    inv.m[5] = 1.0f / proj.m[5];
    inv.m[11] = 1.0f / proj.m[14];
    inv.m[14] = 1.0f / proj.m[11];
    inv.m[15] = -proj.m[10] / (proj.m[14] * proj.m[11]);
    
    return inv;
}

Vec3 mat4_transform_point(Mat4 m, Vec3 p) {
    float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];
    return (Vec3){
        (m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12]) / w,
        (m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13]) / w,
        (m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14]) / w
    };
}

// Create buffers
GBuffer* create_gbuffer(int width, int height) {
    GBuffer* gb = (GBuffer*)malloc(sizeof(GBuffer));
    gb->width = width;
    gb->height = height;
    
    int size = width * height;
    gb->positions = (Vec3*)calloc(size, sizeof(Vec3));
    gb->normals = (Vec3*)calloc(size, sizeof(Vec3));
    gb->depth = (float*)calloc(size, sizeof(float));
    gb->worldPositions = (Vec3*)calloc(size, sizeof(Vec3));
    
    return gb;
}

void destroy_gbuffer(GBuffer* gb) {
    free(gb->positions);
    free(gb->normals);
    free(gb->depth);
    free(gb->worldPositions);
    free(gb);
}

// Initialize SSAO data
SSAOData* create_ssao_data(int width, int height, SSAOParams params) {
    SSAOData* ssao = (SSAOData*)malloc(sizeof(SSAOData));
    ssao->params = params;
    
    // Allocate buffers
    int size = width * height;
    ssao->aoBuffer = (float*)malloc(size * sizeof(float));
    ssao->blurredAO = (float*)malloc(size * sizeof(float));
    
    // Generate sampling kernel
    ssao->kernel = (Vec3*)malloc(params.kernelSize * sizeof(Vec3));
    for (int i = 0; i < params.kernelSize; i++) {
        // Generate points in hemisphere
        Vec3 sample = {
            random_range(-1.0f, 1.0f),
            random_range(-1.0f, 1.0f),
            random_range(0.0f, 1.0f)
        };
        sample = vec3_normalize(sample);
        
        // Scale by random length
        float scale = (float)i / params.kernelSize;
        scale = 0.1f + scale * scale * 0.9f; // Bias samples towards center
        sample = vec3_mul(sample, random_range(0.0f, 1.0f) * scale);
        
        ssao->kernel[i] = sample;
    }
    
    // Generate noise texture
    int noiseTexSize = params.noiseSize * params.noiseSize;
    ssao->noise = (Vec3*)malloc(noiseTexSize * sizeof(Vec3));
    for (int i = 0; i < noiseTexSize; i++) {
        Vec3 noise = {
            random_range(-1.0f, 1.0f),
            random_range(-1.0f, 1.0f),
            0.0f
        };
        ssao->noise[i] = vec3_normalize(noise);
    }
    
    return ssao;
}

void destroy_ssao_data(SSAOData* ssao) {
    free(ssao->kernel);
    free(ssao->noise);
    free(ssao->aoBuffer);
    free(ssao->blurredAO);
    free(ssao);
}

// Create tangent space matrix from normal
void create_tbn_matrix(Vec3 normal, Vec3* tangent, Vec3* bitangent) {
    // Choose a vector that's not parallel to normal
    Vec3 up = (fabsf(normal.y) < 0.999f) ? (Vec3){0, 1, 0} : (Vec3){1, 0, 0};
    
    // Calculate tangent and bitangent
    *tangent = vec3_normalize(vec3_sub(up, vec3_mul(normal, vec3_dot(up, normal))));
    *bitangent = vec3_normalize(vec3_sub(
        vec3_sub((Vec3){0, 0, 1}, vec3_mul(normal, normal.z)),
        vec3_mul(*tangent, vec3_dot(*tangent, (Vec3){0, 0, 1}))
    ));
}

// Transform vector by TBN matrix
Vec3 tbn_transform(Vec3 v, Vec3 tangent, Vec3 bitangent, Vec3 normal) {
    return (Vec3){
        v.x * tangent.x + v.y * bitangent.x + v.z * normal.x,
        v.x * tangent.y + v.y * bitangent.y + v.z * normal.y,
        v.x * tangent.z + v.y * bitangent.z + v.z * normal.z
    };
}

// Reconstruct view-space position from depth
Vec3 reconstruct_position(float depth, Vec2 uv, Mat4 invProj) {
    // Convert UV to NDC
    Vec3 ndc = {
        uv.x * 2.0f - 1.0f,
        uv.y * 2.0f - 1.0f,
        depth * 2.0f - 1.0f
    };
    
    // Transform to view space
    Vec3 viewPos = mat4_transform_point(invProj, ndc);
    return viewPos;
}

// Linear depth from perspective depth
float linearize_depth(float depth, float near, float far) {
    float z = depth * 2.0f - 1.0f;
    return (2.0f * near * far) / (far + near - z * (far - near));
}

// Smoothstep function
float smoothstep(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

// Calculate SSAO for a single pixel
float calculate_pixel_ao(int x, int y, GBuffer* gb, SSAOData* ssao, CameraData* camera) {
    int idx = y * gb->width + x;
    
    // Skip background pixels
    if (gb->depth[idx] <= 0.0f || gb->depth[idx] >= 1.0f) {
        return 1.0f;
    }
    
    Vec3 fragPos = gb->positions[idx];
    Vec3 normal = gb->normals[idx];
    
    // Get noise value
    int noiseX = (x % ssao->params.noiseSize);
    int noiseY = (y % ssao->params.noiseSize);
    Vec3 randomVec = ssao->noise[noiseY * ssao->params.noiseSize + noiseX];
    
    // Create TBN matrix
    Vec3 tangent, bitangent;
    create_tbn_matrix(normal, &tangent, &bitangent);
    
    float occlusion = 0.0f;
    
    // Sample kernel
    for (int i = 0; i < ssao->params.kernelSize; i++) {
        // Rotate sample by random vector
        Vec3 sampleVec = ssao->kernel[i];
        
        // Reflect sample across random vector
        if (vec3_dot(sampleVec, randomVec) < 0) {
            sampleVec = vec3_mul(sampleVec, -1.0f);
        }
        
        // Transform sample to view space
        Vec3 sample = tbn_transform(sampleVec, tangent, bitangent, normal);
        sample = vec3_add(fragPos, vec3_mul(sample, ssao->params.radius));
        
        // Project sample to screen space
        Vec3 offset = mat4_transform_point(camera->projection, sample);
        offset.x = offset.x * 0.5f + 0.5f;
        offset.y = offset.y * 0.5f + 0.5f;
        
        // Sample depth at offset
        int sampleX = (int)(offset.x * gb->width);
        int sampleY = (int)(offset.y * gb->height);
        
        // Clamp to screen bounds
        sampleX = fmaxf(0, fminf(gb->width - 1, sampleX));
        sampleY = fmaxf(0, fminf(gb->height - 1, sampleY));
        
        int sampleIdx = sampleY * gb->width + sampleX;
        float sampleDepth = gb->positions[sampleIdx].z;
        
        // Range check
        float rangeCheck = smoothstep(0.0f, 1.0f, 
            ssao->params.radius / fabsf(fragPos.z - sampleDepth));
        
        // Check if sample is occluded
        occlusion += (sampleDepth >= sample.z + ssao->params.bias ? 1.0f : 0.0f) * rangeCheck;
    }
    
    // Average and invert
    occlusion = 1.0f - (occlusion / ssao->params.kernelSize);
    
    // Apply intensity and power
    occlusion = powf(occlusion, ssao->params.power) * ssao->params.intensity;
    
    return fmaxf(0.0f, fminf(1.0f, occlusion));
}

// Calculate SSAO for entire buffer
void calculate_ssao(GBuffer* gb, SSAOData* ssao, CameraData* camera) {
    // Calculate raw AO
    for (int y = 0; y < gb->height; y++) {
        for (int x = 0; x < gb->width; x++) {
            int idx = y * gb->width + x;
            ssao->aoBuffer[idx] = calculate_pixel_ao(x, y, gb, ssao, camera);
        }
    }
}

// Bilateral blur for SSAO
void bilateral_blur_ssao(GBuffer* gb, SSAOData* ssao) {
    int blurSize = ssao->params.blurSize;
    float depthThreshold = 0.1f;
    float normalThreshold = 0.9f;
    
    // Copy to temp buffer
    memcpy(ssao->blurredAO, ssao->aoBuffer, gb->width * gb->height * sizeof(float));
    
    // Two-pass separable blur
    for (int pass = 0; pass < 2; pass++) {
        float* input = (pass == 0) ? ssao->aoBuffer : ssao->blurredAO;
        float* output = (pass == 0) ? ssao->blurredAO : ssao->aoBuffer;
        
        for (int y = 0; y < gb->height; y++) {
            for (int x = 0; x < gb->width; x++) {
                int centerIdx = y * gb->width + x;
                
                if (gb->depth[centerIdx] <= 0.0f || gb->depth[centerIdx] >= 1.0f) {
                    output[centerIdx] = 1.0f;
                    continue;
                }
                
                float centerDepth = gb->depth[centerIdx];
                Vec3 centerNormal = gb->normals[centerIdx];
                
                float totalWeight = 0.0f;
                float totalAO = 0.0f;
                
                // Sample neighbors
                for (int offset = -blurSize; offset <= blurSize; offset++) {
                    int sx = x;
                    int sy = y;
                    
                    if (pass == 0) sx += offset;  // Horizontal pass
                    else sy += offset;             // Vertical pass
                    
                    // Check bounds
                    if (sx < 0 || sx >= gb->width || sy < 0 || sy >= gb->height) {
                        continue;
                    }
                    
                    int sampleIdx = sy * gb->width + sx;
                    
                    // Depth weight
                    float sampleDepth = gb->depth[sampleIdx];
                    float depthDiff = fabsf(centerDepth - sampleDepth);
                    float depthWeight = expf(-depthDiff / depthThreshold);
                    
                    // Normal weight
                    Vec3 sampleNormal = gb->normals[sampleIdx];
                    float normalDot = vec3_dot(centerNormal, sampleNormal);
                    float normalWeight = powf(fmaxf(0.0f, normalDot), 32.0f);
                    
                    // Spatial weight
                    float spatialWeight = expf(-(offset * offset) / (2.0f * blurSize * blurSize));
                    
                    // Combined weight
                    float weight = depthWeight * normalWeight * spatialWeight;
                    
                    totalWeight += weight;
                    totalAO += input[sampleIdx] * weight;
                }
                
                output[centerIdx] = totalAO / fmaxf(0.0001f, totalWeight);
            }
        }
    }
    
    // Final result is in aoBuffer
    memcpy(ssao->blurredAO, ssao->aoBuffer, gb->width * gb->height * sizeof(float));
}

// Generate test scene in G-buffer
void generate_test_scene(GBuffer* gb, CameraData* camera) {
    // Simple scene with spheres and a plane
    srand(42); // Fixed seed for reproducibility
    
    // Clear buffers
    for (int i = 0; i < gb->width * gb->height; i++) {
        gb->positions[i] = (Vec3){0, 0, -1000};
        gb->normals[i] = (Vec3){0, 0, 1};
        gb->depth[i] = 1.0f;
    }
    
    // Ground plane at y = -2
    for (int y = gb->height / 2; y < gb->height; y++) {
        for (int x = 0; x < gb->width; x++) {
            float u = (float)x / gb->width;
            float v = (float)y / gb->height;
            
            // Ray from camera through pixel
            Vec3 rayDir = {
                (u * 2.0f - 1.0f) * tanf(45.0f * PI / 180.0f) * ((float)gb->width / gb->height),
                (v * 2.0f - 1.0f) * tanf(45.0f * PI / 180.0f),
                -1.0f
            };
            rayDir = vec3_normalize(rayDir);
            
            // Intersect with plane y = -2
            float t = (-2.0f - 0.0f) / rayDir.y;
            if (t > 0 && t < 100) {
                Vec3 hitPoint = vec3_mul(rayDir, t);
                
                int idx = y * gb->width + x;
                gb->positions[idx] = hitPoint;
                gb->normals[idx] = (Vec3){0, 1, 0};
                gb->depth[idx] = hitPoint.z / -100.0f; // Normalize depth
                gb->worldPositions[idx] = hitPoint;
            }
        }
    }
    
    // Add some spheres
    struct Sphere {
        Vec3 center;
        float radius;
    } spheres[] = {
        {{0, 0, -5}, 1.0f},
        {{2.5f, -1, -6}, 1.0f},
        {{-2, 0.5f, -4}, 0.8f},
        {{1, -1.5f, -3}, 0.5f},
        {{-1.5f, -1.2f, -5}, 0.7f},
        {{3, 1, -8}, 1.5f},
        {{-3, -0.5f, -7}, 1.2f}
    };
    int numSpheres = sizeof(spheres) / sizeof(spheres[0]);
    
    // Rasterize spheres
    for (int s = 0; s < numSpheres; s++) {
        Vec3 center = spheres[s].center;
        float radius = spheres[s].radius;
        
        // Project sphere bounds to screen
        Vec3 minBound = vec3_sub(center, (Vec3){radius, radius, radius});
        Vec3 maxBound = vec3_add(center, (Vec3){radius, radius, radius});
        
        // Simple screen bounds (approximate)
        int minX = fmaxf(0, (int)((minBound.x / -minBound.z + 1) * 0.5f * gb->width - radius * 50));
        int maxX = fminf(gb->width - 1, (int)((maxBound.x / -maxBound.z + 1) * 0.5f * gb->width + radius * 50));
        int minY = fmaxf(0, (int)((minBound.y / -minBound.z + 1) * 0.5f * gb->height - radius * 50));
        int maxY = fminf(gb->height - 1, (int)((maxBound.y / -maxBound.z + 1) * 0.5f * gb->height + radius * 50));
        
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                float u = (float)x / gb->width;
                float v = (float)y / gb->height;
                
                // Ray from camera through pixel
                Vec3 rayOrigin = {0, 0, 0};
                Vec3 rayDir = {
                    (u * 2.0f - 1.0f) * tanf(45.0f * PI / 180.0f) * ((float)gb->width / gb->height),
                    (v * 2.0f - 1.0f) * tanf(45.0f * PI / 180.0f),
                    -1.0f
                };
                rayDir = vec3_normalize(rayDir);
                
                // Ray-sphere intersection
                Vec3 oc = vec3_sub(rayOrigin, center);
                float a = vec3_dot(rayDir, rayDir);
                float b = 2.0f * vec3_dot(oc, rayDir);
                float c = vec3_dot(oc, oc) - radius * radius;
                float discriminant = b * b - 4 * a * c;
                
                if (discriminant > 0) {
                    float t = (-b - sqrtf(discriminant)) / (2.0f * a);
                    if (t > 0.1f && t < 100) {
                        Vec3 hitPoint = vec3_add(rayOrigin, vec3_mul(rayDir, t));
                        float depth = hitPoint.z / -100.0f;
                        
                        int idx = y * gb->width + x;
                        if (depth < gb->depth[idx]) {
                            gb->positions[idx] = hitPoint;
                            gb->normals[idx] = vec3_normalize(vec3_sub(hitPoint, center));
                            gb->depth[idx] = depth;
                            gb->worldPositions[idx] = hitPoint;
                        }
                    }
                }
            }
        }
    }
    
    // Add some boxes/cubes
    for (int i = 0; i < 5; i++) {
        Vec3 boxCenter = {
            random_range(-4, 4),
            random_range(-1.5f, 0),
            random_range(-3, -8)
        };
        float boxSize = random_range(0.3f, 0.8f);
        
        // Simple box rasterization (just front face for simplicity)
        Vec3 minBox = vec3_sub(boxCenter, vec3_mul((Vec3){1, 1, 1}, boxSize));
        Vec3 maxBox = vec3_add(boxCenter, vec3_mul((Vec3){1, 1, 1}, boxSize));
        
        int minX = fmaxf(0, (int)((minBox.x / -minBox.z + 1) * 0.5f * gb->width));
        int maxX = fminf(gb->width - 1, (int)((maxBox.x / -maxBox.z + 1) * 0.5f * gb->width));
        int minY = fmaxf(0, (int)((minBox.y / -minBox.z + 1) * 0.5f * gb->height));
        int maxY = fminf(gb->height - 1, (int)((maxBox.y / -maxBox.z + 1) * 0.5f * gb->height));
        
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                int idx = y * gb->width + x;
                float z = boxCenter.z + boxSize;
                float depth = z / -100.0f;
                
                if (depth > 0 && depth < gb->depth[idx]) {
                    gb->positions[idx] = (Vec3){
                        (x / (float)gb->width * 2.0f - 1.0f) * -z,
                        (y / (float)gb->height * 2.0f - 1.0f) * -z,
                        z
                    };
                    gb->normals[idx] = (Vec3){0, 0, 1};
                    gb->depth[idx] = depth;
                }
            }
        }
    }
}

// Save PPM image
void save_ppm(const char* filename, float* data, int width, int height, int invert) {
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        float val = data[i];
        if (invert) val = 1.0f - val;
        
        unsigned char c = (unsigned char)(fmaxf(0.0f, fminf(1.0f, val)) * 255);
        fputc(c, file);
        fputc(c, file);
        fputc(c, file);
    }
    
    fclose(file);
}

// Save color PPM
void save_color_ppm(const char* filename, Vec3* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        fputc((unsigned char)(fmaxf(0.0f, fminf(1.0f, data[i].x)) * 255), file);
        fputc((unsigned char)(fmaxf(0.0f, fminf(1.0f, data[i].y)) * 255), file);
        fputc((unsigned char)(fmaxf(0.0f, fminf(1.0f, data[i].z)) * 255), file);
    }
    
    fclose(file);
}

// Apply SSAO to scene
void apply_ssao_to_scene(Framebuffer* fb, GBuffer* gb, SSAOData* ssao) {
    for (int i = 0; i < fb->width * fb->height; i++) {
        if (gb->depth[i] > 0.0f && gb->depth[i] < 1.0f) {
            // Simple shading
            Vec3 normal = gb->normals[i];
            Vec3 lightDir = vec3_normalize((Vec3){1, 1, 1});
            float diffuse = fmaxf(0.0f, vec3_dot(normal, lightDir));
            
            // Base color (use normal as color for visualization)
            Vec3 baseColor = {
                normal.x * 0.5f + 0.5f,
                normal.y * 0.5f + 0.5f,
                normal.z * 0.5f + 0.5f
            };
            
            // Apply AO
            float ao = ssao->blurredAO[i];
            
            fb->pixels[i] = vec3_mul(baseColor, (0.3f + 0.7f * diffuse) * ao);
        } else {
            // Background
            fb->pixels[i] = (Vec3){0.1f, 0.1f, 0.15f};
        }
    }
}

// Create comparison images
void create_comparison_images(GBuffer* gb, SSAOData* ssao, int width, int height) {
    Framebuffer* fbNoAO = (Framebuffer*)malloc(sizeof(Framebuffer));
    fbNoAO->width = width;
    fbNoAO->height = height;
    fbNoAO->pixels = (Vec3*)malloc(width * height * sizeof(Vec3));
    
    Framebuffer* fbWithAO = (Framebuffer*)malloc(sizeof(Framebuffer));
    fbWithAO->width = width;
    fbWithAO->height = height;
    fbWithAO->pixels = (Vec3*)malloc(width * height * sizeof(Vec3));
    
    // Render without AO
    for (int i = 0; i < width * height; i++) {
        if (gb->depth[i] > 0.0f && gb->depth[i] < 1.0f) {
            Vec3 normal = gb->normals[i];
            Vec3 lightDir = vec3_normalize((Vec3){1, 1, 1});
            float diffuse = fmaxf(0.0f, vec3_dot(normal, lightDir));
            
            Vec3 baseColor = {
                normal.x * 0.5f + 0.5f,
                normal.y * 0.5f + 0.5f,
                normal.z * 0.5f + 0.5f
            };
            
            fbNoAO->pixels[i] = vec3_mul(baseColor, 0.3f + 0.7f * diffuse);
        } else {
            fbNoAO->pixels[i] = (Vec3){0.1f, 0.1f, 0.15f};
        }
    }
    
    // Render with AO
    apply_ssao_to_scene(fbWithAO, gb, ssao);
    
    save_color_ppm("scene_without_ao.ppm", fbNoAO->pixels, width, height);
    save_color_ppm("scene_with_ao.ppm", fbWithAO->pixels, width, height);
    
    free(fbNoAO->pixels);
    free(fbNoAO);
    free(fbWithAO->pixels);
    free(fbWithAO);
}

int main() {
    printf("SSAO (Screen-Space Ambient Occlusion) Demo\n");
    printf("==========================================\n\n");
    
    int width = 800;
    int height = 600;
    
    // Initialize random seed
    srand(time(NULL));
    
    // Create buffers
    printf("Creating buffers...\n");
    GBuffer* gbuffer = create_gbuffer(width, height);
    
    // Setup camera
    CameraData camera;
    camera.near = 0.1f;
    camera.far = 100.0f;
    camera.projection = mat4_perspective(45.0f, (float)width / height, camera.near, camera.far);
    camera.invProjection = mat4_inverse_projection(camera.projection);
    
    // Generate test scene
    printf("Generating test scene...\n");
    generate_test_scene(gbuffer, &camera);
    
    // SSAO parameters - experiment with these!
    SSAOParams params = {
        .kernelSize = 32,          // More samples = better quality, slower
        .radius = 0.5f,            // Larger = wider occlusion
        .bias = 0.025f,            // Prevents self-occlusion
        .intensity = 1.5f,         // Overall strength
        .power = 2.0f,             // Contrast adjustment
        .noiseSize = NOISE_SIZE,   // Noise texture size
        .blurSize = 4,             // Blur kernel size
        .rangeCheckScale = 0.5f    // Range check falloff
    };
    
    // Create SSAO data
    printf("Initializing SSAO with %d samples...\n", params.kernelSize);
    SSAOData* ssao = create_ssao_data(width, height, params);
    
    // Calculate SSAO
    printf("Calculating ambient occlusion...\n");
    calculate_ssao(gbuffer, ssao, &camera);
    
    // Save raw AO
    save_ppm("ssao_raw.ppm", ssao->aoBuffer, width, height, 1);
    
    // Apply bilateral blur
    printf("Applying bilateral blur...\n");
    bilateral_blur_ssao(gbuffer, ssao);
    
    // Save blurred AO
    save_ppm("ssao_blurred.ppm", ssao->blurredAO, width, height, 1);
    
    // Save depth buffer
    save_ppm("depth_buffer.ppm", gbuffer->depth, width, height, 0);
    
    // Save normals (for debugging)
    float* normalVis = (float*)malloc(width * height * sizeof(float) * 3);
    for (int i = 0; i < width * height; i++) {
        normalVis[i * 3 + 0] = gbuffer->normals[i].x * 0.5f + 0.5f;
        normalVis[i * 3 + 1] = gbuffer->normals[i].y * 0.5f + 0.5f;
        normalVis[i * 3 + 2] = gbuffer->normals[i].z * 0.5f + 0.5f;
    }
    save_color_ppm("normals.ppm", (Vec3*)normalVis, width, height);
    free(normalVis);
    
    // Create comparison images
    printf("Creating comparison images...\n");
    create_comparison_images(gbuffer, ssao, width, height);
    
    // Test different quality settings
    printf("\nGenerating quality comparison...\n");
    
    // Low quality
    params.kernelSize = 8;
    params.blurSize = 2;
    SSAOData* ssaoLow = create_ssao_data(width, height, params);
    calculate_ssao(gbuffer, ssaoLow, &camera);
    bilateral_blur_ssao(gbuffer, ssaoLow);
    save_ppm("ssao_low_quality.ppm", ssaoLow->blurredAO, width, height, 1);
    
    // High quality
    params.kernelSize = 64;
    params.blurSize = 8;
    SSAOData* ssaoHigh = create_ssao_data(width, height, params);
    calculate_ssao(gbuffer, ssaoHigh, &camera);
    bilateral_blur_ssao(gbuffer, ssaoHigh);
    save_ppm("ssao_high_quality.ppm", ssaoHigh->blurredAO, width, height, 1);
    
    // Cleanup
    destroy_gbuffer(gbuffer);
    destroy_ssao_data(ssao);
    destroy_ssao_data(ssaoLow);
    destroy_ssao_data(ssaoHigh);
    
    printf("\nGenerated files:\n");
    printf("  - ssao_raw.ppm (unblurred AO)\n");
    printf("  - ssao_blurred.ppm (final AO)\n");
    printf("  - ssao_low_quality.ppm (8 samples)\n");
    printf("  - ssao_high_quality.ppm (64 samples)\n");
    printf("  - scene_without_ao.ppm (no AO)\n");
    printf("  - scene_with_ao.ppm (with AO)\n");
    printf("  - depth_buffer.ppm (depth visualization)\n");
    printf("  - normals.ppm (normal visualization)\n");
    
    return 0;
}