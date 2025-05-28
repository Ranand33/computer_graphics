#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define PI 3.14159265359f
#define FRAMEBUFFER_WIDTH 800
#define FRAMEBUFFER_HEIGHT 600
#define BLOOM_KERNEL_SIZE 5
#define LUT_SIZE 16 // 16x16x16 3D LUT

// Basic math structures
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float x, y;
} Vec2;

// Framebuffer pixel
typedef struct {
    Vec3 color; // RGB color
    float depth; // Depth value
    Vec2 velocity; // Motion vector (screen space)
} Pixel;

// Framebuffer structure
typedef struct {
    int width, height;
    Pixel* data;
} Framebuffer;

// Bloom parameters
typedef struct {
    float threshold; // Brightness threshold for bloom
    float intensity; // Bloom intensity
    int kernelSize; // Gaussian blur kernel size
    float sigma; // Gaussian blur sigma
} BloomParams;

// Motion blur parameters
typedef struct {
    int samples; // Number of samples along velocity vector
    float maxVelocity; // Maximum velocity for clamping
} MotionBlurParams;

// Color grading LUT
typedef struct {
    Vec3 data[LUT_SIZE][LUT_SIZE][LUT_SIZE]; // 3D RGB LUT
} ColorLUT;

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

Vec3 vec3_clamp(Vec3 v, float min, float max) {
    return (Vec3){
        fmaxf(min, fminf(max, v.x)),
        fmaxf(min, fminf(max, v.y)),
        fmaxf(min, fminf(max, v.z))
    };
}

// Framebuffer operations
Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = (Framebuffer*)malloc(sizeof(Framebuffer));
    fb->width = width;
    fb->height = height;
    fb->data = (Pixel*)calloc(width * height, sizeof(Pixel));
    return fb;
}

void destroy_framebuffer(Framebuffer* fb) {
    free(fb->data);
    free(fb);
}

void clear_framebuffer(Framebuffer* fb) {
    memset(fb->data, 0, fb->width * fb->height * sizeof(Pixel));
    for (int i = 0; i < fb->width * fb->height; i++) {
        fb->data[i].depth = FLT_MAX;
    }
}

// Gaussian kernel for bloom
float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma)) / (sqrtf(2.0f * PI) * sigma);
}

void generate_gaussian_kernel(int size, float sigma, float* kernel) {
    float sum = 0.0f;
    int halfSize = size / 2;
    
    for (int i = -halfSize; i <= halfSize; i++) {
        kernel[i + halfSize] = gaussian((float)i, sigma);
        sum += kernel[i + halfSize];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

// Bloom effect
void apply_bloom(Framebuffer* src, Framebuffer* dst, BloomParams params) {
    // Temporary buffer for bright pass and blur
    Framebuffer* brightPass = create_framebuffer(src->width, src->height);
    Framebuffer* temp = create_framebuffer(src->width, src->height);
    
    // Bright pass: Extract pixels above threshold
    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            int idx = y * src->width + x;
            Vec3 color = src->data[idx].color;
            float brightness = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
            
            if (brightness > params.threshold) {
                brightPass->data[idx].color = vec3_mul(color, (brightness - params.threshold) / brightness);
            } else {
                brightPass->data[idx].color = (Vec3){0, 0, 0};
            }
        }
    }
    
    // Generate Gaussian kernel
    float kernel[BLOOM_KERNEL_SIZE];
    generate_gaussian_kernel(params.kernelSize, params.sigma, kernel);
    
    // Horizontal blur
    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            Vec3 sum = {0, 0, 0};
            for (int k = -params.kernelSize / 2; k <= params.kernelSize / 2; k++) {
                int srcX = x + k;
                if (srcX >= 0 && srcX < src->width) {
                    int srcIdx = y * src->width + srcX;
                    sum = vec3_add(sum, vec3_mul(brightPass->data[srcIdx].color, kernel[k + params.kernelSize / 2]));
                }
            }
            temp->data[y * src->width + x].color = sum;
        }
    }
    
    // Vertical blur
    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            Vec3 sum = {0, 0, 0};
            for (int k = -params.kernelSize / 2; k <= params.kernelSize / 2; k++) {
                int srcY = y + k;
                if (srcY >= 0 && srcY < src->height) {
                    int srcIdx = srcY * src->width + x;
                    sum = vec3_add(sum, vec3_mul(temp->data[srcIdx].color, kernel[k + params.kernelSize / 2]));
                }
            }
            brightPass->data[y * src->width + x].color = sum;
        }
    }
    
    // Composite bloom with original image
    for (int i = 0; i < src->width * src->height; i++) {
        Vec3 bloomColor = vec3_mul(brightPass->data[i].color, params.intensity);
        dst->data[i].color = vec3_add(src->data[i].color, bloomColor);
        dst->data[i].depth = src->data[i].depth;
        dst->data[i].velocity = src->data[i].velocity;
    }
    
    destroy_framebuffer(brightPass);
    destroy_framebuffer(temp);
}

// Motion blur effect
void apply_motion_blur(Framebuffer* src, Framebuffer* dst, MotionBlurParams params) {
    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            int idx = y * src->width + x;
            Vec2 velocity = src->data[idx].velocity;
            
            // Clamp velocity
            float velLen = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);
            if (velLen > params.maxVelocity) {
                velocity.x *= params.maxVelocity / velLen;
                velocity.y *= params.maxVelocity / velLen;
            }
            
            Vec3 sum = {0, 0, 0};
            int validSamples = 0;
            
            // Sample along velocity vector
            for (int s = -params.samples / 2; s <= params.samples / 2; s++) {
                float t = (float)s / (params.samples / 2);
                int sampleX = x + (int)(velocity.x * t);
                int sampleY = y + (int)(velocity.y * t);
                
                if (sampleX >= 0 && sampleX < src->width && sampleY >= 0 && sampleY < src->height) {
                    int sampleIdx = sampleY * src->width + sampleX;
                    sum = vec3_add(sum, src->data[sampleIdx].color);
                    validSamples++;
                }
            }
            
            if (validSamples > 0) {
                dst->data[idx].color = vec3_mul(sum, 1.0f / validSamples);
            } else {
                dst->data[idx].color = src->data[idx].color;
            }
            dst->data[idx].depth = src->data[idx].depth;
            dst->data[idx].velocity = src->data[idx].velocity;
        }
    }
}

// Color grading with 3D LUT
void init_color_lut(ColorLUT* lut) {
    // Simple cinematic LUT: Increase contrast, add slight blue tint
    for (int r = 0; r < LUT_SIZE; r++) {
        for (int g = 0; g < LUT_SIZE; g++) {
            for (int b = 0; b < LUT_SIZE; b++) {
                float rn = (float)r / (LUT_SIZE - 1);
                float gn = (float)g / (LUT_SIZE - 1);
                float bn = (float)b / (LUT_SIZE - 1);
                
                // Sigmoid contrast
                rn = 1.0f / (1.0f + expf(-10.0f * (rn - 0.5f)));
                gn = 1.0f / (1.0f + expf(-10.0f * (gn - 0.5f)));
                bn = 1.0f / (1.0f + expf(-10.0f * (bn - 0.5f)));
                
                // Blue tint
                bn *= 1.1f;
                rn *= 0.95f;
                gn *= 0.95f;
                
                lut->data[r][g][b] = (Vec3){
                    fmaxf(0.0f, fminf(1.0f, rn)),
                    fmaxf(0.0f, fminf(1.0f, gn)),
                    fmaxf(0.0f, fminf(1.0f, bn))
                };
            }
        }
    }
}

void apply_color_grading(Framebuffer* src, Framebuffer* dst, ColorLUT* lut) {
    for (int i = 0; i < src->width * src->height; i++) {
        Vec3 color = vec3_clamp(src->data[i].color, 0.0f, 1.0f);
        
        // Map to LUT coordinates
        float r = color.x * (LUT_SIZE - 1);
        float g = color.y * (LUT_SIZE - 1);
        float b = color.z * (LUT_SIZE - 1);
        
        int r0 = (int)r;
        int g0 = (int)g;
        int b0 = (int)b;
        int r1 = r0 + 1;
        int g1 = g0 + 1;
        int b1 = b0 + 1;
        
        r1 = r1 < LUT_SIZE ? r1 : r0;
        g1 = g1 < LUT_SIZE ? g1 : g0;
        b1 = b1 < LUT_SIZE ? b1 : b0;
        
        float fr = r - r0;
        float fg = g - g0;
        float fb = b - b0;
        
        // Trilinear interpolation
        Vec3 c000 = lut->data[r0][g0][b0];
        Vec3 c100 = lut->data[r1][g0][b0];
        Vec3 c010 = lut->data[r0][g1][b0];
        Vec3 c110 = lut->data[r1][g1][b0];
        Vec3 c001 = lut->data[r0][g0][b1];
        Vec3 c101 = lut->data[r1][g0][b1];
        Vec3 c011 = lut->data[r0][g1][b1];
        Vec3 c111 = lut->data[r1][g1][b1];
        
        Vec3 c00 = vec3_add(vec3_mul(c000, 1.0f - fr), vec3_mul(c100, fr));
        Vec3 c01 = vec3_add(vec3_mul(c001, 1.0f - fr), vec3_mul(c101, fr));
        Vec3 c10 = vec3_add(vec3_mul(c010, 1.0f - fr), vec3_mul(c110, fr));
        Vec3 c11 = vec3_add(vec3_mul(c011, 1.0f - fr), vec3_mul(c111, fr));
        
        Vec3 c0 = vec3_add(vec3_mul(c00, 1.0f - fg), vec3_mul(c10, fg));
        Vec3 c1 = vec3_add(vec3_mul(c01, 1.0f - fg), vec3_mul(c11, fg));
        
        Vec3 finalColor = vec3_add(vec3_mul(c0, 1.0f - fb), vec3_mul(c1, fb));
        
        dst->data[i].color = finalColor;
        dst->data[i].depth = src->data[i].depth;
        dst->data[i].velocity = src->data[i].velocity;
    }
}

// Save framebuffer as PPM
void save_framebuffer_ppm(const char* filename, Framebuffer* fb) {
    FILE* file = fopen(filename, "wb");
    if (!file) return;
    
    fprintf(file, "P6\n%d %d\n255\n", fb->width, fb->height);
    
    for (int i = 0; i < fb->width * fb->height; i++) {
        Vec3 color = vec3_clamp(fb->data[i].color, 0.0f, 1.0f);
        fputc((unsigned char)(color.x * 255), file);
        fputc((unsigned char)(color.y * 255), file);
        fputc((unsigned char)(color.z * 255), file);
    }
    
    fclose(file);
}

// Initialize a test scene
void init_test_scene(Framebuffer* fb) {
    // Simulate a scene with some bright areas and motion
    for (int y = 0; y < fb->height; y++) {
        for (int x = 0; x < fb->width; x++) {
            int idx = y * fb->width + x;
            float u = (float)x / fb->width;
            float v = (float)y / fb->height;
            
            // Base color: Gradient with some bright spots
            Vec3 color = (Vec3){u, v, 0.5f};
            
            // Add bright spot for bloom
            float dx = u - 0.5f;
            float dy = v - 0.5f;
            if (dx * dx + dy * dy < 0.05f) {
                color = (Vec3){2.0f, 2.0f, 2.0f}; // Bright white
            }
            
            // Add moving object (simulated cube)
            if (fabsf(u - 0.7f) < 0.1f && fabsf(v - 0.3f) < 0.1f) {
                color = (Vec3){1.0f, 0.5f, 0.5f};
                fb->data[idx].velocity = (Vec2){10.0f, 5.0f}; // Screen-space motion
            }
            
            fb->data[idx].color = color;
            fb->data[idx].depth = 1.0f - v; // Simple depth gradient
        }
    }
}

int main() {
    printf("Post-Processing Demo\n");
    printf("===================\n\n");
    
    // Create framebuffers
    Framebuffer* mainBuffer = create_framebuffer(FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT);
    Framebuffer* tempBuffer = create_framebuffer(FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT);
    
    // Initialize test scene
    printf("Initializing test scene...\n");
    init_test_scene(mainBuffer);
    save_framebuffer_ppm("scene_original.ppm", mainBuffer);
    
    // Apply bloom
    printf("Applying bloom effect...\n");
    BloomParams bloomParams = {
        .threshold = 1.0f,
        .intensity = 0.5f,
        .kernelSize = BLOOM_KERNEL_SIZE,
        .sigma = 2.0f
    };
    apply_bloom(mainBuffer, tempBuffer, bloomParams);
    save_framebuffer_ppm("scene_bloom.ppm", tempBuffer);
    
    // Apply motion blur
    printf("Applying motion blur effect...\n");
    MotionBlurParams blurParams = {
        .samples = 8,
        .maxVelocity = 20.0f
    };
    apply_motion_blur(tempBuffer, mainBuffer, blurParams);
    save_framebuffer_ppm("scene_motion_blur.ppm", mainBuffer);
    
    // Apply color grading
    printf("Applying color grading effect...\n");
    ColorLUT lut;
    init_color_lut(&lut);
    apply_color_grading(mainBuffer, tempBuffer, &lut);
    save_framebuffer_ppm("scene_final.ppm", tempBuffer);
    
    // Cleanup
    destroy_framebuffer(mainBuffer);
    destroy_framebuffer(tempBuffer);
    
    printf("\nGenerated files:\n");
    printf("  - scene_original.ppm (original scene)\n");
    printf("  - scene_bloom.ppm (after bloom)\n");
    printf("  - scene_motion_blur.ppm (after motion blur)\n");
    printf("  - scene_final.ppm (after color grading)\n");
    
    return 0;
}