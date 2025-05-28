#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define MAX_MIP_LEVELS 12
#define MAX_ANISOTROPY 16.0f
#define PI 3.14159265359f

// Texture filtering modes
typedef enum {
    FILTER_NEAREST,
    FILTER_LINEAR,
    FILTER_NEAREST_MIPMAP_NEAREST,
    FILTER_LINEAR_MIPMAP_NEAREST,
    FILTER_NEAREST_MIPMAP_LINEAR,
    FILTER_LINEAR_MIPMAP_LINEAR  // Trilinear
} FilterMode;

// Color structure
typedef struct {
    uint8_t r, g, b, a;
} Color;

// Floating point color for calculations
typedef struct {
    float r, g, b, a;
} ColorF;

// 2D Vector
typedef struct {
    float x, y;
} Vec2;

// Texture structure with mipmap support
typedef struct {
    int width, height;
    int mip_levels;
    Color** mip_data;  // Array of pointers to mipmap levels
    int* mip_widths;   // Width of each mip level
    int* mip_heights;  // Height of each mip level
    FilterMode filter_mode;
    float anisotropy;  // Anisotropic filtering level (1.0 = isotropic)
} Texture;

// Texture sampling context
typedef struct {
    Vec2 uv;           // Base texture coordinates
    Vec2 dudx, dudy;   // Partial derivatives (for anisotropic filtering)
    float lod_bias;    // Manual LOD bias
} SamplingContext;

// Create texture with mipmap support
Texture* create_texture(int width, int height) {
    Texture* tex = malloc(sizeof(Texture));
    tex->width = width;
    tex->height = height;
    tex->filter_mode = FILTER_LINEAR_MIPMAP_LINEAR;
    tex->anisotropy = 1.0f;
    
    // Calculate number of mip levels
    tex->mip_levels = (int)floor(log2(fmax(width, height))) + 1;
    tex->mip_levels = fmin(tex->mip_levels, MAX_MIP_LEVELS);
    
    // Allocate mipmap arrays
    tex->mip_data = malloc(tex->mip_levels * sizeof(Color*));
    tex->mip_widths = malloc(tex->mip_levels * sizeof(int));
    tex->mip_heights = malloc(tex->mip_levels * sizeof(int));
    
    // Initialize mip level dimensions
    for (int i = 0; i < tex->mip_levels; i++) {
        tex->mip_widths[i] = fmax(1, width >> i);
        tex->mip_heights[i] = fmax(1, height >> i);
        tex->mip_data[i] = calloc(tex->mip_widths[i] * tex->mip_heights[i], sizeof(Color));
    }
    
    return tex;
}

// Generate procedural checkerboard texture
void generate_checkerboard_texture(Texture* tex, int checker_size) {
    int width = tex->mip_widths[0];
    int height = tex->mip_heights[0];
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int checker_x = x / checker_size;
            int checker_y = y / checker_size;
            
            Color color;
            if ((checker_x + checker_y) % 2 == 0) {
                // White with slight color variation
                color = (Color){255, 255, 255, 255};
            } else {
                // Black with slight blue tint
                color = (Color){0, 0, 32, 255};
            }
            
            // Add some noise for detail
            float noise = (sinf(x * 0.1f) * cosf(y * 0.1f) + 1.0f) * 0.5f;
            color.r = (uint8_t)(color.r * (0.8f + noise * 0.2f));
            color.g = (uint8_t)(color.g * (0.8f + noise * 0.2f));
            color.b = (uint8_t)(color.b * (0.8f + noise * 0.2f));
            
            tex->mip_data[0][y * width + x] = color;
        }
    }
}

// Box filter for mipmap generation
ColorF box_filter(Color* src, int src_width, int src_height, float u, float v, float filter_size) {
    ColorF result = {0, 0, 0, 0};
    int samples = 0;
    
    int start_x = (int)((u - filter_size * 0.5f) * src_width);
    int end_x = (int)((u + filter_size * 0.5f) * src_width);
    int start_y = (int)((v - filter_size * 0.5f) * src_height);
    int end_y = (int)((v + filter_size * 0.5f) * src_height);
    
    start_x = fmax(0, start_x);
    end_x = fmin(src_width - 1, end_x);
    start_y = fmax(0, start_y);
    end_y = fmin(src_height - 1, end_y);
    
    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            Color pixel = src[y * src_width + x];
            result.r += pixel.r;
            result.g += pixel.g;
            result.b += pixel.b;
            result.a += pixel.a;
            samples++;
        }
    }
    
    if (samples > 0) {
        result.r /= samples;
        result.g /= samples;
        result.b /= samples;
        result.a /= samples;
    }
    
    return result;
}

// Generate mipmaps using box filtering
void generate_mipmaps(Texture* tex) {
    printf("Generating %d mipmap levels...\n", tex->mip_levels);
    
    for (int level = 1; level < tex->mip_levels; level++) {
        int src_width = tex->mip_widths[level - 1];
        int src_height = tex->mip_heights[level - 1];
        int dst_width = tex->mip_widths[level];
        int dst_height = tex->mip_heights[level];
        
        Color* src = tex->mip_data[level - 1];
        Color* dst = tex->mip_data[level];
        
        printf("  Level %d: %dx%d -> %dx%d\n", level, src_width, src_height, dst_width, dst_height);
        
        for (int y = 0; y < dst_height; y++) {
            for (int x = 0; x < dst_width; x++) {
                float u = (x + 0.5f) / dst_width;
                float v = (y + 0.5f) / dst_height;
                
                // Sample with 2x2 box filter
                ColorF filtered = box_filter(src, src_width, src_height, u, v, 2.0f / dst_width);
                
                dst[y * dst_width + x] = (Color){
                    (uint8_t)fmin(255, filtered.r),
                    (uint8_t)fmin(255, filtered.g),
                    (uint8_t)fmin(255, filtered.b),
                    (uint8_t)fmin(255, filtered.a)
                };
            }
        }
    }
}

// Calculate LOD (Level of Detail) from texture gradients
float calculate_lod(Vec2 dudx, Vec2 dudy, int tex_width, int tex_height) {
    // Convert UV derivatives to texel space
    float dudx_texel = dudx.x * tex_width;
    float dvdx_texel = dudx.y * tex_height;
    float dudy_texel = dudy.x * tex_width;
    float dvdy_texel = dudy.y * tex_height;
    
    // Calculate maximum rate of change
    float rho_x = sqrtf(dudx_texel * dudx_texel + dvdx_texel * dvdx_texel);
    float rho_y = sqrtf(dudy_texel * dudy_texel + dvdy_texel * dvdy_texel);
    float rho = fmaxf(rho_x, rho_y);
    
    // LOD is log2 of the maximum rate of change
    return log2f(fmaxf(rho, 1.0f));
}

// Bilinear interpolation
ColorF bilinear_sample(Color* data, int width, int height, float u, float v) {
    // Wrap coordinates
    u = u - floorf(u);
    v = v - floorf(v);
    
    // Convert to texel coordinates
    float x = u * (width - 1);
    float y = v * (height - 1);
    
    // Get integer and fractional parts
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % width;
    int y1 = (y0 + 1) % height;
    
    float fx = x - x0;
    float fy = y - y0;
    
    // Sample four neighboring pixels
    Color c00 = data[y0 * width + x0];
    Color c10 = data[y0 * width + x1];
    Color c01 = data[y1 * width + x0];
    Color c11 = data[y1 * width + x1];
    
    // Bilinear interpolation
    ColorF result;
    result.r = (1-fx)*(1-fy)*c00.r + fx*(1-fy)*c10.r + (1-fx)*fy*c01.r + fx*fy*c11.r;
    result.g = (1-fx)*(1-fy)*c00.g + fx*(1-fy)*c10.g + (1-fx)*fy*c01.g + fx*fy*c11.g;
    result.b = (1-fx)*(1-fy)*c00.b + fx*(1-fy)*c10.b + (1-fx)*fy*c01.b + fx*fy*c11.b;
    result.a = (1-fx)*(1-fy)*c00.a + fx*(1-fy)*c10.a + (1-fx)*fy*c01.a + fx*fy*c11.a;
    
    return result;
}

// Nearest neighbor sampling
ColorF nearest_sample(Color* data, int width, int height, float u, float v) {
    u = u - floorf(u);
    v = v - floorf(v);
    
    int x = (int)(u * width) % width;
    int y = (int)(v * height) % height;
    
    Color pixel = data[y * width + x];
    return (ColorF){pixel.r, pixel.g, pixel.b, pixel.a};
}

// Sample single mip level
ColorF sample_mip_level(Texture* tex, int level, float u, float v, bool use_linear) {
    if (level < 0) level = 0;
    if (level >= tex->mip_levels) level = tex->mip_levels - 1;
    
    Color* data = tex->mip_data[level];
    int width = tex->mip_widths[level];
    int height = tex->mip_heights[level];
    
    if (use_linear) {
        return bilinear_sample(data, width, height, u, v);
    } else {
        return nearest_sample(data, width, height, u, v);
    }
}

// Trilinear sampling (linear interpolation between mip levels)
ColorF trilinear_sample(Texture* tex, float u, float v, float lod) {
    // Clamp LOD to valid range
    lod = fmaxf(0.0f, fminf(lod, tex->mip_levels - 1));
    
    int level0 = (int)floorf(lod);
    int level1 = level0 + 1;
    float frac = lod - level0;
    
    // Sample both levels
    ColorF color0 = sample_mip_level(tex, level0, u, v, true);
    ColorF color1 = sample_mip_level(tex, level1, u, v, true);
    
    // Linear interpolation between levels
    ColorF result;
    result.r = color0.r * (1.0f - frac) + color1.r * frac;
    result.g = color0.g * (1.0f - frac) + color1.g * frac;
    result.b = color0.b * (1.0f - frac) + color1.b * frac;
    result.a = color0.a * (1.0f - frac) + color1.a * frac;
    
    return result;
}

// Anisotropic filtering implementation
ColorF anisotropic_sample(Texture* tex, SamplingContext* ctx) {
    if (tex->anisotropy <= 1.0f) {
        // Fall back to isotropic filtering
        float lod = calculate_lod(ctx->dudx, ctx->dudy, tex->width, tex->height);
        return trilinear_sample(tex, ctx->uv.x, ctx->uv.y, lod + ctx->lod_bias);
    }
    
    // Calculate anisotropy direction and ratio
    float dudx_len = sqrtf(ctx->dudx.x * ctx->dudx.x + ctx->dudx.y * ctx->dudx.y);
    float dudy_len = sqrtf(ctx->dudy.x * ctx->dudy.x + ctx->dudy.y * ctx->dudy.y);
    
    Vec2 major_axis, minor_axis;
    float major_len, minor_len;
    
    if (dudx_len > dudy_len) {
        major_axis = (Vec2){ctx->dudx.x / dudx_len, ctx->dudx.y / dudx_len};
        minor_axis = (Vec2){ctx->dudy.x / dudy_len, ctx->dudy.y / dudy_len};
        major_len = dudx_len;
        minor_len = dudy_len;
    } else {
        major_axis = (Vec2){ctx->dudy.x / dudy_len, ctx->dudy.y / dudy_len};
        minor_axis = (Vec2){ctx->dudx.x / dudx_len, ctx->dudx.y / dudx_len};
        major_len = dudy_len;
        minor_len = dudx_len;
    }
    
    // Calculate anisotropy ratio
    float aniso_ratio = fminf(major_len / fmaxf(minor_len, 0.0001f), tex->anisotropy);
    
    // Calculate LOD for minor axis
    float lod = log2f(fmaxf(minor_len * tex->width, 1.0f)) + ctx->lod_bias;
    
    // Number of samples along major axis
    int num_samples = (int)ceilf(aniso_ratio);
    num_samples = fmin(num_samples, 16); // Limit samples for performance
    
    ColorF result = {0, 0, 0, 0};
    
    // Sample along the major axis
    for (int i = 0; i < num_samples; i++) {
        float offset = (i - (num_samples - 1) * 0.5f) / num_samples;
        Vec2 sample_uv = {
            ctx->uv.x + major_axis.x * offset / tex->width,
            ctx->uv.y + major_axis.y * offset / tex->height
        };
        
        ColorF sample = trilinear_sample(tex, sample_uv.x, sample_uv.y, lod);
        result.r += sample.r;
        result.g += sample.g;
        result.b += sample.b;
        result.a += sample.a;
    }
    
    // Average the samples
    result.r /= num_samples;
    result.g /= num_samples;
    result.b /= num_samples;
    result.a /= num_samples;
    
    return result;
}

// Main texture sampling function
ColorF sample_texture(Texture* tex, SamplingContext* ctx) {
    switch (tex->filter_mode) {
        case FILTER_NEAREST:
            return sample_mip_level(tex, 0, ctx->uv.x, ctx->uv.y, false);
            
        case FILTER_LINEAR:
            return sample_mip_level(tex, 0, ctx->uv.x, ctx->uv.y, true);
            
        case FILTER_NEAREST_MIPMAP_NEAREST: {
            float lod = calculate_lod(ctx->dudx, ctx->dudy, tex->width, tex->height);
            int level = (int)roundf(lod + ctx->lod_bias);
            return sample_mip_level(tex, level, ctx->uv.x, ctx->uv.y, false);
        }
        
        case FILTER_LINEAR_MIPMAP_NEAREST: {
            float lod = calculate_lod(ctx->dudx, ctx->dudy, tex->width, tex->height);
            int level = (int)roundf(lod + ctx->lod_bias);
            return sample_mip_level(tex, level, ctx->uv.x, ctx->uv.y, true);
        }
        
        case FILTER_NEAREST_MIPMAP_LINEAR: {
            float lod = calculate_lod(ctx->dudx, ctx->dudy, tex->width, tex->height);
            lod += ctx->lod_bias;
            
            int level0 = (int)floorf(lod);
            int level1 = level0 + 1;
            float frac = lod - level0;
            
            ColorF color0 = sample_mip_level(tex, level0, ctx->uv.x, ctx->uv.y, false);
            ColorF color1 = sample_mip_level(tex, level1, ctx->uv.x, ctx->uv.y, false);
            
            ColorF result;
            result.r = color0.r * (1.0f - frac) + color1.r * frac;
            result.g = color0.g * (1.0f - frac) + color1.g * frac;
            result.b = color0.b * (1.0f - frac) + color1.b * frac;
            result.a = color0.a * (1.0f - frac) + color1.a * frac;
            return result;
        }
        
        case FILTER_LINEAR_MIPMAP_LINEAR: {
            if (tex->anisotropy > 1.0f) {
                return anisotropic_sample(tex, ctx);
            } else {
                float lod = calculate_lod(ctx->dudx, ctx->dudy, tex->width, tex->height);
                return trilinear_sample(tex, ctx->uv.x, ctx->uv.y, lod + ctx->lod_bias);
            }
        }
    }
    
    return (ColorF){255, 0, 255, 255}; // Magenta error color
}

// Render test pattern to demonstrate filtering
typedef struct {
    int width, height;
    Color* pixels;
} Framebuffer;

Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = malloc(sizeof(Framebuffer));
    fb->width = width;
    fb->height = height;
    fb->pixels = calloc(width * height, sizeof(Color));
    return fb;
}

void render_perspective_plane(Framebuffer* fb, Texture* tex, float distance) {
    printf("Rendering perspective plane at distance %.1f...\n", distance);
    
    for (int y = 0; y < fb->height; y++) {
        for (int x = 0; x < fb->width; x++) {
            // Convert screen coordinates to world coordinates
            float screen_x = (x / (float)fb->width - 0.5f) * 2.0f;
            float screen_y = (y / (float)fb->height - 0.5f) * 2.0f;
            
            // Calculate UV coordinates with perspective
            float u = screen_x * distance + 0.5f;
            float v = screen_y * distance + 0.5f;
            
            // Calculate texture gradients (partial derivatives)
            float pixel_size = 1.0f / fb->width;
            Vec2 dudx = {pixel_size * distance, 0};
            Vec2 dudy = {0, pixel_size * distance};
            
            // Create sampling context
            SamplingContext ctx = {
                .uv = {u, v},
                .dudx = dudx,
                .dudy = dudy,
                .lod_bias = 0.0f
            };
            
            // Sample texture
            ColorF color = sample_texture(tex, &ctx);
            
            // Convert to 8-bit and store
            fb->pixels[y * fb->width + x] = (Color){
                (uint8_t)fmin(255, color.r),
                (uint8_t)fmin(255, color.g),
                (uint8_t)fmin(255, color.b),
                (uint8_t)fmin(255, color.a)
            };
        }
    }
}

void render_anisotropic_test(Framebuffer* fb, Texture* tex) {
    printf("Rendering anisotropic filtering test...\n");
    
    for (int y = 0; y < fb->height; y++) {
        for (int x = 0; x < fb->width; x++) {
            // Create anisotropic viewing condition
            float u = x / (float)fb->width;
            float v = y / (float)fb->height;
            
            // Simulate oblique viewing angle with different gradients
            float angle = PI * 0.25f; // 45 degree angle
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            
            Vec2 dudx = {cos_a / fb->width, sin_a / fb->width};
            Vec2 dudy = {-sin_a * 0.1f / fb->height, cos_a * 0.1f / fb->height};
            
            SamplingContext ctx = {
                .uv = {u * 4.0f, v * 4.0f}, // Scale up UVs to show detail
                .dudx = dudx,
                .dudy = dudy,
                .lod_bias = 0.0f
            };
            
            ColorF color = sample_texture(tex, &ctx);
            
            fb->pixels[y * fb->width + x] = (Color){
                (uint8_t)fmin(255, color.r),
                (uint8_t)fmin(255, color.g),
                (uint8_t)fmin(255, color.b),
                (uint8_t)fmin(255, color.a)
            };
        }
    }
}

void save_ppm(Framebuffer* fb, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fprintf(f, "P6\n%d %d\n255\n", fb->width, fb->height);
    for (int i = 0; i < fb->width * fb->height; i++) {
        fwrite(&fb->pixels[i], 3, 1, f);
    }
    
    fclose(f);
    printf("Saved %s\n", filename);
}

void save_mip_level(Texture* tex, int level, const char* filename) {
    if (level >= tex->mip_levels) return;
    
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    int width = tex->mip_widths[level];
    int height = tex->mip_heights[level];
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        fwrite(&tex->mip_data[level][i], 3, 1, f);
    }
    
    fclose(f);
    printf("Saved mip level %d (%dx%d) to %s\n", level, width, height, filename);
}

void analyze_filtering_performance(Texture* tex) {
    printf("\nPerformance Analysis:\n");
    printf("====================\n");
    
    const int test_samples = 100000;
    SamplingContext ctx = {
        .uv = {0.5f, 0.5f},
        .dudx = {0.001f, 0.0f},
        .dudy = {0.0f, 0.001f},
        .lod_bias = 0.0f
    };
    
    // Test different filtering modes
    FilterMode modes[] = {
        FILTER_NEAREST,
        FILTER_LINEAR,
        FILTER_LINEAR_MIPMAP_NEAREST,
        FILTER_LINEAR_MIPMAP_LINEAR
    };
    
    const char* mode_names[] = {
        "Nearest",
        "Linear",
        "Linear + Mipmap Nearest",
        "Trilinear"
    };
    
    for (int i = 0; i < 4; i++) {
        tex->filter_mode = modes[i];
        
        clock_t start = clock();
        for (int j = 0; j < test_samples; j++) {
            ctx.uv.x = (j % 1000) / 1000.0f;
            ctx.uv.y = (j / 1000) / 100.0f;
            sample_texture(tex, &ctx);
        }
        clock_t end = clock();
        
        double time = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("%s: %.3f seconds (%.1f samples/sec)\n", 
               mode_names[i], time, test_samples / time);
    }
    
    // Test anisotropic filtering
    tex->filter_mode = FILTER_LINEAR_MIPMAP_LINEAR;
    tex->anisotropy = 8.0f;
    
    clock_t start = clock();
    for (int j = 0; j < test_samples / 10; j++) { // Fewer samples due to higher cost
        ctx.uv.x = (j % 100) / 100.0f;
        ctx.uv.y = (j / 100) / 1000.0f;
        ctx.dudx = (Vec2){0.01f, 0.0f};
        ctx.dudy = (Vec2){0.0f, 0.001f};
        sample_texture(tex, &ctx);
    }
    clock_t end = clock();
    
    double time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Anisotropic 8x: %.3f seconds (%.1f samples/sec)\n", 
           time, (test_samples / 10) / time);
}

void demonstrate_mipmap_selection(Texture* tex) {
    printf("\nMipmap Selection Demonstration:\n");
    printf("==============================\n");
    
    SamplingContext ctx = {
        .lod_bias = 0.0f
    };
    
    float distances[] = {0.1f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
    
    for (int i = 0; i < 6; i++) {
        ctx.uv = (Vec2){0.5f, 0.5f};
        ctx.dudx = (Vec2){distances[i] / tex->width, 0};
        ctx.dudy = (Vec2){0, distances[i] / tex->height};
        
        float lod = calculate_lod(ctx.dudx, ctx.dudy, tex->width, tex->height);
        
        printf("Distance %.1f -> LOD %.2f (Mip level %d)\n", 
               distances[i], lod, (int)roundf(lod));
    }
}

void free_texture(Texture* tex) {
    if (tex) {
        for (int i = 0; i < tex->mip_levels; i++) {
            free(tex->mip_data[i]);
        }
        free(tex->mip_data);
        free(tex->mip_widths);
        free(tex->mip_heights);
        free(tex);
    }
}

void free_framebuffer(Framebuffer* fb) {
    if (fb) {
        free(fb->pixels);
        free(fb);
    }
}

int main() {
    printf("Mipmapping and Anisotropic Filtering Demo\n");
    printf("=========================================\n\n");
    
    // Create test texture
    const int tex_size = 512;
    Texture* tex = create_texture(tex_size, tex_size);
    
    // Generate checkerboard pattern
    generate_checkerboard_texture(tex, 16);
    
    // Generate mipmaps
    generate_mipmaps(tex);
    
    // Save individual mip levels for inspection
    printf("\nSaving mipmap levels:\n");
    for (int i = 0; i < fmin(6, tex->mip_levels); i++) {
        char filename[64];
        snprintf(filename, sizeof(filename), "mip_level_%d.ppm", i);
        save_mip_level(tex, i, filename);
    }
    
    // Demonstrate mipmap selection
    demonstrate_mipmap_selection(tex);
    
    // Create framebuffer for rendering tests
    Framebuffer* fb = create_framebuffer(800, 600);
    
    // Test 1: No mipmapping (aliasing artifacts)
    printf("\nTest 1: No mipmapping (nearest filtering)\n");
    tex->filter_mode = FILTER_NEAREST;
    render_perspective_plane(fb, tex, 4.0f);
    save_ppm(fb, "no_mipmap.ppm");
    
    // Test 2: Linear filtering only
    printf("\nTest 2: Linear filtering (no mipmapping)\n");
    tex->filter_mode = FILTER_LINEAR;
    render_perspective_plane(fb, tex, 4.0f);
    save_ppm(fb, "linear_only.ppm");
    
    // Test 3: Trilinear filtering
    printf("\nTest 3: Trilinear filtering\n");
    tex->filter_mode = FILTER_LINEAR_MIPMAP_LINEAR;
    tex->anisotropy = 1.0f;
    render_perspective_plane(fb, tex, 4.0f);
    save_ppm(fb, "trilinear.ppm");
    
    // Test 4: Anisotropic filtering 4x
    printf("\nTest 4: Anisotropic filtering 4x\n");
    tex->anisotropy = 4.0f;
    render_anisotropic_test(fb, tex);
    save_ppm(fb, "anisotropic_4x.ppm");
    
    // Test 5: Anisotropic filtering 16x
    printf("\nTest 5: Anisotropic filtering 16x\n");
    tex->anisotropy = 16.0f;
    render_anisotropic_test(fb, tex);
    save_ppm(fb, "anisotropic_16x.ppm");
    
    // Test different distances for perspective comparison
    printf("\nGenerating perspective comparison images:\n");
    float test_distances[] = {0.5f, 1.0f, 2.0f, 4.0f};
    const char* distance_names[] = {"close", "medium", "far", "very_far"};
    
    tex->filter_mode = FILTER_LINEAR_MIPMAP_LINEAR;
    tex->anisotropy = 8.0f;
    
    for (int i = 0; i < 4; i++) {
        render_perspective_plane(fb, tex, test_distances[i]);
        
        char filename[64];
        snprintf(filename, sizeof(filename), "perspective_%s.ppm", distance_names[i]);
        save_ppm(fb, filename);
    }
    
    // Performance analysis
    analyze_filtering_performance(tex);
    
    // Print memory usage analysis
    printf("\nMemory Usage Analysis:\n");
    printf("=====================\n");
    
    size_t total_memory = 0;
    for (int i = 0; i < tex->mip_levels; i++) {
        size_t level_memory = tex->mip_widths[i] * tex->mip_heights[i] * sizeof(Color);
        total_memory += level_memory;
        printf("Level %d (%dx%d): %zu bytes\n", i, tex->mip_widths[i], tex->mip_heights[i], level_memory);
    }
    
    size_t base_memory = tex->width * tex->height * sizeof(Color);
    float overhead = ((float)total_memory / base_memory - 1.0f) * 100.0f;
    
    printf("Base texture: %zu bytes\n", base_memory);
    printf("Total with mipmaps: %zu bytes\n", total_memory);
    printf("Memory overhead: %.1f%%\n", overhead);
    
    printf("\nFiltering Quality Comparison:\n");
    printf("============================\n");
    printf("1. no_mipmap.ppm - Shows aliasing artifacts\n");
    printf("2. linear_only.ppm - Reduced aliasing but still present\n");
    printf("3. trilinear.ppm - Smooth LOD transitions\n");
    printf("4. anisotropic_4x.ppm - Better oblique angle quality\n");
    printf("5. anisotropic_16x.ppm - Highest oblique angle quality\n");
    printf("6. perspective_*.ppm - Distance-based quality demonstration\n");
    
    // Cleanup
    free_texture(tex);
    free_framebuffer(fb);
    
    printf("\nDemo completed successfully!\n");
    printf("Generated images demonstrate the visual and performance\n");
    printf("benefits of mipmapping and anisotropic filtering.\n");
    
    return 0;
}