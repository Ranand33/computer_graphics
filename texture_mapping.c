#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// RGB color structure
typedef struct {
    uint8_t r, g, b;
} Color;

// 2D vector for UV coordinates
typedef struct {
    float u, v;
} Vec2;

// 3D vector for vertices
typedef struct {
    float x, y, z;
} Vec3;

// Vertex with position and texture coordinates
typedef struct {
    Vec3 pos;
    Vec2 uv;
} Vertex;

// Texture structure
typedef struct {
    int width, height;
    Color* pixels;
} Texture;

// Framebuffer structure
typedef struct {
    int width, height;
    Color* pixels;
} Framebuffer;

// Create a simple procedural texture (checkerboard pattern)
Texture* create_checkerboard_texture(int width, int height, int checker_size) {
    Texture* tex = malloc(sizeof(Texture));
    tex->width = width;
    tex->height = height;
    tex->pixels = malloc(width * height * sizeof(Color));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int checker_x = x / checker_size;
            int checker_y = y / checker_size;
            
            Color color;
            if ((checker_x + checker_y) % 2 == 0) {
                color = (Color){255, 255, 255}; // White
            } else {
                color = (Color){0, 0, 0};       // Black
            }
            
            tex->pixels[y * width + x] = color;
        }
    }
    
    return tex;
}

// Create a gradient texture
Texture* create_gradient_texture(int width, int height) {
    Texture* tex = malloc(sizeof(Texture));
    tex->width = width;
    tex->height = height;
    tex->pixels = malloc(width * height * sizeof(Color));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float u = (float)x / (width - 1);
            float v = (float)y / (height - 1);
            
            Color color = {
                (uint8_t)(u * 255),
                (uint8_t)(v * 255),
                (uint8_t)((1.0f - u) * 255)
            };
            
            tex->pixels[y * width + x] = color;
        }
    }
    
    return tex;
}

// Clamp value between 0 and 1
float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Sample texture with bilinear interpolation
Color sample_texture(const Texture* tex, float u, float v) {
    // Wrap UV coordinates
    u = u - floor(u);
    v = v - floor(v);
    
    // Convert to pixel coordinates
    float x = u * (tex->width - 1);
    float y = v * (tex->height - 1);
    
    // Get integer parts
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = (x0 + 1) % tex->width;
    int y1 = (y0 + 1) % tex->height;
    
    // Get fractional parts
    float fx = x - x0;
    float fy = y - y0;
    
    // Sample four neighboring pixels
    Color c00 = tex->pixels[y0 * tex->width + x0];
    Color c10 = tex->pixels[y0 * tex->width + x1];
    Color c01 = tex->pixels[y1 * tex->width + x0];
    Color c11 = tex->pixels[y1 * tex->width + x1];
    
    // Bilinear interpolation
    Color result;
    result.r = (uint8_t)(
        c00.r * (1 - fx) * (1 - fy) +
        c10.r * fx * (1 - fy) +
        c01.r * (1 - fx) * fy +
        c11.r * fx * fy
    );
    result.g = (uint8_t)(
        c00.g * (1 - fx) * (1 - fy) +
        c10.g * fx * (1 - fy) +
        c01.g * (1 - fx) * fy +
        c11.g * fx * fy
    );
    result.b = (uint8_t)(
        c00.b * (1 - fx) * (1 - fy) +
        c10.b * fx * (1 - fy) +
        c01.b * (1 - fx) * fy +
        c11.b * fx * fy
    );
    
    return result;
}

// Barycentric coordinates calculation
Vec3 barycentric(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    Vec2 v0 = {c.u - a.u, c.v - a.v};
    Vec2 v1 = {b.u - a.u, b.v - a.v};
    Vec2 v2 = {p.u - a.u, p.v - a.v};
    
    float dot00 = v0.u * v0.u + v0.v * v0.v;
    float dot01 = v0.u * v1.u + v0.v * v1.v;
    float dot02 = v0.u * v2.u + v0.v * v2.v;
    float dot11 = v1.u * v1.u + v1.v * v1.v;
    float dot12 = v1.u * v2.u + v1.v * v2.v;
    
    float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    return (Vec3){1.0f - u - v, v, u};
}

// Check if point is inside triangle
int point_in_triangle(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    Vec3 bary = barycentric(p, a, b, c);
    return (bary.x >= 0) && (bary.y >= 0) && (bary.z >= 0);
}

// Render a textured triangle using scanline rasterization
void render_textured_triangle(Framebuffer* fb, const Texture* tex, 
                             Vertex v0, Vertex v1, Vertex v2) {
    // Convert 3D positions to 2D screen coordinates (simple orthographic projection)
    Vec2 p0 = {v0.pos.x, v0.pos.y};
    Vec2 p1 = {v1.pos.x, v1.pos.y};
    Vec2 p2 = {v2.pos.x, v2.pos.y};
    
    // Find bounding box
    float min_x = fminf(fminf(p0.u, p1.u), p2.u);
    float max_x = fmaxf(fmaxf(p0.u, p1.u), p2.u);
    float min_y = fminf(fminf(p0.v, p1.v), p2.v);
    float max_y = fmaxf(fmaxf(p0.v, p1.v), p2.v);
    
    // Clip to framebuffer bounds
    int start_x = (int)fmaxf(0, floor(min_x));
    int end_x = (int)fminf(fb->width - 1, ceil(max_x));
    int start_y = (int)fmaxf(0, floor(min_y));
    int end_y = (int)fminf(fb->height - 1, ceil(max_y));
    
    // Rasterize triangle
    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            Vec2 pixel = {(float)x + 0.5f, (float)y + 0.5f};
            
            // Check if pixel is inside triangle
            Vec3 bary = barycentric(pixel, p0, p1, p2);
            
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                // Interpolate UV coordinates using barycentric coordinates
                float u = bary.x * v0.uv.u + bary.y * v1.uv.u + bary.z * v2.uv.u;
                float v = bary.x * v0.uv.v + bary.y * v1.uv.v + bary.z * v2.uv.v;
                
                // Sample texture
                Color texel = sample_texture(tex, u, v);
                
                // Set pixel
                fb->pixels[y * fb->width + x] = texel;
            }
        }
    }
}

// Create framebuffer
Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = malloc(sizeof(Framebuffer));
    fb->width = width;
    fb->height = height;
    fb->pixels = calloc(width * height, sizeof(Color));
    return fb;
}

// Clear framebuffer
void clear_framebuffer(Framebuffer* fb, Color color) {
    for (int i = 0; i < fb->width * fb->height; i++) {
        fb->pixels[i] = color;
    }
}

// Save framebuffer as PPM image
void save_ppm(const Framebuffer* fb, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    fprintf(f, "P6\n%d %d\n255\n", fb->width, fb->height);
    
    for (int i = 0; i < fb->width * fb->height; i++) {
        fwrite(&fb->pixels[i], sizeof(Color), 1, f);
    }
    
    fclose(f);
    printf("Saved framebuffer to %s\n", filename);
}

// Free memory
void free_texture(Texture* tex) {
    if (tex) {
        free(tex->pixels);
        free(tex);
    }
}

void free_framebuffer(Framebuffer* fb) {
    if (fb) {
        free(fb->pixels);
        free(fb);
    }
}

// Example usage
int main() {
    printf("Texture Mapping Demo\n");
    printf("===================\n");
    
    // Create framebuffer
    Framebuffer* fb = create_framebuffer(800, 600);
    clear_framebuffer(fb, (Color){64, 128, 192}); // Sky blue background
    
    // Create textures
    Texture* checkerboard = create_checkerboard_texture(64, 64, 8);
    Texture* gradient = create_gradient_texture(128, 128);
    
    // Define triangles with texture coordinates
    Vertex triangle1[3] = {
        {{100, 100, 0}, {0.0f, 0.0f}},
        {{300, 100, 0}, {1.0f, 0.0f}},
        {{200, 300, 0}, {0.5f, 1.0f}}
    };
    
    Vertex triangle2[3] = {
        {{400, 150, 0}, {0.0f, 0.0f}},
        {{600, 200, 0}, {2.0f, 0.0f}}, // UV > 1 will wrap
        {{500, 400, 0}, {1.0f, 2.0f}}
    };
    
    Vertex quad[6] = {
        // First triangle of quad
        {{150, 350, 0}, {0.0f, 0.0f}},
        {{350, 350, 0}, {1.0f, 0.0f}},
        {{150, 550, 0}, {0.0f, 1.0f}},
        // Second triangle of quad
        {{350, 350, 0}, {1.0f, 0.0f}},
        {{350, 550, 0}, {1.0f, 1.0f}},
        {{150, 550, 0}, {0.0f, 1.0f}}
    };
    
    printf("Rendering textured triangles...\n");
    
    // Render triangles with different textures
    render_textured_triangle(fb, checkerboard, triangle1[0], triangle1[1], triangle1[2]);
    render_textured_triangle(fb, checkerboard, triangle2[0], triangle2[1], triangle2[2]);
    
    // Render textured quad (two triangles)
    render_textured_triangle(fb, gradient, quad[0], quad[1], quad[2]);
    render_textured_triangle(fb, gradient, quad[3], quad[4], quad[5]);
    
    // Save result
    save_ppm(fb, "textured_output.ppm");
    
    // Cleanup
    free_texture(checkerboard);
    free_texture(gradient);
    free_framebuffer(fb);
    
    printf("Demo completed successfully!\n");
    printf("Output saved as 'textured_output.ppm'\n");
    
    return 0;
}