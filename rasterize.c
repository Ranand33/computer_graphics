#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Simple 2D point structure
typedef struct {
    float x, y;
} Point2D;

// Color structure (RGB)
typedef struct {
    unsigned char r, g, b;
} Color;

// Triangle structure
typedef struct {
    Point2D v0, v1, v2;
    Color color;
} Triangle;

// Framebuffer structure
typedef struct {
    int width, height;
    Color* pixels;
} Framebuffer;

// Create a new framebuffer
Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = (Framebuffer*)malloc(sizeof(Framebuffer));
    fb->width = width;
    fb->height = height;
    fb->pixels = (Color*)calloc(width * height, sizeof(Color));
    return fb;
}

// Free framebuffer memory
void destroy_framebuffer(Framebuffer* fb) {
    free(fb->pixels);
    free(fb);
}

// Clear framebuffer with a color
void clear_framebuffer(Framebuffer* fb, Color color) {
    for (int i = 0; i < fb->width * fb->height; i++) {
        fb->pixels[i] = color;
    }
}

// Set pixel in framebuffer (with bounds checking)
void set_pixel(Framebuffer* fb, int x, int y, Color color) {
    if (x >= 0 && x < fb->width && y >= 0 && y < fb->height) {
        fb->pixels[y * fb->width + x] = color;
    }
}

// Edge function for triangle rasterization
// Returns positive value if point P is on the left side of edge V0V1
float edge_function(Point2D v0, Point2D v1, Point2D p) {
    return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}

// Find minimum of three values
float min3(float a, float b, float c) {
    return fminf(a, fminf(b, c));
}

// Find maximum of three values
float max3(float a, float b, float c) {
    return fmaxf(a, fmaxf(b, c));
}

// Rasterize a triangle using the edge function approach
void rasterize_triangle(Framebuffer* fb, Triangle tri) {
    // Get the bounding box of the triangle
    float minX = min3(tri.v0.x, tri.v1.x, tri.v2.x);
    float minY = min3(tri.v0.y, tri.v1.y, tri.v2.y);
    float maxX = max3(tri.v0.x, tri.v1.x, tri.v2.x);
    float maxY = max3(tri.v0.y, tri.v1.y, tri.v2.y);
    
    // Clip to screen bounds
    minX = fmaxf(0, minX);
    minY = fmaxf(0, minY);
    maxX = fminf(fb->width - 1, maxX);
    maxY = fminf(fb->height - 1, maxY);
    
    // Calculate area of triangle (used for normalization)
    float area = edge_function(tri.v0, tri.v1, tri.v2);
    
    // Skip degenerate triangles
    if (area == 0) return;
    
    // Iterate through all pixels in the bounding box
    for (int y = (int)minY; y <= (int)maxY; y++) {
        for (int x = (int)minX; x <= (int)maxX; x++) {
            Point2D p = {x + 0.5f, y + 0.5f}; // Sample at pixel center
            
            // Calculate edge functions for each edge
            float w0 = edge_function(tri.v1, tri.v2, p);
            float w1 = edge_function(tri.v2, tri.v0, p);
            float w2 = edge_function(tri.v0, tri.v1, p);
            
            // Check if point is inside triangle
            // For CCW triangles, all edge functions should be >= 0
            // For CW triangles, all edge functions should be <= 0
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                set_pixel(fb, x, y, tri.color);
            }
        }
    }
}

// Alternative scanline-based triangle rasterization
void rasterize_triangle_scanline(Framebuffer* fb, Triangle tri) {
    // Sort vertices by Y coordinate
    Point2D v0 = tri.v0, v1 = tri.v1, v2 = tri.v2;
    Point2D temp;
    
    if (v0.y > v1.y) { temp = v0; v0 = v1; v1 = temp; }
    if (v1.y > v2.y) { temp = v1; v1 = v2; v2 = temp; }
    if (v0.y > v1.y) { temp = v0; v0 = v1; v1 = temp; }
    
    // Calculate slopes
    float invslope1 = (v1.x - v0.x) / (v1.y - v0.y);
    float invslope2 = (v2.x - v0.x) / (v2.y - v0.y);
    float invslope3 = (v2.x - v1.x) / (v2.y - v1.y);
    
    // Rasterize the triangle in two parts
    float curx1 = v0.x, curx2 = v0.x;
    
    // Upper part of triangle
    for (int y = (int)v0.y; y <= (int)v1.y; y++) {
        if (y >= 0 && y < fb->height) {
            int startX = (int)fminf(curx1, curx2);
            int endX = (int)fmaxf(curx1, curx2);
            
            for (int x = startX; x <= endX; x++) {
                set_pixel(fb, x, y, tri.color);
            }
        }
        curx1 += invslope1;
        curx2 += invslope2;
    }
    
    // Lower part of triangle
    curx1 = v1.x;
    for (int y = (int)v1.y + 1; y <= (int)v2.y; y++) {
        if (y >= 0 && y < fb->height) {
            int startX = (int)fminf(curx1, curx2);
            int endX = (int)fmaxf(curx1, curx2);
            
            for (int x = startX; x <= endX; x++) {
                set_pixel(fb, x, y, tri.color);
            }
        }
        curx1 += invslope3;
        curx2 += invslope2;
    }
}

// Save framebuffer as PPM image
void save_ppm(const char* filename, Framebuffer* fb) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(file, "P6\n%d %d\n255\n", fb->width, fb->height);
    
    // Write pixel data
    for (int y = 0; y < fb->height; y++) {
        for (int x = 0; x < fb->width; x++) {
            Color pixel = fb->pixels[y * fb->width + x];
            fputc(pixel.r, file);
            fputc(pixel.g, file);
            fputc(pixel.b, file);
        }
    }
    
    fclose(file);
    printf("Saved image to %s\n", filename);
}

// Draw a line (using Bresenham's algorithm)
void draw_line(Framebuffer* fb, int x0, int y0, int x1, int y1, Color color) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    while (1) {
        set_pixel(fb, x0, y0, color);
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

// Draw triangle wireframe
void draw_triangle_wireframe(Framebuffer* fb, Triangle tri, Color color) {
    draw_line(fb, (int)tri.v0.x, (int)tri.v0.y, (int)tri.v1.x, (int)tri.v1.y, color);
    draw_line(fb, (int)tri.v1.x, (int)tri.v1.y, (int)tri.v2.x, (int)tri.v2.y, color);
    draw_line(fb, (int)tri.v2.x, (int)tri.v2.y, (int)tri.v0.x, (int)tri.v0.y, color);
}

int main() {
    // Create a 800x600 framebuffer
    Framebuffer* fb = create_framebuffer(800, 600);
    
    // Clear to black
    Color black = {0, 0, 0};
    clear_framebuffer(fb, black);
    
    // Define some test triangles
    Triangle triangles[] = {
        // Red triangle
        {{100, 100}, {300, 150}, {200, 300}, {255, 0, 0}},
        
        // Green triangle
        {{400, 100}, {600, 100}, {500, 250}, {0, 255, 0}},
        
        // Blue triangle
        {{150, 350}, {350, 400}, {250, 550}, {0, 0, 255}},
        
        // Yellow triangle (overlapping)
        {{450, 300}, {650, 350}, {550, 500}, {255, 255, 0}},
        
        // Cyan triangle
        {{50, 50}, {150, 50}, {100, 150}, {0, 255, 255}},
        
        // Magenta triangle
        {{650, 450}, {750, 450}, {700, 550}, {255, 0, 255}}
    };
    
    int num_triangles = sizeof(triangles) / sizeof(Triangle);
    
    // Rasterize all triangles using edge function method
    printf("Rasterizing %d triangles...\n", num_triangles);
    for (int i = 0; i < num_triangles; i++) {
        rasterize_triangle(fb, triangles[i]);
    }
    
    // Draw wireframe overlay for one triangle
    Color white = {255, 255, 255};
    draw_triangle_wireframe(fb, triangles[0], white);
    
    // Save the result
    save_ppm("triangle_raster.ppm", fb);
    
    // Clean up
    destroy_framebuffer(fb);
    
    // Demonstrate scanline method
    printf("\nCreating scanline rasterization example...\n");
    fb = create_framebuffer(800, 600);
    clear_framebuffer(fb, black);
    
    // Create a large triangle for scanline demo
    Triangle big_tri = {{100, 50}, {700, 200}, {400, 550}, {100, 150, 255}};
    rasterize_triangle_scanline(fb, big_tri);
    
    save_ppm("scanline_raster.ppm", fb);
    destroy_framebuffer(fb);
    
    printf("Done! Check triangle_raster.ppm and scanline_raster.ppm\n");
    
    return 0;
}