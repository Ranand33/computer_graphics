#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define PI 3.14159265359f
#define MAX_LIGHTS 8
#define MAX_MARCH_STEPS 64
#define EPSILON 0.001f

// 3D vector structure
typedef struct {
    float x, y, z;
} Vec3;

// Color structure with float precision
typedef struct {
    float r, g, b, a;
} ColorF;

// RGB color for output
typedef struct {
    uint8_t r, g, b;
} Color;

// Light structure
typedef struct {
    Vec3 position;
    ColorF color;
    float intensity;
    float range;
} Light;

// Camera structure
typedef struct {
    Vec3 position;
    Vec3 direction;
    Vec3 up;
    float fov;
    float near_plane;
    float far_plane;
} Camera;

// Volumetric fog parameters
typedef struct {
    float density;
    float scattering;
    float absorption;
    float phase_g; // Henyey-Greenstein phase function parameter
    ColorF base_color;
    Vec3 wind_direction;
    float wind_strength;
    float noise_scale;
    float noise_strength;
} FogParams;

// Scene structure
typedef struct {
    Light lights[MAX_LIGHTS];
    int light_count;
    Camera camera;
    FogParams fog;
    float time;
} Scene;

// Framebuffer
typedef struct {
    int width, height;
    ColorF* pixels;
} Framebuffer;

// Vector operations
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
    if (len > EPSILON) {
        return vec3_mul(v, 1.0f / len);
    }
    return (Vec3){0, 0, 0};
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Color operations
ColorF color_add(ColorF a, ColorF b) {
    return (ColorF){a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a};
}

ColorF color_mul(ColorF c, float s) {
    return (ColorF){c.r * s, c.g * s, c.b * s, c.a * s};
}

ColorF color_mul_color(ColorF a, ColorF b) {
    return (ColorF){a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a};
}

// Clamp function
float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Hash function for noise
static float hash3d(int x, int y, int z) {
    int n = x + y * 57 + z * 113;
    n = (n << 13) ^ n;
    return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

// Smooth interpolation function
static float smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

// Simple 3D noise function (Perlin-like)
float noise3d(float x, float y, float z) {
    // Simple hash-based noise
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    int iz = (int)floor(z);
    
    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;
    
    // Sample 8 corners of cube
    float c000 = hash3d(ix, iy, iz);
    float c001 = hash3d(ix, iy, iz + 1);
    float c010 = hash3d(ix, iy + 1, iz);
    float c011 = hash3d(ix, iy + 1, iz + 1);
    float c100 = hash3d(ix + 1, iy, iz);
    float c101 = hash3d(ix + 1, iy, iz + 1);
    float c110 = hash3d(ix + 1, iy + 1, iz);
    float c111 = hash3d(ix + 1, iy + 1, iz + 1);
    
    // Smooth interpolation
    fx = smoothstep(fx);
    fy = smoothstep(fy);
    fz = smoothstep(fz);
    
    // Trilinear interpolation
    float c00 = c000 * (1 - fx) + c100 * fx;
    float c01 = c001 * (1 - fx) + c101 * fx;
    float c10 = c010 * (1 - fx) + c110 * fx;
    float c11 = c011 * (1 - fx) + c111 * fx;
    
    float c0 = c00 * (1 - fy) + c10 * fy;
    float c1 = c01 * (1 - fy) + c11 * fy;
    
    return c0 * (1 - fz) + c1 * fz;
}

// Fractional Brownian Motion for complex noise
float fbm(Vec3 p, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise3d(p.x * frequency, p.y * frequency, p.z * frequency);
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    
    return value;
}

// Get fog density at a point
float get_fog_density(Vec3 pos, const FogParams* fog, float time) {
    // Apply wind displacement
    Vec3 wind_offset = vec3_mul(fog->wind_direction, time * fog->wind_strength);
    Vec3 sample_pos = vec3_add(pos, wind_offset);
    
    // Scale position for noise sampling
    Vec3 noise_pos = vec3_mul(sample_pos, fog->noise_scale);
    
    // Sample noise with multiple octaves
    float noise_value = fbm(noise_pos, 4);
    
    // Combine base density with noise
    float density = fog->density * (1.0f + noise_value * fog->noise_strength);
    
    // Height-based density falloff
    float height_factor = expf(-pos.y * 0.1f);
    density *= height_factor;
    
    return fmaxf(0.0f, density);
}

// Henyey-Greenstein phase function
float henyey_greenstein_phase(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (4.0f * PI * powf(denom, 1.5f));
}

// Calculate light scattering at a point
ColorF calculate_scattering(Vec3 pos, Vec3 view_dir, const Scene* scene) {
    ColorF scattered_light = {0, 0, 0, 0};
    
    for (int i = 0; i < scene->light_count; i++) {
        const Light* light = &scene->lights[i];
        
        Vec3 light_dir = vec3_sub(light->position, pos);
        float light_distance = vec3_length(light_dir);
        light_dir = vec3_normalize(light_dir);
        
        // Attenuation
        float attenuation = 1.0f / (1.0f + light_distance * light_distance / (light->range * light->range));
        attenuation = clamp(attenuation, 0.0f, 1.0f);
        
        // Phase function (scattering direction)
        float cos_theta = vec3_dot(vec3_mul(view_dir, -1.0f), light_dir);
        float phase = henyey_greenstein_phase(cos_theta, scene->fog.phase_g);
        
        // Sample density along light ray for shadowing
        float shadow = 1.0f;
        int shadow_steps = 8;
        float shadow_step_size = light_distance / shadow_steps;
        
        for (int j = 1; j <= shadow_steps; j++) {
            Vec3 shadow_pos = vec3_add(pos, vec3_mul(light_dir, j * shadow_step_size));
            float shadow_density = get_fog_density(shadow_pos, &scene->fog, scene->time);
            shadow *= expf(-shadow_density * scene->fog.absorption * shadow_step_size);
            
            if (shadow < 0.01f) break; // Early termination
        }
        
        // Combine light contribution
        ColorF light_contrib = color_mul(light->color, 
            light->intensity * attenuation * phase * shadow * scene->fog.scattering);
        
        scattered_light = color_add(scattered_light, light_contrib);
    }
    
    return scattered_light;
}

// Ray marching through fog volume
ColorF march_fog(Vec3 ray_start, Vec3 ray_dir, float max_distance, const Scene* scene) {
    ColorF accumulated_color = {0, 0, 0, 0};
    float accumulated_transmission = 1.0f;
    
    float step_size = max_distance / MAX_MARCH_STEPS;
    Vec3 current_pos = ray_start;
    
    for (int step = 0; step < MAX_MARCH_STEPS && accumulated_transmission > 0.01f; step++) {
        // Sample fog density
        float density = get_fog_density(current_pos, &scene->fog, scene->time);
        
        if (density > EPSILON) {
            // Calculate scattering
            ColorF scattered = calculate_scattering(current_pos, ray_dir, scene);
            
            // Apply fog base color
            scattered = color_mul_color(scattered, scene->fog.base_color);
            
            // Beer's law for transmission
            float step_transmission = expf(-density * scene->fog.absorption * step_size);
            
            // Integrate scattering
            ColorF step_color = color_mul(scattered, density * step_size * accumulated_transmission);
            accumulated_color = color_add(accumulated_color, step_color);
            
            // Update transmission
            accumulated_transmission *= step_transmission;
        }
        
        // March forward
        current_pos = vec3_add(current_pos, vec3_mul(ray_dir, step_size));
    }
    
    accumulated_color.a = 1.0f - accumulated_transmission;
    return accumulated_color;
}

// Generate camera ray for pixel
Vec3 get_camera_ray(int x, int y, int width, int height, const Camera* camera) {
    // Convert pixel to normalized device coordinates
    float px = (2.0f * x / width - 1.0f) * tanf(camera->fov * 0.5f) * (float)width / height;
    float py = (1.0f - 2.0f * y / height) * tanf(camera->fov * 0.5f);
    
    // Calculate camera coordinate system
    Vec3 forward = vec3_normalize(camera->direction);
    Vec3 right = vec3_normalize(vec3_cross(forward, camera->up));
    Vec3 up = vec3_cross(right, forward);
    
    // Calculate ray direction
    Vec3 ray_dir = vec3_add(vec3_add(forward, vec3_mul(right, px)), vec3_mul(up, py));
    return vec3_normalize(ray_dir);
}

// Create framebuffer
Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = malloc(sizeof(Framebuffer));
    fb->width = width;
    fb->height = height;
    fb->pixels = calloc(width * height, sizeof(ColorF));
    return fb;
}

// Convert float color to 8-bit
Color color_to_8bit(ColorF c) {
    return (Color){
        (uint8_t)clamp(c.r * 255.0f, 0.0f, 255.0f),
        (uint8_t)clamp(c.g * 255.0f, 0.0f, 255.0f),
        (uint8_t)clamp(c.b * 255.0f, 0.0f, 255.0f)
    };
}

// Tone mapping (simple Reinhard)
ColorF tone_map(ColorF color) {
    return (ColorF){
        color.r / (1.0f + color.r),
        color.g / (1.0f + color.g),
        color.b / (1.0f + color.b),
        color.a
    };
}

// Render volumetric fog
void render_volumetric_fog(Framebuffer* fb, const Scene* scene) {
    printf("Rendering volumetric fog (%dx%d)...\n", fb->width, fb->height);
    
    for (int y = 0; y < fb->height; y++) {
        if (y % 50 == 0) {
            printf("Progress: %.1f%%\n", (float)y / fb->height * 100.0f);
        }
        
        for (int x = 0; x < fb->width; x++) {
            // Generate camera ray
            Vec3 ray_dir = get_camera_ray(x, y, fb->width, fb->height, &scene->camera);
            
            // March through fog
            ColorF fog_color = march_fog(scene->camera.position, ray_dir, 
                                       scene->camera.far_plane, scene);
            
            // Add background color (sky)
            ColorF sky_color = {0.3f, 0.6f, 1.0f, 1.0f};
            ColorF final_color = color_add(
                color_mul(sky_color, 1.0f - fog_color.a),
                fog_color
            );
            
            // Apply tone mapping
            final_color = tone_map(final_color);
            
            fb->pixels[y * fb->width + x] = final_color;
        }
    }
}

// Save framebuffer as PPM
void save_ppm(const Framebuffer* fb, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    fprintf(f, "P6\n%d %d\n255\n", fb->width, fb->height);
    
    for (int i = 0; i < fb->width * fb->height; i++) {
        Color c = color_to_8bit(fb->pixels[i]);
        fwrite(&c, sizeof(Color), 1, f);
    }
    
    fclose(f);
    printf("Saved framebuffer to %s\n", filename);
}

// Initialize scene
Scene create_scene() {
    Scene scene = {0};
    
    // Setup camera
    scene.camera.position = (Vec3){0, 2, -10};
    scene.camera.direction = (Vec3){0, 0, 1};
    scene.camera.up = (Vec3){0, 1, 0};
    scene.camera.fov = PI / 3.0f; // 60 degrees
    scene.camera.near_plane = 0.1f;
    scene.camera.far_plane = 100.0f;
    
    // Add lights
    scene.lights[0] = (Light){
        .position = {-5, 8, 5},
        .color = {1.0f, 0.9f, 0.7f, 1.0f},
        .intensity = 20.0f,
        .range = 30.0f
    };
    
    scene.lights[1] = (Light){
        .position = {10, 5, -5},
        .color = {0.7f, 0.8f, 1.0f, 1.0f},
        .intensity = 15.0f,
        .range = 25.0f
    };
    
    scene.light_count = 2;
    
    // Setup fog parameters
    scene.fog = (FogParams){
        .density = 0.1f,
        .scattering = 0.8f,
        .absorption = 0.3f,
        .phase_g = 0.3f, // Forward scattering
        .base_color = {0.8f, 0.9f, 1.0f, 1.0f},
        .wind_direction = {1.0f, 0.0f, 0.5f},
        .wind_strength = 0.5f,
        .noise_scale = 0.1f,
        .noise_strength = 0.6f
    };
    
    scene.time = 0.0f;
    
    return scene;
}

// Free framebuffer
void free_framebuffer(Framebuffer* fb) {
    if (fb) {
        free(fb->pixels);
        free(fb);
    }
}

int main() {
    printf("Volumetric Fog Renderer\n");
    printf("======================\n");
    
    // Create scene and framebuffer
    Scene scene = create_scene();
    Framebuffer* fb = create_framebuffer(800, 600);
    
    // Render multiple frames with time progression
    for (int frame = 0; frame < 3; frame++) {
        printf("\nRendering frame %d...\n", frame + 1);
        scene.time = frame * 2.0f;
        
        render_volumetric_fog(fb, &scene);
        
        // Save frame
        char filename[64];
        snprintf(filename, sizeof(filename), "volumetric_fog_frame_%d.ppm", frame + 1);
        save_ppm(fb, filename);
    }
    
    // Cleanup
    free_framebuffer(fb);
    
    printf("\nVolumetric fog rendering completed!\n");
    printf("Generated 3 frames showing fog animation.\n");
    
    return 0;
}