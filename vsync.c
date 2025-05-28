#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#define MAX_FRAME_SAMPLES 120
#define TARGET_FPS 60
#define REFRESH_RATE_60HZ (1000.0 / 60.0)  // 16.67ms
#define REFRESH_RATE_120HZ (1000.0 / 120.0) // 8.33ms
#define REFRESH_RATE_144HZ (1000.0 / 144.0) // 6.94ms

// V-Sync modes
typedef enum {
    VSYNC_OFF,           // No synchronization
    VSYNC_ON,            // Standard V-Sync
    VSYNC_ADAPTIVE,      // Adaptive V-Sync (falls back when below refresh rate)
    VSYNC_FAST,          // Fast V-Sync (tears when above refresh rate)
    VSYNC_TRIPLE_BUFFER  // Triple buffering
} VSyncMode;

// Display information
typedef struct {
    int width, height;
    float refresh_rate;
    float frame_time_ms;  // Target frame time in milliseconds
    bool supports_adaptive_sync;
    bool supports_variable_refresh;
    char name[64];
} DisplayInfo;

// Frame timing statistics
typedef struct {
    double frame_times[MAX_FRAME_SAMPLES];
    int sample_count;
    int sample_index;
    double total_time;
    double average_fps;
    double min_frame_time;
    double max_frame_time;
    double variance;
    int tears_detected;
    int frames_dropped;
    int frames_presented;
} FrameStats;

// V-Sync context
typedef struct {
    VSyncMode mode;
    DisplayInfo display;
    FrameStats stats;
    
    // Timing control
    double last_present_time;
    double accumulated_time;
    bool frame_skip_enabled;
    int consecutive_late_frames;
    
    // Triple buffering
    int front_buffer;
    int back_buffer;
    int middle_buffer;
    bool buffer_ready[3];
    
    // Adaptive behavior
    bool adaptive_fallback_active;
    double performance_threshold;
    
    // Simulation parameters
    double simulated_render_time;
    bool simulate_heavy_load;
} VSyncContext;

// Buffer for simulated framebuffer
typedef struct {
    uint32_t* pixels;
    int width, height;
    uint32_t frame_id;
    double timestamp;
} Framebuffer;

// =============================================================================
// PLATFORM-SPECIFIC TIMING FUNCTIONS
// =============================================================================

#ifdef _WIN32
static LARGE_INTEGER frequency;
static bool timer_initialized = false;

void init_high_resolution_timer() {
    if (!timer_initialized) {
        QueryPerformanceFrequency(&frequency);
        timeBeginPeriod(1); // Set timer resolution to 1ms
        timer_initialized = true;
    }
}

double get_time_ms() {
    if (!timer_initialized) init_high_resolution_timer();
    
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / frequency.QuadPart;
}

void precise_sleep_ms(double ms) {
    if (ms <= 0) return;
    
    double start = get_time_ms();
    double target = start + ms;
    
    // Use Sleep for bulk of the time
    if (ms > 2.0) {
        Sleep((DWORD)(ms - 1.0));
    }
    
    // Spin for precise timing
    while (get_time_ms() < target) {
        Sleep(0); // Yield to other threads
    }
}

#else // Unix/Linux
void init_high_resolution_timer() {
    // Nothing needed on Unix systems
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void precise_sleep_ms(double ms) {
    if (ms <= 0) return;
    
    struct timespec req;
    req.tv_sec = (time_t)(ms / 1000.0);
    req.tv_nsec = (long)((ms - req.tv_sec * 1000.0) * 1000000.0);
    
    nanosleep(&req, NULL);
}
#endif

// =============================================================================
// FRAME STATISTICS AND MONITORING
// =============================================================================

void init_frame_stats(FrameStats* stats) {
    memset(stats, 0, sizeof(FrameStats));
    stats->min_frame_time = 1000.0;
    stats->max_frame_time = 0.0;
}

void update_frame_stats(FrameStats* stats, double frame_time) {
    // Add new sample
    if (stats->sample_count < MAX_FRAME_SAMPLES) {
        stats->frame_times[stats->sample_count] = frame_time;
        stats->sample_count++;
    } else {
        // Ring buffer - replace oldest sample
        stats->total_time -= stats->frame_times[stats->sample_index];
        stats->frame_times[stats->sample_index] = frame_time;
        stats->sample_index = (stats->sample_index + 1) % MAX_FRAME_SAMPLES;
    }
    
    stats->total_time += frame_time;
    
    // Update min/max
    if (frame_time < stats->min_frame_time) stats->min_frame_time = frame_time;
    if (frame_time > stats->max_frame_time) stats->max_frame_time = frame_time;
    
    // Calculate average FPS
    int samples = stats->sample_count;
    stats->average_fps = samples * 1000.0 / stats->total_time;
    
    // Calculate variance
    double mean = stats->total_time / samples;
    double variance_sum = 0.0;
    
    int end = (stats->sample_count < MAX_FRAME_SAMPLES) ? stats->sample_count : MAX_FRAME_SAMPLES;
    for (int i = 0; i < end; i++) {
        double diff = stats->frame_times[i] - mean;
        variance_sum += diff * diff;
    }
    stats->variance = variance_sum / samples;
    
    stats->frames_presented++;
}

void detect_screen_tearing(FrameStats* stats, double expected_frame_time, double actual_frame_time) {
    // Simple heuristic: if frame time is significantly different from expected,
    // it might indicate tearing or synchronization issues
    double deviation = fabs(actual_frame_time - expected_frame_time);
    double threshold = expected_frame_time * 0.1; // 10% tolerance
    
    if (deviation > threshold) {
        stats->tears_detected++;
    }
}

// =============================================================================
// DISPLAY AND MONITOR INFORMATION
// =============================================================================

DisplayInfo get_primary_display_info() {
    DisplayInfo display = {0};
    
    // Simulate getting display information
    // In real implementation, would query actual display capabilities
    display.width = 1920;
    display.height = 1080;
    display.refresh_rate = 60.0f;
    display.frame_time_ms = 1000.0f / display.refresh_rate;
    display.supports_adaptive_sync = true;
    display.supports_variable_refresh = true;
    strcpy(display.name, "Primary Display (60Hz)");
    
    return display;
}

DisplayInfo get_high_refresh_display_info() {
    DisplayInfo display = {0};
    
    display.width = 2560;
    display.height = 1440;
    display.refresh_rate = 144.0f;
    display.frame_time_ms = 1000.0f / display.refresh_rate;
    display.supports_adaptive_sync = true;
    display.supports_variable_refresh = true;
    strcpy(display.name, "Gaming Display (144Hz)");
    
    return display;
}

// =============================================================================
// FRAMEBUFFER SIMULATION
// =============================================================================

Framebuffer* create_framebuffer(int width, int height) {
    Framebuffer* fb = malloc(sizeof(Framebuffer));
    fb->pixels = malloc(width * height * sizeof(uint32_t));
    fb->width = width;
    fb->height = height;
    fb->frame_id = 0;
    fb->timestamp = 0.0;
    return fb;
}

void render_test_pattern(Framebuffer* fb, uint32_t frame_id, double time) {
    // Create animated test pattern to visualize tearing
    int stripe_width = fb->width / 10;
    uint32_t base_color = 0xFF000000; // Black with full alpha
    
    for (int y = 0; y < fb->height; y++) {
        for (int x = 0; x < fb->width; x++) {
            // Moving diagonal stripes
            int stripe_pos = (x + (int)(time * 100.0)) / stripe_width;
            int intensity = (stripe_pos + y / 20) % 2 ? 255 : 128;
            
            // Add frame ID as color component for debugging
            uint32_t frame_component = (frame_id % 256) << 16; // Red channel
            uint32_t pattern_component = intensity; // Blue channel
            
            fb->pixels[y * fb->width + x] = base_color | frame_component | pattern_component;
        }
    }
    
    fb->frame_id = frame_id;
    fb->timestamp = time;
}

void free_framebuffer(Framebuffer* fb) {
    if (fb) {
        free(fb->pixels);
        free(fb);
    }
}

// =============================================================================
// V-SYNC IMPLEMENTATION
// =============================================================================

VSyncContext* create_vsync_context(VSyncMode mode, DisplayInfo display) {
    VSyncContext* ctx = calloc(1, sizeof(VSyncContext));
    ctx->mode = mode;
    ctx->display = display;
    ctx->last_present_time = get_time_ms();
    ctx->performance_threshold = display.frame_time_ms * 0.9; // 90% of refresh time
    ctx->simulated_render_time = display.frame_time_ms * 0.7; // 70% of refresh time
    
    init_frame_stats(&ctx->stats);
    
    // Initialize triple buffering
    ctx->front_buffer = 0;
    ctx->back_buffer = 1;
    ctx->middle_buffer = 2;
    ctx->buffer_ready[0] = true;  // Front buffer is being displayed
    ctx->buffer_ready[1] = false; // Back buffer is being rendered
    ctx->buffer_ready[2] = false; // Middle buffer is waiting
    
    return ctx;
}

// Wait for vertical blank (simulate)
void wait_for_vblank(VSyncContext* ctx) {
    double current_time = get_time_ms();
    double time_since_last_vblank = fmod(current_time, ctx->display.frame_time_ms);
    double time_to_next_vblank = ctx->display.frame_time_ms - time_since_last_vblank;
    
    if (time_to_next_vblank > 0.1) { // Only wait if meaningful time remaining
        precise_sleep_ms(time_to_next_vblank);
    }
}

// Present frame with V-Sync
bool present_frame_vsync(VSyncContext* ctx, Framebuffer* framebuffer) {
    double current_time = get_time_ms();
    double frame_time = current_time - ctx->last_present_time;
    
    switch (ctx->mode) {
        case VSYNC_OFF: {
            // No synchronization - present immediately
            ctx->last_present_time = current_time;
            update_frame_stats(&ctx->stats, frame_time);
            detect_screen_tearing(&ctx->stats, ctx->display.frame_time_ms, frame_time);
            break;
        }
        
        case VSYNC_ON: {
            // Standard V-Sync - wait for vertical blank
            wait_for_vblank(ctx);
            
            double present_time = get_time_ms();
            double actual_frame_time = present_time - ctx->last_present_time;
            
            ctx->last_present_time = present_time;
            update_frame_stats(&ctx->stats, actual_frame_time);
            
            // Check for dropped frames
            if (actual_frame_time > ctx->display.frame_time_ms * 1.5) {
                ctx->stats.frames_dropped++;
            }
            break;
        }
        
        case VSYNC_ADAPTIVE: {
            // Adaptive V-Sync - sync when above refresh rate, don't sync when below
            double render_time = ctx->simulated_render_time;
            
            if (render_time < ctx->performance_threshold) {
                // Performance is good - use V-Sync
                wait_for_vblank(ctx);
                ctx->adaptive_fallback_active = false;
            } else {
                // Performance is poor - disable V-Sync to avoid stutter
                ctx->adaptive_fallback_active = true;
                ctx->consecutive_late_frames++;
            }
            
            double present_time = get_time_ms();
            double actual_frame_time = present_time - ctx->last_present_time;
            
            ctx->last_present_time = present_time;
            update_frame_stats(&ctx->stats, actual_frame_time);
            
            if (ctx->adaptive_fallback_active) {
                detect_screen_tearing(&ctx->stats, ctx->display.frame_time_ms, actual_frame_time);
            }
            break;
        }
        
        case VSYNC_FAST: {
            // Fast V-Sync - sync when possible, tear when necessary
            double time_to_vblank = ctx->display.frame_time_ms - 
                                   fmod(current_time, ctx->display.frame_time_ms);
            
            if (time_to_vblank > 2.0) { // Enough time to wait
                wait_for_vblank(ctx);
            } else {
                // Present immediately to avoid stutter
                detect_screen_tearing(&ctx->stats, ctx->display.frame_time_ms, frame_time);
            }
            
            ctx->last_present_time = get_time_ms();
            update_frame_stats(&ctx->stats, get_time_ms() - current_time + frame_time);
            break;
        }
        
        case VSYNC_TRIPLE_BUFFER: {
            // Triple buffering - always have a frame ready
            
            // Swap buffers
            int temp = ctx->front_buffer;
            ctx->front_buffer = ctx->middle_buffer;
            ctx->middle_buffer = ctx->back_buffer;
            ctx->back_buffer = temp;
            
            // Mark new back buffer as not ready
            ctx->buffer_ready[ctx->back_buffer] = false;
            
            // Only wait for vblank if we have a frame ready
            if (ctx->buffer_ready[ctx->middle_buffer]) {
                wait_for_vblank(ctx);
                ctx->buffer_ready[ctx->front_buffer] = true;
            }
            
            double present_time = get_time_ms();
            double actual_frame_time = present_time - ctx->last_present_time;
            
            ctx->last_present_time = present_time;
            update_frame_stats(&ctx->stats, actual_frame_time);
            break;
        }
    }
    
    return true;
}

// =============================================================================
// PERFORMANCE SIMULATION
// =============================================================================

void simulate_variable_load(VSyncContext* ctx, double time) {
    // Simulate varying render load
    double base_time = ctx->display.frame_time_ms * 0.5;
    double variation = sinf(time * 0.001) * ctx->display.frame_time_ms * 0.4;
    
    if (ctx->simulate_heavy_load) {
        // Simulate occasional heavy frames
        if (fmod(time, 1000.0) < 100.0) {
            variation += ctx->display.frame_time_ms * 0.8;
        }
    }
    
    ctx->simulated_render_time = base_time + variation;
    
    // Simulate render time
    precise_sleep_ms(ctx->simulated_render_time);
}

// =============================================================================
// ANALYSIS AND REPORTING
// =============================================================================

void print_vsync_mode_name(VSyncMode mode) {
    switch (mode) {
        case VSYNC_OFF: printf("V-Sync OFF"); break;
        case VSYNC_ON: printf("V-Sync ON"); break;
        case VSYNC_ADAPTIVE: printf("Adaptive V-Sync"); break;
        case VSYNC_FAST: printf("Fast V-Sync"); break;
        case VSYNC_TRIPLE_BUFFER: printf("Triple Buffering"); break;
    }
}

void print_frame_stats(const VSyncContext* ctx) {
    const FrameStats* stats = &ctx->stats;
    
    printf("\n=== Frame Statistics ===\n");
    print_vsync_mode_name(ctx->mode);
    printf(" on %s\n", ctx->display.name);
    
    printf("Frames Presented: %d\n", stats->frames_presented);
    printf("Frames Dropped: %d\n", stats->frames_dropped);
    printf("Screen Tears Detected: %d\n", stats->tears_detected);
    
    printf("Average FPS: %.2f\n", stats->average_fps);
    printf("Target FPS: %.2f\n", ctx->display.refresh_rate);
    printf("FPS Efficiency: %.1f%%\n", 
           (stats->average_fps / ctx->display.refresh_rate) * 100.0);
    
    printf("Frame Time (ms):\n");
    printf("  Average: %.2f\n", stats->total_time / stats->sample_count);
    printf("  Target:  %.2f\n", ctx->display.frame_time_ms);
    printf("  Min:     %.2f\n", stats->min_frame_time);
    printf("  Max:     %.2f\n", stats->max_frame_time);
    printf("  Stddev:  %.2f\n", sqrt(stats->variance));
    
    if (ctx->mode == VSYNC_ADAPTIVE) {
        printf("Adaptive Fallback: %s\n", 
               ctx->adaptive_fallback_active ? "ACTIVE" : "inactive");
        printf("Consecutive Late Frames: %d\n", ctx->consecutive_late_frames);
    }
    
    // Performance analysis
    double frame_time_avg = stats->total_time / stats->sample_count;
    double deviation = fabs(frame_time_avg - ctx->display.frame_time_ms);
    
    printf("\nPerformance Analysis:\n");
    if (stats->tears_detected > 0) {
        printf("⚠ Screen tearing detected (%d instances)\n", stats->tears_detected);
    }
    if (stats->frames_dropped > 0) {
        printf("⚠ Frame drops detected (%d frames)\n", stats->frames_dropped);
    }
    if (deviation > ctx->display.frame_time_ms * 0.1) {
        printf("⚠ Inconsistent frame timing (%.2fms deviation)\n", deviation);
    }
    if (stats->average_fps < ctx->display.refresh_rate * 0.95) {
        printf("⚠ Below target frame rate\n");
    }
    
    if (stats->tears_detected == 0 && stats->frames_dropped == 0 && 
        deviation < ctx->display.frame_time_ms * 0.05) {
        printf("✓ Smooth, tear-free presentation\n");
    }
}

void compare_vsync_modes(DisplayInfo display) {
    printf("\n=== V-Sync Mode Comparison ===\n");
    printf("Testing on %s\n", display.name);
    
    VSyncMode modes[] = {VSYNC_OFF, VSYNC_ON, VSYNC_ADAPTIVE, VSYNC_FAST, VSYNC_TRIPLE_BUFFER};
    const char* mode_names[] = {"OFF", "ON", "ADAPTIVE", "FAST", "TRIPLE"};
    int num_modes = sizeof(modes) / sizeof(modes[0]);
    
    const int test_frames = 180; // 3 seconds at 60fps
    
    for (int i = 0; i < num_modes; i++) {
        printf("\nTesting %s mode...\n", mode_names[i]);
        
        VSyncContext* ctx = create_vsync_context(modes[i], display);
        Framebuffer* fb = create_framebuffer(display.width, display.height);
        
        // Simulate varying load
        ctx->simulate_heavy_load = (i == 2); // Heavy load for adaptive test
        
        double start_time = get_time_ms();
        
        for (int frame = 0; frame < test_frames; frame++) {
            double frame_start = get_time_ms();
            
            // Simulate render work
            simulate_variable_load(ctx, frame_start);
            
            // Render test pattern
            render_test_pattern(fb, frame, frame_start);
            
            // Present with V-Sync
            present_frame_vsync(ctx, fb);
            
            // Periodic progress update
            if (frame % 60 == 0) {
                printf("  Frame %d/%d\n", frame, test_frames);
            }
        }
        
        double total_time = get_time_ms() - start_time;
        printf("Test completed in %.2f seconds\n", total_time / 1000.0);
        
        print_frame_stats(ctx);
        
        free_framebuffer(fb);
        free(ctx);
    }
}

// =============================================================================
// ADAPTIVE SYNC SIMULATION
// =============================================================================

void simulate_adaptive_sync(DisplayInfo display) {
    printf("\n=== Adaptive Sync Simulation ===\n");
    printf("Simulating Variable Refresh Rate (FreeSync/G-Sync)\n");
    
    if (!display.supports_adaptive_sync) {
        printf("Display does not support adaptive sync\n");
        return;
    }
    
    VSyncContext* ctx = create_vsync_context(VSYNC_ADAPTIVE, display);
    Framebuffer* fb = create_framebuffer(display.width, display.height);
    
    // Simulate variable frame rates
    double frame_rates[] = {30.0, 45.0, 60.0, 90.0, 120.0, 144.0};
    int num_rates = sizeof(frame_rates) / sizeof(frame_rates[0]);
    
    for (int r = 0; r < num_rates; r++) {
        double target_fps = frame_rates[r];
        double target_frame_time = 1000.0 / target_fps;
        
        printf("\nTesting %.0f FPS (%.2fms frame time)\n", target_fps, target_frame_time);
        
        // Reset stats
        init_frame_stats(&ctx->stats);
        
        for (int frame = 0; frame < 60; frame++) {
            double frame_start = get_time_ms();
            
            // Simulate render time for target frame rate
            ctx->simulated_render_time = target_frame_time * 0.8;
            simulate_variable_load(ctx, frame_start);
            
            render_test_pattern(fb, frame, frame_start);
            present_frame_vsync(ctx, fb);
        }
        
        printf("Achieved: %.2f FPS (%.2fms avg frame time)\n", 
               ctx->stats.average_fps, 
               ctx->stats.total_time / ctx->stats.sample_count);
        
        if (ctx->adaptive_fallback_active) {
            printf("Status: Adaptive fallback active (tearing allowed)\n");
        } else {
            printf("Status: Synchronized presentation (no tearing)\n");
        }
    }
    
    free_framebuffer(fb);
    free(ctx);
}

// =============================================================================
// INPUT LAG ANALYSIS
// =============================================================================

void analyze_input_lag() {
    printf("\n=== Input Lag Analysis ===\n");
    
    DisplayInfo displays[] = {
        get_primary_display_info(),
        get_high_refresh_display_info()
    };
    
    VSyncMode modes[] = {VSYNC_OFF, VSYNC_ON, VSYNC_TRIPLE_BUFFER};
    const char* mode_names[] = {"V-Sync OFF", "V-Sync ON", "Triple Buffer"};
    
    for (int d = 0; d < 2; d++) {
        printf("\n%s:\n", displays[d].name);
        
        for (int m = 0; m < 3; m++) {
            double input_lag = 0.0;
            
            switch (modes[m]) {
                case VSYNC_OFF:
                    // Minimal lag - present immediately
                    input_lag = displays[d].frame_time_ms * 0.5; // Half frame on average
                    break;
                    
                case VSYNC_ON:
                    // Full frame lag - wait for vblank
                    input_lag = displays[d].frame_time_ms * 1.5; // Up to 1.5 frames
                    break;
                    
                case VSYNC_TRIPLE_BUFFER:
                    // Reduced lag with triple buffering
                    input_lag = displays[d].frame_time_ms * 1.0; // About 1 frame
                    break;
                    
                default:
                    input_lag = displays[d].frame_time_ms;
                    break;
            }
            
            printf("  %s: %.2fms input lag\n", mode_names[m], input_lag);
        }
    }
    
    printf("\nInput Lag Guidelines:\n");
    printf("• Competitive Gaming: < 20ms total system lag\n");
    printf("• Casual Gaming: < 40ms acceptable\n");
    printf("• Professional Esports: < 10ms preferred\n");
    printf("• VR Applications: < 20ms critical for comfort\n");
}

// =============================================================================
// RECOMMENDATIONS ENGINE
// =============================================================================

void generate_recommendations(DisplayInfo display) {
    printf("\n=== V-Sync Recommendations ===\n");
    printf("For %s:\n", display.name);
    
    if (display.refresh_rate >= 120.0f) {
        printf("\nHigh Refresh Rate Display Detected:\n");
        printf("✓ Recommended: V-Sync ON or Adaptive V-Sync\n");
        printf("• High refresh rate reduces input lag impact\n");
        printf("• Smoother motion with reduced tearing visibility\n");
        printf("• Consider G-Sync/FreeSync if available\n");
    } else {
        printf("\nStandard Refresh Rate Display:\n");
        printf("⚠ Consider: Adaptive V-Sync or Triple Buffering\n");
        printf("• Standard V-Sync may cause noticeable input lag\n");
        printf("• Adaptive sync helps maintain performance\n");
    }
    
    if (display.supports_variable_refresh) {
        printf("\nVariable Refresh Rate Available:\n");
        printf("✓ Strongly Recommended: Enable G-Sync/FreeSync/VRR\n");
        printf("• Best of both worlds: no tearing + low latency\n");
        printf("• Works across wide frame rate ranges\n");
        printf("• Reduces need for traditional V-Sync\n");
    }
    
    printf("\nApplication-Specific Recommendations:\n");
    
    printf("\nCompetitive Gaming:\n");
    printf("• Primary: V-Sync OFF (lowest input lag)\n");
    printf("• Alternative: Fast V-Sync (tearing when needed)\n");
    printf("• Consider: High refresh rate monitor (240Hz+)\n");
    
    printf("\nCasual Gaming:\n");
    printf("• Primary: Adaptive V-Sync\n");
    printf("• Alternative: Triple Buffering\n");
    printf("• Focus: Visual quality and smoothness\n");
    
    printf("\nContent Creation/Productivity:\n");
    printf("• Primary: V-Sync ON\n");
    printf("• Focus: Stable, consistent presentation\n");
    printf("• Input lag less critical than visual quality\n");
    
    printf("\nVR Applications:\n");
    printf("• Primary: Always On + Low Latency Mode\n");
    printf("• Critical: Maintain target FPS (90/120Hz)\n");
    printf("• Motion-to-photon latency must be minimized\n");
}

void free_vsync_context(VSyncContext* ctx) {
    free(ctx);
}

int main() {
    printf("V-Sync Implementation and Frame Synchronization\n");
    printf("===============================================\n");
    
    init_high_resolution_timer();
    
    // Get display information
    DisplayInfo primary_display = get_primary_display_info();
    DisplayInfo gaming_display = get_high_refresh_display_info();
    
    printf("Detected Displays:\n");
    printf("• %s (%.0fHz, %.2fms frame time)\n", 
           primary_display.name, primary_display.refresh_rate, primary_display.frame_time_ms);
    printf("• %s (%.0fHz, %.2fms frame time)\n", 
           gaming_display.name, gaming_display.refresh_rate, gaming_display.frame_time_ms);
    
    // Compare V-Sync modes on primary display
    compare_vsync_modes(primary_display);
    
    // Test adaptive sync capabilities
    simulate_adaptive_sync(gaming_display);
    
    // Analyze input lag implications
    analyze_input_lag();
    
    // Generate recommendations
    generate_recommendations(primary_display);
    generate_recommendations(gaming_display);
    
    printf("\n=== V-Sync Technology Summary ===\n");
    printf("V-Sync Modes Explained:\n\n");
    
    printf("V-Sync OFF:\n");
    printf("• Pros: Lowest input lag, highest frame rates\n");
    printf("• Cons: Screen tearing, inconsistent frame pacing\n");
    printf("• Best for: Competitive gaming, high-performance scenarios\n\n");
    
    printf("V-Sync ON:\n");
    printf("• Pros: Eliminates tearing, smooth motion\n");
    printf("• Cons: Input lag, stuttering when fps < refresh rate\n");
    printf("• Best for: Visual quality priority, stable performance\n\n");
    
    printf("Adaptive V-Sync:\n");
    printf("• Pros: Dynamic behavior, reduces stuttering\n");
    printf("• Cons: Some tearing when performance drops\n");
    printf("• Best for: Variable performance applications\n\n");
    
    printf("Fast V-Sync:\n");
    printf("• Pros: Lower latency than standard V-Sync\n");
    printf("• Cons: Occasional tearing at high frame rates\n");
    printf("• Best for: High-performance gaming with quality focus\n\n");
    
    printf("Triple Buffering:\n");
    printf("• Pros: Reduced input lag, maintains frame rate\n");
    printf("• Cons: Higher memory usage, complexity\n");
    printf("• Best for: Consistent performance with quality\n\n");
    
    printf("Modern Adaptive Sync (G-Sync/FreeSync):\n");
    printf("• Pros: Variable refresh rate, eliminates tearing and stuttering\n");
    printf("• Cons: Requires compatible hardware\n");
    printf("• Best for: Gaming across wide performance ranges\n\n");
    
    printf("Key Performance Metrics:\n");
    printf("• Frame Rate: Frames rendered per second\n");
    printf("• Frame Time: Time between frame presentations\n");
    printf("• Input Lag: Time from input to visual response\n");
    printf("• Screen Tearing: Visual artifacts from unsynchronized updates\n");
    printf("• Stuttering: Inconsistent frame timing perception\n\n");
    
    printf("Optimization Guidelines:\n");
    printf("• Match application frame rate to display refresh rate\n");
    printf("• Use frame rate limiting to prevent excessive GPU load\n");
    printf("• Consider adaptive sync for variable performance\n");
    printf("• Profile input lag for responsive applications\n");
    printf("• Test different modes for specific use cases\n\n");
    
    printf("V-Sync implementation demonstration completed!\n");
    printf("This simulation shows the trade-offs between different\n");
    printf("synchronization strategies for optimal user experience.\n");
    
#ifdef _WIN32
    timeEndPeriod(1);
#endif
    
    return 0;
}