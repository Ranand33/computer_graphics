#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265359f
#define FASTFLOOR(x) ((int)(x) - ((x) < 0 && (x) != (int)(x)))

// =============================================================================
// PERMUTATION TABLE AND GRADIENTS
// =============================================================================

// Ken Perlin's original permutation table
static const int p_original[256] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

// Extended permutation table (512 elements to avoid modulo)
static int perm[512];
static int perm_mod12[512];

// Gradient vectors for 3D noise
static const float grad3[12][3] = {
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};

// Gradient vectors for 4D noise
static const float grad4[32][4] = {
    {0,1,1,1}, {0,1,1,-1}, {0,1,-1,1}, {0,1,-1,-1},
    {0,-1,1,1}, {0,-1,1,-1}, {0,-1,-1,1}, {0,-1,-1,-1},
    {1,0,1,1}, {1,0,1,-1}, {1,0,-1,1}, {1,0,-1,-1},
    {-1,0,1,1}, {-1,0,1,-1}, {-1,0,-1,1}, {-1,0,-1,-1},
    {1,1,0,1}, {1,1,0,-1}, {1,-1,0,1}, {1,-1,0,-1},
    {-1,1,0,1}, {-1,1,0,-1}, {-1,-1,0,1}, {-1,-1,0,-1},
    {1,1,1,0}, {1,1,-1,0}, {1,-1,1,0}, {1,-1,-1,0},
    {-1,1,1,0}, {-1,1,-1,0}, {-1,-1,1,0}, {-1,-1,-1,0}
};

// Initialize permutation tables
void init_noise(uint32_t seed) {
    srand(seed);
    
    // Create a random permutation
    for (int i = 0; i < 256; i++) {
        perm[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
    
    // Extend to 512 to avoid modulo operations
    for (int i = 0; i < 256; i++) {
        perm[256 + i] = perm[i];
        perm_mod12[i] = perm[i] % 12;
        perm_mod12[256 + i] = perm[i] % 12;
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Fade function for smooth interpolation
float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Linear interpolation
float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

// Dot product for gradient noise
float dot2(const float g[2], float x, float y) {
    return g[0] * x + g[1] * y;
}

float dot3(const float g[3], float x, float y, float z) {
    return g[0] * x + g[1] * y + g[2] * z;
}

float dot4(const float g[4], float x, float y, float z, float w) {
    return g[0] * x + g[1] * y + g[2] * z + g[3] * w;
}

// =============================================================================
// CLASSIC PERLIN NOISE (1985)
// =============================================================================

float perlin_1d(float x) {
    int X = FASTFLOOR(x) & 255;
    x -= FASTFLOOR(x);
    float u = fade(x);
    
    int A = perm[X];
    int B = perm[X + 1];
    
    return lerp(u, 
                (perm[A] & 1) ? -x : x,
                (perm[B] & 1) ? -(x - 1) : (x - 1));
}

float perlin_2d(float x, float y) {
    int X = FASTFLOOR(x) & 255;
    int Y = FASTFLOOR(y) & 255;
    
    x -= FASTFLOOR(x);
    y -= FASTFLOOR(y);
    
    float u = fade(x);
    float v = fade(y);
    
    int A = perm[X] + Y;
    int AA = perm[A];
    int AB = perm[A + 1];
    int B = perm[X + 1] + Y;
    int BA = perm[B];
    int BB = perm[B + 1];
    
    // Use simplified gradients for 2D
    float grad_aa = ((perm[AA] & 1) ? -x : x) + ((perm[AA] & 2) ? -y : y);
    float grad_ba = ((perm[BA] & 1) ? -(x-1) : (x-1)) + ((perm[BA] & 2) ? -y : y);
    float grad_ab = ((perm[AB] & 1) ? -x : x) + ((perm[AB] & 2) ? -(y-1) : (y-1));
    float grad_bb = ((perm[BB] & 1) ? -(x-1) : (x-1)) + ((perm[BB] & 2) ? -(y-1) : (y-1));
    
    return lerp(v, 
                lerp(u, grad_aa, grad_ba),
                lerp(u, grad_ab, grad_bb));
}

float perlin_3d(float x, float y, float z) {
    int X = FASTFLOOR(x) & 255;
    int Y = FASTFLOOR(y) & 255;
    int Z = FASTFLOOR(z) & 255;
    
    x -= FASTFLOOR(x);
    y -= FASTFLOOR(y);
    z -= FASTFLOOR(z);
    
    float u = fade(x);
    float v = fade(y);
    float w = fade(z);
    
    int A = perm[X] + Y;
    int AA = perm[A] + Z;
    int AB = perm[A + 1] + Z;
    int B = perm[X + 1] + Y;
    int BA = perm[B] + Z;
    int BB = perm[B + 1] + Z;
    
    return lerp(w,
                lerp(v,
                     lerp(u, dot3(grad3[perm[AA] % 12], x, y, z),
                             dot3(grad3[perm[BA] % 12], x-1, y, z)),
                     lerp(u, dot3(grad3[perm[AB] % 12], x, y-1, z),
                             dot3(grad3[perm[BB] % 12], x-1, y-1, z))),
                lerp(v,
                     lerp(u, dot3(grad3[perm[AA+1] % 12], x, y, z-1),
                             dot3(grad3[perm[BA+1] % 12], x-1, y, z-1)),
                     lerp(u, dot3(grad3[perm[AB+1] % 12], x, y-1, z-1),
                             dot3(grad3[perm[BB+1] % 12], x-1, y-1, z-1))));
}

// =============================================================================
// SIMPLEX NOISE (Ken Perlin's improved algorithm)
// =============================================================================

// 2D Simplex noise
float simplex_2d(float xin, float yin) {
    const float F2 = 0.5f * (sqrtf(3.0f) - 1.0f);
    const float G2 = (3.0f - sqrtf(3.0f)) / 6.0f;
    
    float n0, n1, n2; // Noise contributions from the three corners
    
    // Skew the input space to determine which simplex cell we're in
    float s = (xin + yin) * F2; // Hairy factor for 2D
    int i = FASTFLOOR(xin + s);
    int j = FASTFLOOR(yin + s);
    float t = (i + j) * G2;
    float X0 = i - t; // Unskew the cell origin back to (x,y) space
    float Y0 = j - t;
    float x0 = xin - X0; // The x,y distances from the cell origin
    float y0 = yin - Y0;
    
    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    int i1, j1; // Offsets for second (middle) corner of simplex in (i,j) coords
    if (x0 > y0) {
        i1 = 1; j1 = 0; // lower triangle, XY order: (0,0)->(1,0)->(1,1)
    } else {
        i1 = 0; j1 = 1; // upper triangle, YX order: (0,0)->(0,1)->(1,1)
    }
    
    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6
    float x1 = x0 - i1 + G2; // Offsets for middle corner in (x,y) unskewed coords
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2; // Offsets for last corner in (x,y) unskewed coords
    float y2 = y0 - 1.0f + 2.0f * G2;
    
    // Work out the hashed gradient indices of the three simplex corners
    int ii = i & 255;
    int jj = j & 255;
    int gi0 = perm_mod12[ii + perm[jj]];
    int gi1 = perm_mod12[ii + i1 + perm[jj + j1]];
    int gi2 = perm_mod12[ii + 1 + perm[jj + 1]];
    
    // Calculate the contribution from the three corners
    float t0 = 0.5f - x0*x0 - y0*y0;
    if (t0 < 0) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * dot2(&grad3[gi0][0], x0, y0);
    }
    
    float t1 = 0.5f - x1*x1 - y1*y1;
    if (t1 < 0) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * dot2(&grad3[gi1][0], x1, y1);
    }
    
    float t2 = 0.5f - x2*x2 - y2*y2;
    if (t2 < 0) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * dot2(&grad3[gi2][0], x2, y2);
    }
    
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    return 70.0f * (n0 + n1 + n2);
}

// 3D Simplex noise
float simplex_3d(float xin, float yin, float zin) {
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;
    
    float n0, n1, n2, n3; // Noise contributions from the four corners
    
    // Skew the input space to determine which simplex cell we're in
    float s = (xin + yin + zin) * F3; // Very nice and simple skew factor for 3D
    int i = FASTFLOOR(xin + s);
    int j = FASTFLOOR(yin + s);
    int k = FASTFLOOR(zin + s);
    float t = (i + j + k) * G3;
    float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = xin - X0; // The x,y,z distances from the cell origin
    float y0 = yin - Y0;
    float z0 = zin - Z0;
    
    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
    
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; // X Y Z order
        } else if (x0 >= z0) {
            i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; // X Z Y order
        } else {
            i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; // Z X Y order
        }
    } else { // x0<y0
        if (y0 < z0) {
            i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; // Z Y X order
        } else if (x0 < z0) {
            i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; // Y Z X order
        } else {
            i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; // Y X Z order
        }
    }
    
    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.
    float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - j2 + 2.0f*G3;
    float z2 = z0 - k2 + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f*G3;
    float z3 = z0 - 1.0f + 3.0f*G3;
    
    // Work out the hashed gradient indices of the four simplex corners
    int ii = i & 255;
    int jj = j & 255;
    int kk = k & 255;
    int gi0 = perm_mod12[ii+perm[jj+perm[kk]]];
    int gi1 = perm_mod12[ii+i1+perm[jj+j1+perm[kk+k1]]];
    int gi2 = perm_mod12[ii+i2+perm[jj+j2+perm[kk+k2]]];
    int gi3 = perm_mod12[ii+1+perm[jj+1+perm[kk+1]]];
    
    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
    if (t0 < 0) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * dot3(grad3[gi0], x0, y0, z0);
    }
    
    float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
    if (t1 < 0) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * dot3(grad3[gi1], x1, y1, z1);
    }
    
    float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
    if (t2 < 0) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * dot3(grad3[gi2], x2, y2, z2);
    }
    
    float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
    if (t3 < 0) {
        n3 = 0.0f;
    } else {
        t3 *= t3;
        n3 = t3 * t3 * dot3(grad3[gi3], x3, y3, z3);
    }
    
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3);
}

// =============================================================================
// FRACTAL NOISE FUNCTIONS
// =============================================================================

// Fractional Brownian Motion (fBm)
float fbm_perlin_2d(float x, float y, int octaves, float lacunarity, float persistence) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float max_value = 0.0f; // Used for normalizing result to [-1,1]
    
    for (int i = 0; i < octaves; i++) {
        value += perlin_2d(x * frequency, y * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / max_value;
}

float fbm_simplex_2d(float x, float y, int octaves, float lacunarity, float persistence) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float max_value = 0.0f;
    
    for (int i = 0; i < octaves; i++) {
        value += simplex_2d(x * frequency, y * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / max_value;
}

// Turbulence (absolute value of noise)
float turbulence_perlin_2d(float x, float y, int octaves, float lacunarity, float persistence) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        value += fabsf(perlin_2d(x * frequency, y * frequency)) * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value;
}

// Ridged noise (inverted turbulence)
float ridged_perlin_2d(float x, float y, int octaves, float lacunarity, float persistence) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        float n = perlin_2d(x * frequency, y * frequency);
        n = 1.0f - fabsf(n);
        n = n * n; // Square for sharper ridges
        value += n * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value;
}

// =============================================================================
// WORLEY NOISE (CELLULAR/VORONOI NOISE)
// =============================================================================

typedef struct {
    float x, y;
} Point2D;

// Simple hash function for Worley noise
uint32_t hash_2d(int32_t x, int32_t y) {
    uint32_t h = ((uint32_t)x * 73856093) ^ ((uint32_t)y * 19349663);
    h = (h ^ (h >> 16)) * 0x85ebca6b;
    h = (h ^ (h >> 13)) * 0xc2b2ae35;
    h = h ^ (h >> 16);
    return h;
}

float worley_2d(float x, float y) {
    int32_t xi = FASTFLOOR(x);
    int32_t yi = FASTFLOOR(y);
    
    float min_dist = 10.0f;
    
    // Check 9 neighboring cells
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int32_t cx = xi + dx;
            int32_t cy = yi + dy;
            
            // Generate random point in cell
            uint32_t h = hash_2d(cx, cy);
            float px = cx + (float)(h & 0xFFFF) / 65536.0f;
            float py = cy + (float)((h >> 16) & 0xFFFF) / 65536.0f;
            
            // Calculate distance
            float dx_f = x - px;
            float dy_f = y - py;
            float dist = dx_f * dx_f + dy_f * dy_f;
            
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }
    
    return sqrtf(min_dist);
}

// =============================================================================
// VALUE NOISE (Simpler alternative to Perlin)
// =============================================================================

float value_noise_2d(float x, float y) {
    int X = FASTFLOOR(x) & 255;
    int Y = FASTFLOOR(y) & 255;
    
    x -= FASTFLOOR(x);
    y -= FASTFLOOR(y);
    
    float u = fade(x);
    float v = fade(y);
    
    int A = perm[X] + Y;
    int B = perm[X + 1] + Y;
    
    // Use hash values directly as noise values
    float n00 = (float)(perm[perm[A] & 255]) / 255.0f;
    float n10 = (float)(perm[perm[B] & 255]) / 255.0f;
    float n01 = (float)(perm[perm[A + 1] & 255]) / 255.0f;
    float n11 = (float)(perm[perm[B + 1] & 255]) / 255.0f;
    
    return lerp(v, 
                lerp(u, n00, n10),
                lerp(u, n01, n11)) * 2.0f - 1.0f; // Scale to [-1, 1]
}

// =============================================================================
// NOISE UTILITIES AND APPLICATIONS
// =============================================================================

// Domain warping
float domain_warp_2d(float x, float y, float warp_strength) {
    float warp_x = fbm_perlin_2d(x, y, 4, 2.0f, 0.5f) * warp_strength;
    float warp_y = fbm_perlin_2d(x + 100.0f, y + 100.0f, 4, 2.0f, 0.5f) * warp_strength;
    
    return fbm_perlin_2d(x + warp_x, y + warp_y, 6, 2.0f, 0.5f);
}

// Billowy noise (squared fBm for cloud-like effects)
float billowy_2d(float x, float y, int octaves) {
    float noise = fbm_perlin_2d(x, y, octaves, 2.0f, 0.5f);
    return noise * noise * (noise < 0 ? -1.0f : 1.0f);
}

// Marble texture using sine waves and noise
float marble_2d(float x, float y) {
    float noise = fbm_perlin_2d(x * 0.1f, y * 0.1f, 4, 2.0f, 0.5f);
    return sinf((x + noise * 10.0f) * 0.1f);
}

// Wood texture using radial distance and noise
float wood_2d(float x, float y) {
    float distance = sqrtf(x * x + y * y);
    float noise = fbm_perlin_2d(x * 0.1f, y * 0.1f, 3, 2.0f, 0.6f);
    return sinf((distance + noise * 2.0f) * 0.5f);
}

// Heightmap generation
void generate_heightmap(float* heightmap, int width, int height, float scale) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float fx = (float)x / width * scale;
            float fy = (float)y / height * scale;
            
            // Combine multiple noise types for interesting terrain
            float base = fbm_perlin_2d(fx, fy, 6, 2.0f, 0.5f);
            float ridges = ridged_perlin_2d(fx * 2.0f, fy * 2.0f, 4, 2.0f, 0.5f) * 0.3f;
            float detail = turbulence_perlin_2d(fx * 8.0f, fy * 8.0f, 3, 2.0f, 0.3f) * 0.1f;
            
            heightmap[y * width + x] = (base + ridges + detail) * 0.5f + 0.5f; // Normalize to [0,1]
        }
    }
}

// =============================================================================
// DEMONSTRATION AND TESTING
// =============================================================================

// Save noise as PGM image
void save_noise_pgm(const char* filename, float* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        uint8_t pixel = (uint8_t)((data[i] * 0.5f + 0.5f) * 255.0f);
        fwrite(&pixel, 1, 1, f);
    }
    
    fclose(f);
    printf("Saved %s\n", filename);
}

// Generate noise sample
void generate_noise_sample(const char* name, float (*noise_func)(float, float), 
                          int width, int height, float scale) {
    float* data = malloc(width * height * sizeof(float));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float fx = (float)x / width * scale;
            float fy = (float)y / height * scale;
            data[y * width + x] = noise_func(fx, fy);
        }
    }
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s.pgm", name);
    save_noise_pgm(filename, data, width, height);
    
    free(data);
}

// Performance benchmark
void benchmark_noise_functions() {
    printf("\nPerformance Benchmark (1M samples):\n");
    printf("===================================\n");
    
    const int samples = 1000000;
    float total = 0.0f;
    
    // Perlin 2D
    clock_t start = clock();
    for (int i = 0; i < samples; i++) {
        float x = (float)(i % 1000) * 0.01f;
        float y = (float)(i / 1000) * 0.01f;
        total += perlin_2d(x, y);
    }
    clock_t end = clock();
    double perlin_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Simplex 2D
    start = clock();
    for (int i = 0; i < samples; i++) {
        float x = (float)(i % 1000) * 0.01f;
        float y = (float)(i / 1000) * 0.01f;
        total += simplex_2d(x, y);
    }
    end = clock();
    double simplex_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Value noise
    start = clock();
    for (int i = 0; i < samples; i++) {
        float x = (float)(i % 1000) * 0.01f;
        float y = (float)(i / 1000) * 0.01f;
        total += value_noise_2d(x, y);
    }
    end = clock();
    double value_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Worley noise
    start = clock();
    for (int i = 0; i < samples / 10; i++) { // Fewer samples due to higher cost
        float x = (float)(i % 100) * 0.1f;
        float y = (float)(i / 100) * 0.1f;
        total += worley_2d(x, y);
    }
    end = clock();
    double worley_time = ((double)(end - start)) / CLOCKS_PER_SEC * 10; // Scale up
    
    printf("Perlin 2D:   %.3f seconds (%.1fM samples/sec)\n", perlin_time, samples / perlin_time / 1000000);
    printf("Simplex 2D:  %.3f seconds (%.1fM samples/sec)\n", simplex_time, samples / simplex_time / 1000000);
    printf("Value 2D:    %.3f seconds (%.1fM samples/sec)\n", value_time, samples / value_time / 1000000);
    printf("Worley 2D:   %.3f seconds (%.1fM samples/sec)\n", worley_time, samples / worley_time / 1000000);
    
    // Prevent optimization of unused total
    if (total > 1e30f) printf("Unlikely result: %f\n", total);
}

// Wrapper functions for sample generation
float sample_perlin_2d(float x, float y) { return perlin_2d(x, y); }
float sample_simplex_2d(float x, float y) { return simplex_2d(x, y); }
float sample_value_2d(float x, float y) { return value_noise_2d(x, y); }
float sample_worley_2d(float x, float y) { return worley_2d(x, y); }
float sample_fbm(float x, float y) { return fbm_perlin_2d(x, y, 6, 2.0f, 0.5f); }
float sample_turbulence(float x, float y) { return turbulence_perlin_2d(x, y, 6, 2.0f, 0.5f); }
float sample_ridged(float x, float y) { return ridged_perlin_2d(x, y, 6, 2.0f, 0.5f); }
float sample_domain_warp(float x, float y) { return domain_warp_2d(x, y, 0.5f); }
float sample_marble(float x, float y) { return marble_2d(x, y); }
float sample_wood(float x, float y) { return wood_2d(x, y); }

int main() {
    printf("Comprehensive Noise Functions Library\n");
    printf("====================================\n\n");
    
    // Initialize with seed
    init_noise(12345);
    
    printf("Generating noise samples (512x512)...\n");
    const int size = 512;
    const float scale = 8.0f;
    
    // Generate various noise types
    generate_noise_sample("perlin_2d", sample_perlin_2d, size, size, scale);
    generate_noise_sample("simplex_2d", sample_simplex_2d, size, size, scale);
    generate_noise_sample("value_2d", sample_value_2d, size, size, scale);
    generate_noise_sample("worley_2d", sample_worley_2d, size, size, scale * 0.5f);
    
    // Generate fractal variations
    generate_noise_sample("fbm_perlin", sample_fbm, size, size, scale);
    generate_noise_sample("turbulence", sample_turbulence, size, size, scale);
    generate_noise_sample("ridged", sample_ridged, size, size, scale);
    generate_noise_sample("domain_warp", sample_domain_warp, size, size, scale);
    
    // Generate texture patterns
    generate_noise_sample("marble", sample_marble, size, size, scale * 2.0f);
    generate_noise_sample("wood", sample_wood, size, size, scale);
    
    // Generate heightmap
    printf("Generating terrain heightmap...\n");
    float* heightmap = malloc(size * size * sizeof(float));
    generate_heightmap(heightmap, size, size, scale);
    save_noise_pgm("heightmap.pgm", heightmap, size, size);
    free(heightmap);
    
    // Test 3D noise
    printf("\nTesting 3D noise functions...\n");
    printf("Perlin 3D at (1,2,3): %.6f\n", perlin_3d(1.0f, 2.0f, 3.0f));
    printf("Simplex 3D at (1,2,3): %.6f\n", simplex_3d(1.0f, 2.0f, 3.0f));
    
    // Performance benchmark
    benchmark_noise_functions();
    
    printf("\nNoise Function Characteristics:\n");
    printf("==============================\n");
    printf("Perlin Noise:\n");
    printf("  • Smooth, natural-looking gradients\n");
    printf("  • Good for terrain and organic textures\n");
    printf("  • Relatively expensive computation\n");
    printf("  • Can show directional artifacts\n\n");
    
    printf("Simplex Noise:\n");
    printf("  • Lower computational complexity than Perlin\n");
    printf("  • Better visual isotropy (no directional bias)\n");
    printf("  • Excellent for high-dimensional noise\n");
    printf("  • Ken Perlin's improved algorithm\n\n");
    
    printf("Value Noise:\n");
    printf("  • Fastest to compute\n");
    printf("  • More blocky appearance\n");
    printf("  • Good for rough textures\n");
    printf("  • Less smooth than gradient-based noise\n\n");
    
    printf("Worley Noise:\n");
    printf("  • Cellular/organic patterns\n");
    printf("  • Excellent for stone, water, cell textures\n");
    printf("  • More computationally expensive\n");
    printf("  • Based on Voronoi diagrams\n\n");
    
    printf("Fractal Techniques:\n");
    printf("  • fBm: Layered detail, natural terrain\n");
    printf("  • Turbulence: Chaotic, fire-like patterns\n");
    printf("  • Ridged: Sharp mountain ridges\n");
    printf("  • Domain Warping: Swirled, organic distortion\n\n");
    
    printf("Applications:\n");
    printf("=============\n");
    printf("• Procedural terrain generation\n");
    printf("• Texture synthesis (marble, wood, clouds)\n");
    printf("• Particle system effects\n");
    printf("• Animation and morphing\n");
    printf("• Atmospheric effects\n");
    printf("• Level generation in games\n");
    printf("• Material property variation\n");
    printf("• Displacement mapping\n\n");
    
    printf("Generated files:\n");
    printf("================\n");
    printf("• perlin_2d.pgm - Classic Perlin noise\n");
    printf("• simplex_2d.pgm - Simplex noise\n");
    printf("• value_2d.pgm - Value noise\n");
    printf("• worley_2d.pgm - Worley/cellular noise\n");
    printf("• fbm_perlin.pgm - Fractional Brownian Motion\n");
    printf("• turbulence.pgm - Turbulence function\n");
    printf("• ridged.pgm - Ridged noise\n");
    printf("• domain_warp.pgm - Domain-warped noise\n");
    printf("• marble.pgm - Marble texture pattern\n");
    printf("• wood.pgm - Wood texture pattern\n");
    printf("• heightmap.pgm - Terrain heightmap\n\n");
    
    printf("Noise library demonstration completed!\n");
    return 0;
}