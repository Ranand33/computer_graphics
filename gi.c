#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M_PI 3.14159265358979323846
#define NUM_PATCHES 6
#define NUM_SAMPLES 1000
#define MAX_ITER 100
#define EPS 0.001f

typedef struct {
    float x, y, z;
} Vector3;

typedef struct {
    Vector3 position;  // Center of the patch
    Vector3 normal;    // Normal direction
    float area;        // Area of the patch
    Vector3 emissive;  // Light emitted by the patch
    Vector3 diffuse;   // Reflectivity of the patch
    Vector3 radiosity; // Computed radiosity
} Patch;

// Vector operations
Vector3 add(Vector3 a, Vector3 b) {
    return (Vector3){a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector3 scale(Vector3 a, float s) {
    return (Vector3){a.x * s, a.y * s, a.z * s};
}

Vector3 subtract(Vector3 a, Vector3 b) {
    return (Vector3){a.x - b.x, a.y - b.y, a.z - b.z};
}

float dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Generate a random point on a patch based on its index
Vector3 randomPointOnPatch(int patchIndex) {
    float u = (float)rand() / RAND_MAX;
    float v = (float)rand() / RAND_MAX;
    switch (patchIndex) {
        case 0: return (Vector3){u, v, 0}; // Front (z=0)
        case 1: return (Vector3){u, v, 1}; // Back (z=1)
        case 2: return (Vector3){0, u, v}; // Left (x=0)
        case 3: return (Vector3){1, u, v}; // Right (x=1)
        case 4: return (Vector3){u, 0, v}; // Bottom (y=0)
        case 5: return (Vector3){u, 1, v}; // Top (y=1)
        default: return (Vector3){0, 0, 0};
    }
}

// Compute form factor between two patches using Monte Carlo
float computeFormFactor(int i, int j, Patch* patches) {
    if (i == j) return 0.0f; // A patch doesn't illuminate itself
    float sum = 0.0f;
    for (int k = 0; k < NUM_SAMPLES; k++) {
        Vector3 p = randomPointOnPatch(i);
        Vector3 q = randomPointOnPatch(j);
        Vector3 d = subtract(q, p);
        float r2 = dot(d, d);
        if (r2 < 1e-6) continue; // Avoid division by zero
        float r = sqrtf(r2);
        d = scale(d, 1.0f / r); // Normalize direction
        float cos_i = dot(patches[i].normal, d);
        float cos_j = -dot(patches[j].normal, d);
        if (cos_i > 0 && cos_j > 0) {
            sum += (cos_i * cos_j) / (M_PI * r2);
        }
    }
    return (sum / NUM_SAMPLES) * patches[j].area;
}

int main() {
    // Initialize patches for a unit cube
    Patch patches[NUM_PATCHES] = {
        {{0.5f, 0.5f, 0.0f}, {0, 0, -1}, 1.0f, {0, 0, 0}, {0.8f, 0.8f, 0.8f}, {0, 0, 0}}, // Front
        {{0.5f, 0.5f, 1.0f}, {0, 0, 1}, 1.0f, {0, 0, 0}, {0.8f, 0.8f, 0.8f}, {0, 0, 0}},  // Back
        {{0.0f, 0.5f, 0.5f}, {-1, 0, 0}, 1.0f, {0, 0, 0}, {0.8f, 0.8f, 0.8f}, {0, 0, 0}}, // Left
        {{1.0f, 0.5f, 0.5f}, {1, 0, 0}, 1.0f, {0, 0, 0}, {0.8f, 0.8f, 0.8f}, {0, 0, 0}},  // Right
        {{0.5f, 0.0f, 0.5f}, {0, -1, 0}, 1.0f, {0, 0, 0}, {0.8f, 0.8f, 0.8f}, {0, 0, 0}}, // Bottom
        {{0.5f, 1.0f, 0.5f}, {0, 1, 0}, 1.0f, {1.0f, 1.0f, 1.0f}, {0.8f, 0.8f, 0.8f}, {1.0f, 1.0f, 1.0f}} // Top (light source)
    };

    // Compute form factor matrix
    float F[NUM_PATCHES][NUM_PATCHES];
    for (int i = 0; i < NUM_PATCHES; i++) {
        for (int j = 0; j < NUM_PATCHES; j++) {
            F[i][j] = computeFormFactor(i, j, patches);
        }
    }

    // Iterative radiosity solver
    for (int iter = 0; iter < MAX_ITER; iter++) {
        float maxDiff = 0.0f;
        for (int i = 0; i < NUM_PATCHES; i++) {
            Vector3 sum = {0, 0, 0};
            for (int j = 0; j < NUM_PATCHES; j++) {
                if (i != j) {
                    sum = add(sum, scale(patches[j].radiosity, F[j][i]));
                }
            }
            Vector3 newRad = add(patches[i].emissive, (Vector3){
                patches[i].diffuse.x * sum.x,
                patches[i].diffuse.y * sum.y,
                patches[i].diffuse.z * sum.z
            });
            Vector3 diff = subtract(newRad, patches[i].radiosity);
            float diffMag = sqrtf(dot(diff, diff));
            if (diffMag > maxDiff) maxDiff = diffMag;
            patches[i].radiosity = newRad;
        }
        if (maxDiff < EPS) break; // Convergence check
    }

    // Output radiosity values
    printf("Computed Radiosity Values (RGB):\n");
    for (int i = 0; i < NUM_PATCHES; i++) {
        printf("Patch %d: (%0.3f, %0.3f, %0.3f)\n", i,
               patches[i].radiosity.x, patches[i].radiosity.y, patches[i].radiosity.z);
    }

    return 0;
}