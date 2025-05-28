#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

// Window dimensions
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

// Voxel grid dimensions
#define GRID_SIZE 8

// Camera controls
float cameraX = 0.0f, cameraY = 0.0f, cameraZ = 20.0f;
float angleX = 0.0f, angleY = 0.0f;

// Voxel data (1 = solid, 0 = empty)
int voxels[GRID_SIZE][GRID_SIZE][GRID_SIZE];

// Mouse controls
int lastX = 0, lastY = 0;
bool mouseDown = false;

// Initialize voxel grid with some sample data
void initVoxels() {
    for (int x = 0; x < GRID_SIZE; x++) {
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int z = 0; z < GRID_SIZE; z++) {
                // Create a simple sphere-like pattern
                float dx = x - GRID_SIZE/2;
                float dy = y - GRID_SIZE/2;
                float dz = z - GRID_SIZE/2;
                voxels[x][y][z] = (dx*dx + dy*dy + dz*dz < GRID_SIZE*GRID_SIZE/4) ? 1 : 0;
            }
        }
    }
}

// Draw a single cube at position (x,y,z)
void drawCube(float x, float y, float z) {
    glPushMatrix();
    glTranslatef(x, y, z);
    
    float s = 0.5f; // Half-size of cube
    glBegin(GL_QUADS);
    
    // Front face
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-s, -s, s);
    glVertex3f(s, -s, s);
    glVertex3f(s, s, s);
    glVertex3f(-s, s, s);
    
    // Back face
    glVertex3f(-s, -s, -s);
    glVertex3f(s, -s, -s);
    glVertex3f(s, s, -s);
    glVertex3f(-s, s, -s);
    
    // Left face
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-s, -s, -s);
    glVertex3f(-s, -s, s);
    glVertex3f(-s, s, s);
    glVertex3f(-s, s, -s);
    
    // Right face
    glVertex3f(s, -s, -s);
    glVertex3f(s, -s, s);
    glVertex3f(s, s, s);
    glVertex3f(s, s, -s);
    
    // Top face
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-s, s, -s);
    glVertex3f(s, s, -s);
    glVertex3f(s, s, s);
    glVertex3f(-s, s, s);
    
    // Bottom face
    glVertex3f(-s, -s, -s);
    glVertex3f(s, -s, -s);
    glVertex3f(s, -s, s);
    glVertex3f(-s, -s, s);
    
    glEnd();
    glPopMatrix();
}

// Display callback
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    // Set up camera
    glTranslatef(0.0f, 0.0f, -cameraZ);
    glRotatef(angleX, 1.0f, 0.0f, 0.0f);
    glRotatef(angleY, 0.0f, 1.0f, 0.0f);
    
    // Draw voxels
    for (int x = 0; x < GRID_SIZE; x++) {
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int z = 0; z < GRID_SIZE; z++) {
                if (voxels[x][y][z]) {
                    drawCube(x - GRID_SIZE/2, y - GRID_SIZE/2, z - GRID_SIZE/2);
                }
            }
        }
    }
    
    glutSwapBuffers();
}

// Reshape callback
void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)w/h, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

// Keyboard callback
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // ESC
            exit(0);
            break;
        case 'w':
            cameraZ -= 0.5f;
            break;
        case 's':
            cameraZ += 0.5f;
            break;
    }
    glutPostRedisplay();
}

// Mouse motion callback
void mouseMotion(int x, int y) {
    if (mouseDown) {
        angleY += (x - lastX) * 0.5f;
        angleX += (y - lastY) * 0.5f;
        lastX = x;
        lastY = y;
        glutPostRedisplay();
    }
}

// Mouse callback
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseDown = true;
            lastX = x;
            lastY = y;
        } else {
            mouseDown = false;
        }
    }
}

// Initialize OpenGL
void init() {
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    initVoxels();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Voxel Rendering");
    
    init();
    
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    
    glutMainLoop();
    return 0;
}