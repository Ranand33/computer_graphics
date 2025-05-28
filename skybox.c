#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>

// Window dimensions
#define WIDTH 800
#define HEIGHT 600

// Camera settings
float cameraPos[3] = {0.0f, 0.0f, 3.0f};
float cameraFront[3] = {0.0f, 0.0f, -1.0f};
float cameraUp[3] = {0.0f, 1.0f, 0.0f};
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;
float fov = 45.0f;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Shader sources
const char* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "uniform mat4 projection;\n"
    "uniform mat4 view;\n"
    "out vec3 TexCoords;\n"
    "void main()\n"
    "{\n"
    "    TexCoords = aPos;\n"
    "    vec4 pos = projection * view * vec4(aPos, 1.0);\n"
    "    gl_Position = pos.xyww;\n"
    "}\n";

const char* fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec3 TexCoords;\n"
    "uniform samplerCube skybox;\n"
    "void main()\n"
    "{\n"
    "    FragColor = texture(skybox, TexCoords);\n"
    "}\n";

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadCubemap(char* faces[]);
void createShaderProgram(const char* vertexSource, const char* fragmentSource, unsigned int* programId);

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Skybox Renderer", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Capture the mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    // Configure global OpenGL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);  // Important for skybox depth testing

    // Create and compile shaders
    unsigned int skyboxShader;
    createShaderProgram(vertexShaderSource, fragmentShaderSource, &skyboxShader);

    // Skybox vertices (a cube)
    float skyboxVertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };

    // Skybox VAO
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // Load skybox textures
    // You'll need 6 images for your skybox (right, left, top, bottom, front, back)
    char* faces[] = {
        "right.jpg",
        "left.jpg",
        "top.jpg",
        "bottom.jpg",
        "front.jpg",
        "back.jpg"
    };
    unsigned int cubemapTexture = loadCubemap(faces);

    // Set up shader uniforms
    glUseProgram(skyboxShader);
    glUniform1i(glGetUniformLocation(skyboxShader, "skybox"), 0);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Process input
        processInput(window);

        // Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw skybox
        glUseProgram(skyboxShader);
        
        // Create view and projection matrices
        // View matrix
        float view[16];
        memset(view, 0, sizeof(view));
        
        // Remove translation from view matrix (only keep rotation)
        float tempView[3];
        tempView[0] = cameraFront[0];
        tempView[1] = cameraFront[1];
        tempView[2] = cameraFront[2];
        
        // Calculate right vector
        float right[3];
        right[0] = cameraFront[1] * cameraUp[2] - cameraFront[2] * cameraUp[1];
        right[1] = cameraFront[2] * cameraUp[0] - cameraFront[0] * cameraUp[2];
        right[2] = cameraFront[0] * cameraUp[1] - cameraFront[1] * cameraUp[0];
        
        float rightLength = sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
        right[0] /= rightLength;
        right[1] /= rightLength;
        right[2] /= rightLength;
        
        // Recalculate up vector
        float up[3];
        up[0] = right[1] * tempView[2] - right[2] * tempView[1];
        up[1] = right[2] * tempView[0] - right[0] * tempView[2];
        up[2] = right[0] * tempView[1] - right[1] * tempView[0];
        
        view[0] = right[0];
        view[4] = right[1];
        view[8] = right[2];
        
        view[1] = up[0];
        view[5] = up[1];
        view[9] = up[2];
        
        view[2] = -tempView[0];
        view[6] = -tempView[1];
        view[10] = -tempView[2];
        
        view[15] = 1.0f;
        
        // Projection matrix
        float projection[16];
        memset(projection, 0, sizeof(projection));
        
        float aspect = (float)WIDTH / (float)HEIGHT;
        float tanHalfFovy = tan(fov * 0.5f * 3.14159f / 180.0f);
        
        projection[0] = 1.0f / (aspect * tanHalfFovy);
        projection[5] = 1.0f / tanHalfFovy;
        projection[10] = -1.0f;
        projection[14] = -1.0f;
        projection[11] = -1.0f;
        
        // Send matrices to shader
        glUniformMatrix4fv(glGetUniformLocation(skyboxShader, "view"), 1, GL_FALSE, view);
        glUniformMatrix4fv(glGetUniformLocation(skyboxShader, "projection"), 1, GL_FALSE, projection);
        
        // Draw skybox cube
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up resources
    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);
    glDeleteProgram(skyboxShader);

    // Terminate GLFW
    glfwTerminate();
    return 0;
}

// Process all input
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos[0] += cameraSpeed * cameraFront[0];
        cameraPos[1] += cameraSpeed * cameraFront[1];
        cameraPos[2] += cameraSpeed * cameraFront[2];
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPos[0] -= cameraSpeed * cameraFront[0];
        cameraPos[1] -= cameraSpeed * cameraFront[1];
        cameraPos[2] -= cameraSpeed * cameraFront[2];
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        float rightVector[3];
        rightVector[0] = cameraFront[1] * cameraUp[2] - cameraFront[2] * cameraUp[1];
        rightVector[1] = cameraFront[2] * cameraUp[0] - cameraFront[0] * cameraUp[2];
        rightVector[2] = cameraFront[0] * cameraUp[1] - cameraFront[1] * cameraUp[0];
        
        float length = sqrt(rightVector[0] * rightVector[0] + 
                            rightVector[1] * rightVector[1] + 
                            rightVector[2] * rightVector[2]);
        rightVector[0] /= length;
        rightVector[1] /= length;
        rightVector[2] /= length;
        
        cameraPos[0] -= rightVector[0] * cameraSpeed;
        cameraPos[1] -= rightVector[1] * cameraSpeed;
        cameraPos[2] -= rightVector[2] * cameraSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        float rightVector[3];
        rightVector[0] = cameraFront[1] * cameraUp[2] - cameraFront[2] * cameraUp[1];
        rightVector[1] = cameraFront[2] * cameraUp[0] - cameraFront[0] * cameraUp[2];
        rightVector[2] = cameraFront[0] * cameraUp[1] - cameraFront[1] * cameraUp[0];
        
        float length = sqrt(rightVector[0] * rightVector[0] + 
                            rightVector[1] * rightVector[1] + 
                            rightVector[2] * rightVector[2]);
        rightVector[0] /= length;
        rightVector[1] /= length;
        rightVector[2] /= length;
        
        cameraPos[0] += rightVector[0] * cameraSpeed;
        cameraPos[1] += rightVector[1] * cameraSpeed;
        cameraPos[2] += rightVector[2] * cameraSpeed;
    }
}

// When the window is resized
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Mouse movement callback
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = (float)xposIn;
    float ypos = (float)yposIn;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed: y ranges from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain pitch
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    // Update camera front vector
    cameraFront[0] = cos((float)(yaw * 3.14159 / 180.0)) * cos((float)(pitch * 3.14159 / 180.0));
    cameraFront[1] = sin((float)(pitch * 3.14159 / 180.0));
    cameraFront[2] = sin((float)(yaw * 3.14159 / 180.0)) * cos((float)(pitch * 3.14159 / 180.0));
    
    // Normalize
    float length = sqrt(cameraFront[0] * cameraFront[0] + 
                        cameraFront[1] * cameraFront[1] + 
                        cameraFront[2] * cameraFront[2]);
    cameraFront[0] /= length;
    cameraFront[1] /= length;
    cameraFront[2] /= length;
}

// Scroll callback for zoom
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}

// Utility function to load a cubemap texture
unsigned int loadCubemap(char* faces[]) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < 6; i++) {
        unsigned char* data = stbi_load(faces[i], &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                         0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        } else {
            printf("Cubemap texture failed to load at path: %s\n", faces[i]);
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

// Create and compile shaders
void createShaderProgram(const char* vertexSource, const char* fragmentSource, unsigned int* programId) {
    // Vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for compilation errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
    }
    
    // Fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check for compilation errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
    }
    
    // Link shaders
    *programId = glCreateProgram();
    glAttachShader(*programId, vertexShader);
    glAttachShader(*programId, fragmentShader);
    glLinkProgram(*programId);
    
    // Check for linking errors
    glGetProgramiv(*programId, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*programId, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
    }
    
    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}