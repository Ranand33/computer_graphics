#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#define MAX_UNIFORMS 64
#define MAX_ATTRIBUTES 32
#define MAX_OUTPUTS 16
#define MAX_SAMPLERS 32
#define MAX_UNIFORM_BLOCKS 16
#define MAX_SHADER_STAGES 6
#define MAX_NAME_LENGTH 128
#define MAX_SOURCE_LENGTH 16384

// Shader data types
typedef enum {
    SHADER_TYPE_FLOAT,
    SHADER_TYPE_VEC2,
    SHADER_TYPE_VEC3,
    SHADER_TYPE_VEC4,
    SHADER_TYPE_INT,
    SHADER_TYPE_IVEC2,
    SHADER_TYPE_IVEC3,
    SHADER_TYPE_IVEC4,
    SHADER_TYPE_UINT,
    SHADER_TYPE_UVEC2,
    SHADER_TYPE_UVEC3,
    SHADER_TYPE_UVEC4,
    SHADER_TYPE_BOOL,
    SHADER_TYPE_BVEC2,
    SHADER_TYPE_BVEC3,
    SHADER_TYPE_BVEC4,
    SHADER_TYPE_MAT2,
    SHADER_TYPE_MAT3,
    SHADER_TYPE_MAT4,
    SHADER_TYPE_MAT2X3,
    SHADER_TYPE_MAT2X4,
    SHADER_TYPE_MAT3X2,
    SHADER_TYPE_MAT3X4,
    SHADER_TYPE_MAT4X2,
    SHADER_TYPE_MAT4X3,
    SHADER_TYPE_SAMPLER_1D,
    SHADER_TYPE_SAMPLER_2D,
    SHADER_TYPE_SAMPLER_3D,
    SHADER_TYPE_SAMPLER_CUBE,
    SHADER_TYPE_SAMPLER_2D_ARRAY,
    SHADER_TYPE_SAMPLER_CUBE_ARRAY,
    SHADER_TYPE_SAMPLER_2D_SHADOW,
    SHADER_TYPE_IMAGE_2D,
    SHADER_TYPE_ATOMIC_UINT,
    SHADER_TYPE_UNKNOWN
} ShaderDataType;

// Shader stages
typedef enum {
    SHADER_STAGE_VERTEX,
    SHADER_STAGE_FRAGMENT,
    SHADER_STAGE_GEOMETRY,
    SHADER_STAGE_TESS_CONTROL,
    SHADER_STAGE_TESS_EVALUATION,
    SHADER_STAGE_COMPUTE
} ShaderStage;

// Variable qualifiers
typedef enum {
    QUALIFIER_NONE = 0,
    QUALIFIER_IN = 1,
    QUALIFIER_OUT = 2,
    QUALIFIER_UNIFORM = 4,
    QUALIFIER_BUFFER = 8,
    QUALIFIER_SHARED = 16,
    QUALIFIER_ATTRIBUTE = 32,
    QUALIFIER_VARYING = 64
} VariableQualifier;

// Uniform variable information
typedef struct {
    char name[MAX_NAME_LENGTH];
    ShaderDataType type;
    int location;
    int binding;
    int array_size;
    int offset;        // For uniform blocks
    int array_stride;  // For arrays in uniform blocks
    int matrix_stride; // For matrices in uniform blocks
    bool is_row_major;
    bool is_active;
} UniformInfo;

// Vertex attribute information
typedef struct {
    char name[MAX_NAME_LENGTH];
    ShaderDataType type;
    int location;
    int array_size;
    bool is_active;
} AttributeInfo;

// Shader output information
typedef struct {
    char name[MAX_NAME_LENGTH];
    ShaderDataType type;
    int location;
    int array_size;
    bool is_active;
} OutputInfo;

// Uniform block information
typedef struct {
    char name[MAX_NAME_LENGTH];
    int binding;
    int size;
    int uniform_count;
    UniformInfo uniforms[MAX_UNIFORMS];
    bool is_active;
} UniformBlockInfo;

// Sampler information
typedef struct {
    char name[MAX_NAME_LENGTH];
    ShaderDataType type;
    int binding;
    int location;
    bool is_active;
} SamplerInfo;

// Shader stage reflection data
typedef struct {
    ShaderStage stage;
    char source[MAX_SOURCE_LENGTH];
    
    // Inputs (vertex attributes for vertex shader, varyings for others)
    AttributeInfo inputs[MAX_ATTRIBUTES];
    int input_count;
    
    // Outputs (varyings for vertex shader, fragment outputs for fragment shader)
    OutputInfo outputs[MAX_OUTPUTS];
    int output_count;
    
    // Uniforms
    UniformInfo uniforms[MAX_UNIFORMS];
    int uniform_count;
    
    // Uniform blocks
    UniformBlockInfo uniform_blocks[MAX_UNIFORM_BLOCKS];
    int uniform_block_count;
    
    // Samplers and images
    SamplerInfo samplers[MAX_SAMPLERS];
    int sampler_count;
    
    // Compute shader specific
    int local_size[3]; // Local work group size
    
    bool is_compiled;
} ShaderStageReflection;

// Complete shader program reflection
typedef struct {
    char name[MAX_NAME_LENGTH];
    uint32_t program_id;
    
    ShaderStageReflection stages[MAX_SHADER_STAGES];
    int stage_count;
    
    // Combined program-level information
    UniformInfo active_uniforms[MAX_UNIFORMS];
    int active_uniform_count;
    
    AttributeInfo active_attributes[MAX_ATTRIBUTES];
    int active_attribute_count;
    
    OutputInfo active_outputs[MAX_OUTPUTS];
    int active_output_count;
    
    UniformBlockInfo active_uniform_blocks[MAX_UNIFORM_BLOCKS];
    int active_uniform_block_count;
    
    SamplerInfo active_samplers[MAX_SAMPLERS];
    int active_sampler_count;
    
    bool is_linked;
} ShaderProgramReflection;

// Reflection context for parsing
typedef struct {
    const char* source;
    int position;
    int line;
    int column;
} ReflectionContext;

// Utility functions for string parsing
bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool is_identifier_char(char c) {
    return isalnum(c) || c == '_';
}

void skip_whitespace(ReflectionContext* ctx) {
    while (ctx->source[ctx->position] && is_whitespace(ctx->source[ctx->position])) {
        if (ctx->source[ctx->position] == '\n') {
            ctx->line++;
            ctx->column = 0;
        } else {
            ctx->column++;
        }
        ctx->position++;
    }
}

void skip_line(ReflectionContext* ctx) {
    while (ctx->source[ctx->position] && ctx->source[ctx->position] != '\n') {
        ctx->position++;
        ctx->column++;
    }
}

bool match_keyword(ReflectionContext* ctx, const char* keyword) {
    int len = strlen(keyword);
    if (strncmp(&ctx->source[ctx->position], keyword, len) == 0) {
        // Check that it's not part of a larger identifier
        char next_char = ctx->source[ctx->position + len];
        if (!is_identifier_char(next_char)) {
            ctx->position += len;
            ctx->column += len;
            return true;
        }
    }
    return false;
}

bool read_identifier(ReflectionContext* ctx, char* buffer, int max_len) {
    int i = 0;
    while (i < max_len - 1 && is_identifier_char(ctx->source[ctx->position])) {
        buffer[i++] = ctx->source[ctx->position++];
        ctx->column++;
    }
    buffer[i] = '\0';
    return i > 0;
}

bool read_number(ReflectionContext* ctx, int* value) {
    *value = 0;
    int digits = 0;
    
    while (isdigit(ctx->source[ctx->position])) {
        *value = *value * 10 + (ctx->source[ctx->position] - '0');
        ctx->position++;
        ctx->column++;
        digits++;
    }
    
    return digits > 0;
}

// Type name mapping
ShaderDataType parse_type_name(const char* type_name) {
    if (strcmp(type_name, "float") == 0) return SHADER_TYPE_FLOAT;
    if (strcmp(type_name, "vec2") == 0) return SHADER_TYPE_VEC2;
    if (strcmp(type_name, "vec3") == 0) return SHADER_TYPE_VEC3;
    if (strcmp(type_name, "vec4") == 0) return SHADER_TYPE_VEC4;
    if (strcmp(type_name, "int") == 0) return SHADER_TYPE_INT;
    if (strcmp(type_name, "ivec2") == 0) return SHADER_TYPE_IVEC2;
    if (strcmp(type_name, "ivec3") == 0) return SHADER_TYPE_IVEC3;
    if (strcmp(type_name, "ivec4") == 0) return SHADER_TYPE_IVEC4;
    if (strcmp(type_name, "uint") == 0) return SHADER_TYPE_UINT;
    if (strcmp(type_name, "uvec2") == 0) return SHADER_TYPE_UVEC2;
    if (strcmp(type_name, "uvec3") == 0) return SHADER_TYPE_UVEC3;
    if (strcmp(type_name, "uvec4") == 0) return SHADER_TYPE_UVEC4;
    if (strcmp(type_name, "bool") == 0) return SHADER_TYPE_BOOL;
    if (strcmp(type_name, "bvec2") == 0) return SHADER_TYPE_BVEC2;
    if (strcmp(type_name, "bvec3") == 0) return SHADER_TYPE_BVEC3;
    if (strcmp(type_name, "bvec4") == 0) return SHADER_TYPE_BVEC4;
    if (strcmp(type_name, "mat2") == 0) return SHADER_TYPE_MAT2;
    if (strcmp(type_name, "mat3") == 0) return SHADER_TYPE_MAT3;
    if (strcmp(type_name, "mat4") == 0) return SHADER_TYPE_MAT4;
    if (strcmp(type_name, "mat2x3") == 0) return SHADER_TYPE_MAT2X3;
    if (strcmp(type_name, "mat2x4") == 0) return SHADER_TYPE_MAT2X4;
    if (strcmp(type_name, "mat3x2") == 0) return SHADER_TYPE_MAT3X2;
    if (strcmp(type_name, "mat3x4") == 0) return SHADER_TYPE_MAT3X4;
    if (strcmp(type_name, "mat4x2") == 0) return SHADER_TYPE_MAT4X2;
    if (strcmp(type_name, "mat4x3") == 0) return SHADER_TYPE_MAT4X3;
    if (strcmp(type_name, "sampler1D") == 0) return SHADER_TYPE_SAMPLER_1D;
    if (strcmp(type_name, "sampler2D") == 0) return SHADER_TYPE_SAMPLER_2D;
    if (strcmp(type_name, "sampler3D") == 0) return SHADER_TYPE_SAMPLER_3D;
    if (strcmp(type_name, "samplerCube") == 0) return SHADER_TYPE_SAMPLER_CUBE;
    if (strcmp(type_name, "sampler2DArray") == 0) return SHADER_TYPE_SAMPLER_2D_ARRAY;
    if (strcmp(type_name, "samplerCubeArray") == 0) return SHADER_TYPE_SAMPLER_CUBE_ARRAY;
    if (strcmp(type_name, "sampler2DShadow") == 0) return SHADER_TYPE_SAMPLER_2D_SHADOW;
    if (strcmp(type_name, "image2D") == 0) return SHADER_TYPE_IMAGE_2D;
    if (strcmp(type_name, "atomic_uint") == 0) return SHADER_TYPE_ATOMIC_UINT;
    
    return SHADER_TYPE_UNKNOWN;
}

const char* get_type_name(ShaderDataType type) {
    switch (type) {
        case SHADER_TYPE_FLOAT: return "float";
        case SHADER_TYPE_VEC2: return "vec2";
        case SHADER_TYPE_VEC3: return "vec3";
        case SHADER_TYPE_VEC4: return "vec4";
        case SHADER_TYPE_INT: return "int";
        case SHADER_TYPE_IVEC2: return "ivec2";
        case SHADER_TYPE_IVEC3: return "ivec3";
        case SHADER_TYPE_IVEC4: return "ivec4";
        case SHADER_TYPE_UINT: return "uint";
        case SHADER_TYPE_UVEC2: return "uvec2";
        case SHADER_TYPE_UVEC3: return "uvec3";
        case SHADER_TYPE_UVEC4: return "uvec4";
        case SHADER_TYPE_BOOL: return "bool";
        case SHADER_TYPE_BVEC2: return "bvec2";
        case SHADER_TYPE_BVEC3: return "bvec3";
        case SHADER_TYPE_BVEC4: return "bvec4";
        case SHADER_TYPE_MAT2: return "mat2";
        case SHADER_TYPE_MAT3: return "mat3";
        case SHADER_TYPE_MAT4: return "mat4";
        case SHADER_TYPE_MAT2X3: return "mat2x3";
        case SHADER_TYPE_MAT2X4: return "mat2x4";
        case SHADER_TYPE_MAT3X2: return "mat3x2";
        case SHADER_TYPE_MAT3X4: return "mat3x4";
        case SHADER_TYPE_MAT4X2: return "mat4x2";
        case SHADER_TYPE_MAT4X3: return "mat4x3";
        case SHADER_TYPE_SAMPLER_1D: return "sampler1D";
        case SHADER_TYPE_SAMPLER_2D: return "sampler2D";
        case SHADER_TYPE_SAMPLER_3D: return "sampler3D";
        case SHADER_TYPE_SAMPLER_CUBE: return "samplerCube";
        case SHADER_TYPE_SAMPLER_2D_ARRAY: return "sampler2DArray";
        case SHADER_TYPE_SAMPLER_CUBE_ARRAY: return "samplerCubeArray";
        case SHADER_TYPE_SAMPLER_2D_SHADOW: return "sampler2DShadow";
        case SHADER_TYPE_IMAGE_2D: return "image2D";
        case SHADER_TYPE_ATOMIC_UINT: return "atomic_uint";
        default: return "unknown";
    }
}

int get_type_size(ShaderDataType type) {
    switch (type) {
        case SHADER_TYPE_FLOAT:
        case SHADER_TYPE_INT:
        case SHADER_TYPE_UINT:
        case SHADER_TYPE_BOOL:
            return 4;
        case SHADER_TYPE_VEC2:
        case SHADER_TYPE_IVEC2:
        case SHADER_TYPE_UVEC2:
        case SHADER_TYPE_BVEC2:
            return 8;
        case SHADER_TYPE_VEC3:
        case SHADER_TYPE_IVEC3:
        case SHADER_TYPE_UVEC3:
        case SHADER_TYPE_BVEC3:
            return 12;
        case SHADER_TYPE_VEC4:
        case SHADER_TYPE_IVEC4:
        case SHADER_TYPE_UVEC4:
        case SHADER_TYPE_BVEC4:
            return 16;
        case SHADER_TYPE_MAT2:
            return 16;
        case SHADER_TYPE_MAT3:
            return 36;
        case SHADER_TYPE_MAT4:
            return 64;
        case SHADER_TYPE_MAT2X3:
            return 24;
        case SHADER_TYPE_MAT2X4:
            return 32;
        case SHADER_TYPE_MAT3X2:
            return 24;
        case SHADER_TYPE_MAT3X4:
            return 48;
        case SHADER_TYPE_MAT4X2:
            return 32;
        case SHADER_TYPE_MAT4X3:
            return 48;
        default:
            return 4; // Samplers and other opaque types
    }
}

// Parse layout qualifiers
bool parse_layout_qualifier(ReflectionContext* ctx, int* location, int* binding) {
    *location = -1;
    *binding = -1;
    
    if (!match_keyword(ctx, "layout")) return false;
    
    skip_whitespace(ctx);
    if (ctx->source[ctx->position] != '(') return false;
    ctx->position++;
    ctx->column++;
    
    skip_whitespace(ctx);
    
    while (ctx->source[ctx->position] != ')' && ctx->source[ctx->position] != '\0') {
        char qualifier[64];
        if (!read_identifier(ctx, qualifier, sizeof(qualifier))) break;
        
        skip_whitespace(ctx);
        if (ctx->source[ctx->position] == '=') {
            ctx->position++;
            ctx->column++;
            skip_whitespace(ctx);
            
            int value;
            if (read_number(ctx, &value)) {
                if (strcmp(qualifier, "location") == 0) {
                    *location = value;
                } else if (strcmp(qualifier, "binding") == 0) {
                    *binding = value;
                }
            }
        }
        
        skip_whitespace(ctx);
        if (ctx->source[ctx->position] == ',') {
            ctx->position++;
            ctx->column++;
            skip_whitespace(ctx);
        }
    }
    
    if (ctx->source[ctx->position] == ')') {
        ctx->position++;
        ctx->column++;
        return true;
    }
    
    return false;
}

// Parse variable declaration
bool parse_variable_declaration(ReflectionContext* ctx, VariableQualifier qualifier,
                              UniformInfo* uniform, AttributeInfo* attribute, OutputInfo* output) {
    int location = -1, binding = -1;
    
    // Check for layout qualifier
    parse_layout_qualifier(ctx, &location, &binding);
    skip_whitespace(ctx);
    
    // Skip qualifier keywords
    if (match_keyword(ctx, "in") || match_keyword(ctx, "out") || 
        match_keyword(ctx, "uniform") || match_keyword(ctx, "attribute") || 
        match_keyword(ctx, "varying")) {
        skip_whitespace(ctx);
    }
    
    // Read type name
    char type_name[64];
    if (!read_identifier(ctx, type_name, sizeof(type_name))) return false;
    
    ShaderDataType type = parse_type_name(type_name);
    if (type == SHADER_TYPE_UNKNOWN) return false;
    
    skip_whitespace(ctx);
    
    // Read variable name
    char var_name[MAX_NAME_LENGTH];
    if (!read_identifier(ctx, var_name, sizeof(var_name))) return false;
    
    // Check for array size
    int array_size = 1;
    skip_whitespace(ctx);
    if (ctx->source[ctx->position] == '[') {
        ctx->position++;
        ctx->column++;
        skip_whitespace(ctx);
        
        if (isdigit(ctx->source[ctx->position])) {
            read_number(ctx, &array_size);
        }
        
        // Skip to closing bracket
        while (ctx->source[ctx->position] && ctx->source[ctx->position] != ']') {
            ctx->position++;
            ctx->column++;
        }
        if (ctx->source[ctx->position] == ']') {
            ctx->position++;
            ctx->column++;
        }
    }
    
    // Fill appropriate structure
    if (qualifier & QUALIFIER_UNIFORM && uniform) {
        strncpy(uniform->name, var_name, MAX_NAME_LENGTH - 1);
        uniform->name[MAX_NAME_LENGTH - 1] = '\0';
        uniform->type = type;
        uniform->location = location;
        uniform->binding = binding;
        uniform->array_size = array_size;
        uniform->is_active = true;
        return true;
    } else if ((qualifier & (QUALIFIER_IN | QUALIFIER_ATTRIBUTE)) && attribute) {
        strncpy(attribute->name, var_name, MAX_NAME_LENGTH - 1);
        attribute->name[MAX_NAME_LENGTH - 1] = '\0';
        attribute->type = type;
        attribute->location = location;
        attribute->array_size = array_size;
        attribute->is_active = true;
        return true;
    } else if (qualifier & QUALIFIER_OUT && output) {
        strncpy(output->name, var_name, MAX_NAME_LENGTH - 1);
        output->name[MAX_NAME_LENGTH - 1] = '\0';
        output->type = type;
        output->location = location;
        output->array_size = array_size;
        output->is_active = true;
        return true;
    }
    
    return false;
}

// Parse uniform block
bool parse_uniform_block(ReflectionContext* ctx, UniformBlockInfo* block) {
    if (!match_keyword(ctx, "uniform")) return false;
    skip_whitespace(ctx);
    
    // Read block name
    char block_name[MAX_NAME_LENGTH];
    if (!read_identifier(ctx, block_name, sizeof(block_name))) return false;
    
    strncpy(block->name, block_name, MAX_NAME_LENGTH - 1);
    block->name[MAX_NAME_LENGTH - 1] = '\0';
    
    skip_whitespace(ctx);
    if (ctx->source[ctx->position] != '{') return false;
    ctx->position++;
    ctx->column++;
    
    block->uniform_count = 0;
    block->size = 0;
    
    // Parse block contents
    while (ctx->source[ctx->position] && ctx->source[ctx->position] != '}') {
        skip_whitespace(ctx);
        
        // Skip comments
        if (ctx->source[ctx->position] == '/' && ctx->source[ctx->position + 1] == '/') {
            skip_line(ctx);
            continue;
        }
        
        if (block->uniform_count < MAX_UNIFORMS) {
            UniformInfo* uniform = &block->uniforms[block->uniform_count];
            if (parse_variable_declaration(ctx, QUALIFIER_UNIFORM, uniform, NULL, NULL)) {
                uniform->offset = block->size;
                block->size += get_type_size(uniform->type) * uniform->array_size;
                block->uniform_count++;
            }
        }
        
        // Skip to semicolon or end of line
        while (ctx->source[ctx->position] && ctx->source[ctx->position] != ';' && 
               ctx->source[ctx->position] != '\n' && ctx->source[ctx->position] != '}') {
            ctx->position++;
            ctx->column++;
        }
        if (ctx->source[ctx->position] == ';') {
            ctx->position++;
            ctx->column++;
        }
    }
    
    if (ctx->source[ctx->position] == '}') {
        ctx->position++;
        ctx->column++;
        block->is_active = true;
        return true;
    }
    
    return false;
}

// Parse compute shader local size
void parse_local_size(ReflectionContext* ctx, int local_size[3]) {
    local_size[0] = local_size[1] = local_size[2] = 1;
    
    if (!match_keyword(ctx, "layout")) return;
    
    skip_whitespace(ctx);
    if (ctx->source[ctx->position] != '(') return;
    ctx->position++; ctx->column++;
    
    skip_whitespace(ctx);
    
    while (ctx->source[ctx->position] != ')' && ctx->source[ctx->position] != '\0') {
        char qualifier[64];
        if (!read_identifier(ctx, qualifier, sizeof(qualifier))) break;
        
        skip_whitespace(ctx);
        if (ctx->source[ctx->position] == '=') {
            ctx->position++; ctx->column++;
            skip_whitespace(ctx);
            
            int value;
            if (read_number(ctx, &value)) {
                if (strcmp(qualifier, "local_size_x") == 0) local_size[0] = value;
                else if (strcmp(qualifier, "local_size_y") == 0) local_size[1] = value;
                else if (strcmp(qualifier, "local_size_z") == 0) local_size[2] = value;
            }
        }
        
        skip_whitespace(ctx);
        if (ctx->source[ctx->position] == ',') {
            ctx->position++; ctx->column++;
            skip_whitespace(ctx);
        }
    }
}

// Reflect on shader stage source code
void reflect_shader_stage(ShaderStageReflection* stage, const char* source) {
    ReflectionContext ctx = {source, 0, 1, 0};
    
    // Initialize counters
    stage->input_count = 0;
    stage->output_count = 0;
    stage->uniform_count = 0;
    stage->uniform_block_count = 0;
    stage->sampler_count = 0;
    stage->local_size[0] = stage->local_size[1] = stage->local_size[2] = 1;
    
    // Copy source
    strncpy(stage->source, source, MAX_SOURCE_LENGTH - 1);
    stage->source[MAX_SOURCE_LENGTH - 1] = '\0';
    
    while (ctx.source[ctx.position] != '\0') {
        skip_whitespace(&ctx);
        
        // Skip comments
        if (ctx.source[ctx.position] == '/' && ctx.source[ctx.position + 1] == '/') {
            skip_line(&ctx);
            continue;
        }
        if (ctx.source[ctx.position] == '/' && ctx.source[ctx.position + 1] == '*') {
            ctx.position += 2; ctx.column += 2;
            while (ctx.source[ctx.position] && 
                   !(ctx.source[ctx.position] == '*' && ctx.source[ctx.position + 1] == '/')) {
                if (ctx.source[ctx.position] == '\n') {
                    ctx.line++; ctx.column = 0;
                } else {
                    ctx.column++;
                }
                ctx.position++;
            }
            if (ctx.source[ctx.position]) {
                ctx.position += 2; ctx.column += 2;
            }
            continue;
        }
        
        // Check for layout qualifiers (including compute local size)
        if (strncmp(&ctx.source[ctx.position], "layout", 6) == 0) {
            ReflectionContext temp_ctx = ctx;
            parse_local_size(&temp_ctx, stage->local_size);
        }
        
        // Parse uniform blocks
        if (strncmp(&ctx.source[ctx.position], "uniform", 7) == 0 && 
            stage->uniform_block_count < MAX_UNIFORM_BLOCKS) {
            ReflectionContext temp_ctx = ctx;
            temp_ctx.position += 7; temp_ctx.column += 7;
            skip_whitespace(&temp_ctx);
            
            // Check if this is a uniform block (has a name followed by {)
            char temp_name[64];
            if (read_identifier(&temp_ctx, temp_name, sizeof(temp_name))) {
                skip_whitespace(&temp_ctx);
                if (temp_ctx.source[temp_ctx.position] == '{') {
                    UniformBlockInfo* block = &stage->uniform_blocks[stage->uniform_block_count];
                    if (parse_uniform_block(&ctx, block)) {
                        stage->uniform_block_count++;
                        continue;
                    }
                }
            }
        }
        
        // Parse input variables
        if (strncmp(&ctx.source[ctx.position], "in ", 3) == 0 && 
            stage->input_count < MAX_ATTRIBUTES) {
            AttributeInfo* input = &stage->inputs[stage->input_count];
            if (parse_variable_declaration(&ctx, QUALIFIER_IN, NULL, input, NULL)) {
                stage->input_count++;
                continue;
            }
        }
        
        // Parse output variables
        if (strncmp(&ctx.source[ctx.position], "out ", 4) == 0 && 
            stage->output_count < MAX_OUTPUTS) {
            OutputInfo* output = &stage->outputs[stage->output_count];
            if (parse_variable_declaration(&ctx, QUALIFIER_OUT, NULL, NULL, output)) {
                stage->output_count++;
                continue;
            }
        }
        
        // Parse uniform variables
        if (strncmp(&ctx.source[ctx.position], "uniform ", 8) == 0 && 
            stage->uniform_count < MAX_UNIFORMS) {
            UniformInfo* uniform = &stage->uniforms[stage->uniform_count];
            if (parse_variable_declaration(&ctx, QUALIFIER_UNIFORM, uniform, NULL, NULL)) {
                // Check if it's a sampler
                if (uniform->type >= SHADER_TYPE_SAMPLER_1D && uniform->type <= SHADER_TYPE_ATOMIC_UINT &&
                    stage->sampler_count < MAX_SAMPLERS) {
                    SamplerInfo* sampler = &stage->samplers[stage->sampler_count];
                    strncpy(sampler->name, uniform->name, MAX_NAME_LENGTH - 1);
                    sampler->name[MAX_NAME_LENGTH - 1] = '\0';
                    sampler->type = uniform->type;
                    sampler->binding = uniform->binding;
                    sampler->location = uniform->location;
                    sampler->is_active = true;
                    stage->sampler_count++;
                }
                stage->uniform_count++;
                continue;
            }
        }
        
        // Skip to next potential declaration
        ctx.position++;
        ctx.column++;
    }
    
    stage->is_compiled = true;
}

// Create shader program reflection
ShaderProgramReflection* create_shader_program_reflection(const char* name) {
    ShaderProgramReflection* reflection = calloc(1, sizeof(ShaderProgramReflection));
    strncpy(reflection->name, name, MAX_NAME_LENGTH - 1);
    reflection->name[MAX_NAME_LENGTH - 1] = '\0';
    reflection->program_id = rand() % 1000 + 1; // Mock program ID
    return reflection;
}

// Add shader stage to program
void add_shader_stage(ShaderProgramReflection* program, ShaderStage stage, const char* source) {
    if (program->stage_count >= MAX_SHADER_STAGES) return;
    
    ShaderStageReflection* stage_reflection = &program->stages[program->stage_count];
    stage_reflection->stage = stage;
    
    reflect_shader_stage(stage_reflection, source);
    program->stage_count++;
}

// Link shader program and combine reflection data
void link_shader_program(ShaderProgramReflection* program) {
    program->active_uniform_count = 0;
    program->active_attribute_count = 0;
    program->active_output_count = 0;
    program->active_uniform_block_count = 0;
    program->active_sampler_count = 0;
    
    // Combine uniforms from all stages
    for (int stage = 0; stage < program->stage_count; stage++) {
        ShaderStageReflection* stage_refl = &program->stages[stage];
        
        // Add uniforms
        for (int i = 0; i < stage_refl->uniform_count && 
             program->active_uniform_count < MAX_UNIFORMS; i++) {
            
            // Check if uniform already exists (shared between stages)
            bool exists = false;
            for (int j = 0; j < program->active_uniform_count; j++) {
                if (strcmp(program->active_uniforms[j].name, stage_refl->uniforms[i].name) == 0) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                program->active_uniforms[program->active_uniform_count] = stage_refl->uniforms[i];
                program->active_uniform_count++;
            }
        }
        
        // Add uniform blocks
        for (int i = 0; i < stage_refl->uniform_block_count && 
             program->active_uniform_block_count < MAX_UNIFORM_BLOCKS; i++) {
            
            bool exists = false;
            for (int j = 0; j < program->active_uniform_block_count; j++) {
                if (strcmp(program->active_uniform_blocks[j].name, stage_refl->uniform_blocks[i].name) == 0) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                program->active_uniform_blocks[program->active_uniform_block_count] = stage_refl->uniform_blocks[i];
                program->active_uniform_block_count++;
            }
        }
        
        // Add samplers
        for (int i = 0; i < stage_refl->sampler_count && 
             program->active_sampler_count < MAX_SAMPLERS; i++) {
            
            bool exists = false;
            for (int j = 0; j < program->active_sampler_count; j++) {
                if (strcmp(program->active_samplers[j].name, stage_refl->samplers[i].name) == 0) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                program->active_samplers[program->active_sampler_count] = stage_refl->samplers[i];
                program->active_sampler_count++;
            }
        }
        
        // Add vertex attributes (from vertex shader inputs)
        if (stage_refl->stage == SHADER_STAGE_VERTEX) {
            for (int i = 0; i < stage_refl->input_count && 
                 program->active_attribute_count < MAX_ATTRIBUTES; i++) {
                program->active_attributes[program->active_attribute_count] = stage_refl->inputs[i];
                program->active_attribute_count++;
            }
        }
        
        // Add fragment outputs (from fragment shader outputs)
        if (stage_refl->stage == SHADER_STAGE_FRAGMENT) {
            for (int i = 0; i < stage_refl->output_count && 
                 program->active_output_count < MAX_OUTPUTS; i++) {
                program->active_outputs[program->active_output_count] = stage_refl->outputs[i];
                program->active_output_count++;
            }
        }
    }
    
    program->is_linked = true;
}

// Print reflection information
void print_shader_reflection(const ShaderProgramReflection* program) {
    printf("Shader Program Reflection: %s (ID: %u)\n", program->name, program->program_id);
    printf("========================================\n");
    
    printf("Stages: %d\n", program->stage_count);
    for (int i = 0; i < program->stage_count; i++) {
        const ShaderStageReflection* stage = &program->stages[i];
        const char* stage_names[] = {"Vertex", "Fragment", "Geometry", "Tess Control", "Tess Evaluation", "Compute"};
        printf("  %s Shader:\n", stage_names[stage->stage]);
        
        if (stage->stage == SHADER_STAGE_COMPUTE) {
            printf("    Local Size: %dx%dx%d\n", stage->local_size[0], stage->local_size[1], stage->local_size[2]);
        }
        
        printf("    Inputs: %d, Outputs: %d, Uniforms: %d, Blocks: %d, Samplers: %d\n",
               stage->input_count, stage->output_count, stage->uniform_count, 
               stage->uniform_block_count, stage->sampler_count);
    }
    
    printf("\nActive Vertex Attributes (%d):\n", program->active_attribute_count);
    for (int i = 0; i < program->active_attribute_count; i++) {
        const AttributeInfo* attr = &program->active_attributes[i];
        printf("  [%d] %s %s", attr->location, get_type_name(attr->type), attr->name);
        if (attr->array_size > 1) printf("[%d]", attr->array_size);
        printf("\n");
    }
    
    printf("\nActive Uniforms (%d):\n", program->active_uniform_count);
    for (int i = 0; i < program->active_uniform_count; i++) {
        const UniformInfo* uniform = &program->active_uniforms[i];
        printf("  ");
        if (uniform->location >= 0) printf("[loc=%d] ", uniform->location);
        if (uniform->binding >= 0) printf("[bind=%d] ", uniform->binding);
        printf("%s %s", get_type_name(uniform->type), uniform->name);
        if (uniform->array_size > 1) printf("[%d]", uniform->array_size);
        printf("\n");
    }
    
    printf("\nActive Uniform Blocks (%d):\n", program->active_uniform_block_count);
    for (int i = 0; i < program->active_uniform_block_count; i++) {
        const UniformBlockInfo* block = &program->active_uniform_blocks[i];
        printf("  [bind=%d] %s (size: %d bytes, uniforms: %d)\n", 
               block->binding, block->name, block->size, block->uniform_count);
        
        for (int j = 0; j < block->uniform_count; j++) {
            const UniformInfo* uniform = &block->uniforms[j];
            printf("    [offset=%d] %s %s", uniform->offset, get_type_name(uniform->type), uniform->name);
            if (uniform->array_size > 1) printf("[%d]", uniform->array_size);
            printf("\n");
        }
    }
    
    printf("\nActive Samplers (%d):\n", program->active_sampler_count);
    for (int i = 0; i < program->active_sampler_count; i++) {
        const SamplerInfo* sampler = &program->active_samplers[i];
        printf("  [bind=%d] %s %s\n", sampler->binding, get_type_name(sampler->type), sampler->name);
    }
    
    printf("\nFragment Outputs (%d):\n", program->active_output_count);
    for (int i = 0; i < program->active_output_count; i++) {
        const OutputInfo* output = &program->active_outputs[i];
        printf("  [%d] %s %s", output->location, get_type_name(output->type), output->name);
        if (output->array_size > 1) printf("[%d]", output->array_size);
        printf("\n");
    }
}

// Generate binding configuration code
void generate_binding_code(const ShaderProgramReflection* program) {
    printf("\n// Auto-generated binding code for %s\n", program->name);
    printf("void setup_%s_bindings() {\n", program->name);
    
    // Vertex attributes
    printf("    // Vertex Attributes\n");
    for (int i = 0; i < program->active_attribute_count; i++) {
        const AttributeInfo* attr = &program->active_attributes[i];
        printf("    glBindAttribLocation(program, %d, \"%s\");\n", 
               attr->location >= 0 ? attr->location : i, attr->name);
    }
    
    // Uniform blocks
    printf("\n    // Uniform Blocks\n");
    for (int i = 0; i < program->active_uniform_block_count; i++) {
        const UniformBlockInfo* block = &program->active_uniform_blocks[i];
        if (block->binding >= 0) {
            printf("    glUniformBlockBinding(program, glGetUniformBlockIndex(program, \"%s\"), %d);\n", 
                   block->name, block->binding);
        }
    }
    
    // Samplers
    printf("\n    // Samplers\n");
    for (int i = 0; i < program->active_sampler_count; i++) {
        const SamplerInfo* sampler = &program->active_samplers[i];
        printf("    glUniform1i(glGetUniformLocation(program, \"%s\"), %d);\n", 
               sampler->name, sampler->binding >= 0 ? sampler->binding : i);
    }
    
    printf("}\n");
}

// Performance analysis
void analyze_shader_performance(const ShaderProgramReflection* program) {
    printf("\nPerformance Analysis:\n");
    printf("====================\n");
    
    int total_uniforms = 0;
    int total_uniform_memory = 0;
    int total_samplers = 0;
    
    for (int i = 0; i < program->active_uniform_count; i++) {
        const UniformInfo* uniform = &program->active_uniforms[i];
        total_uniforms++;
        total_uniform_memory += get_type_size(uniform->type) * uniform->array_size;
    }
    
    for (int i = 0; i < program->active_uniform_block_count; i++) {
        const UniformBlockInfo* block = &program->active_uniform_blocks[i];
        total_uniform_memory += block->size;
    }
    
    total_samplers = program->active_sampler_count;
    
    printf("Total Uniforms: %d\n", total_uniforms);
    printf("Uniform Memory: %d bytes\n", total_uniform_memory);
    printf("Texture Units Used: %d\n", total_samplers);
    printf("Vertex Attributes: %d\n", program->active_attribute_count);
    
    // Performance recommendations
    printf("\nRecommendations:\n");
    if (total_uniforms > 32) {
        printf("⚠ Consider using uniform buffers for better performance\n");
    }
    if (total_samplers > 16) {
        printf("⚠ High texture unit usage may impact performance\n");
    }
    if (program->active_attribute_count > 16) {
        printf("⚠ High vertex attribute count may impact vertex throughput\n");
    }
    if (total_uniform_memory > 16384) {
        printf("⚠ Large uniform memory usage (>16KB)\n");
    }
}

// Free reflection data
void free_shader_reflection(ShaderProgramReflection* program) {
    free(program);
}

// Example shader sources for demonstration
const char* example_vertex_shader = R"(
#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;
layout(location = 3) in vec4 a_color;

layout(std140, binding = 0) uniform CameraUniforms {
    mat4 u_view_matrix;
    mat4 u_projection_matrix;
    vec3 u_camera_position;
    float u_time;
};

layout(std140, binding = 1) uniform ModelUniforms {
    mat4 u_model_matrix;
    mat4 u_normal_matrix;
    vec4 u_material_color;
};

uniform float u_animation_speed = 1.0;
uniform vec3 u_light_direction = vec3(0, 1, 0);

out vec3 v_world_position;
out vec3 v_world_normal;
out vec2 v_texcoord;
out vec4 v_color;

void main() {
    vec4 world_pos = u_model_matrix * vec4(a_position, 1.0);
    v_world_position = world_pos.xyz;
    v_world_normal = (u_normal_matrix * vec4(a_normal, 0.0)).xyz;
    v_texcoord = a_texcoord;
    v_color = a_color * u_material_color;
    
    gl_Position = u_projection_matrix * u_view_matrix * world_pos;
}
)";

const char* example_fragment_shader = R"(
#version 450 core

in vec3 v_world_position;
in vec3 v_world_normal;
in vec2 v_texcoord;
in vec4 v_color;

layout(binding = 0) uniform sampler2D u_diffuse_texture;
layout(binding = 1) uniform sampler2D u_normal_texture;
layout(binding = 2) uniform samplerCube u_environment_map;

uniform vec3 u_ambient_color = vec3(0.1);
uniform vec3 u_light_color = vec3(1.0);
uniform float u_metallic = 0.0;
uniform float u_roughness = 0.5;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 BrightColor;

void main() {
    vec4 diffuse = texture(u_diffuse_texture, v_texcoord);
    vec3 normal = normalize(v_world_normal);
    
    vec3 color = diffuse.rgb * v_color.rgb;
    color *= u_ambient_color + u_light_color;
    
    FragColor = vec4(color, diffuse.a);
    
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    BrightColor = brightness > 1.0 ? vec4(color, 1.0) : vec4(0.0);
}
)";

const char* example_compute_shader = R"(
#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec4 position;
    vec4 velocity;
    vec4 color;
    float life;
    float size;
    vec2 padding;
};

layout(std430, binding = 0) restrict buffer ParticleBuffer {
    Particle particles[];
};

layout(binding = 1) uniform sampler2D u_noise_texture;

uniform float u_delta_time;
uniform vec3 u_gravity = vec3(0, -9.8, 0);
uniform float u_damping = 0.98;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= particles.length()) return;
    
    Particle p = particles[index];
    
    if (p.life > 0.0) {
        p.velocity.xyz += u_gravity * u_delta_time;
        p.velocity.xyz *= u_damping;
        p.position.xyz += p.velocity.xyz * u_delta_time;
        p.life -= u_delta_time;
    }
    
    particles[index] = p;
}
)";

int main() {
    printf("Shader Reflection System Demo\n");
    printf("=============================\n\n");
    
    // Create shader program reflection
    ShaderProgramReflection* pbr_program = create_shader_program_reflection("PBR_Material");
    
    // Add shader stages
    add_shader_stage(pbr_program, SHADER_STAGE_VERTEX, example_vertex_shader);
    add_shader_stage(pbr_program, SHADER_STAGE_FRAGMENT, example_fragment_shader);
    
    // Link program to combine reflection data
    link_shader_program(pbr_program);
    
    // Print complete reflection information
    print_shader_reflection(pbr_program);
    
    // Generate binding code
    generate_binding_code(pbr_program);
    
    // Performance analysis
    analyze_shader_performance(pbr_program);
    
    printf("\n");
    
    // Create compute shader example
    ShaderProgramReflection* compute_program = create_shader_program_reflection("ParticleSystem");
    add_shader_stage(compute_program, SHADER_STAGE_COMPUTE, example_compute_shader);
    link_shader_program(compute_program);
    
    printf("\nCompute Shader Example:\n");
    print_shader_reflection(compute_program);
    
    printf("\nReflection System Benefits:\n");
    printf("==========================\n");
    printf("✓ Automatic uniform discovery and binding\n");
    printf("✓ Vertex attribute layout validation\n");
    printf("✓ Uniform buffer size calculation\n");
    printf("✓ Texture unit assignment optimization\n");
    printf("✓ Cross-stage variable validation\n");
    printf("✓ Performance analysis and recommendations\n");
    printf("✓ Auto-generated binding code\n");
    printf("✓ Runtime shader introspection\n");
    
    printf("\nUse Cases:\n");
    printf("==========\n");
    printf("• Game engine material systems\n");
    printf("• Automated render state management\n");
    printf("• Shader hot-reloading systems\n");
    printf("• Graphics debuggers and profilers\n");
    printf("• Asset pipeline validation\n");
    printf("• Runtime shader compilation\n");
    
    // Cleanup
    free_shader_reflection(pbr_program);
    free_shader_reflection(compute_program);
    
    printf("\nShader reflection demo completed successfully!\n");
    return 0;
}