#ifndef DATA_GLSL
#define DATA_GLSL

struct Vertex {
    u16vec4 position;       
    u8vec4 color;           
    u16vec2 block_texture;  
    u16vec2 light_texture;  
    u16vec2 mid_tex_coord;  
    i8vec4 tangent;         
    i8vec3 normal;          
    uint8_t padA__;         
    i16vec2 block_id;       
    i8vec3 mid_block;       
    uint8_t padB__;         
}; 

struct Quad {
    Vertex vertices[4];
};


#endif // DATA_GLSL