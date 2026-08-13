// ===========================================================================
// buffer_io.glsl — SSBO buffer I/O convenience header (formerly denoise.glsl)
//
// This file has been refactored into separate, functionally-organized modules.
// It remains as a convenience include for existing consumers; new code should
// include only the specific modules it needs.
//
// New module structure:
//   lib/common/pack_half.glsl        — half-float pack/unpack utilities
//   lib/buffers/addr.glsl            — SSBO tiled addressing + buffer declarations
//   lib/buffers/gbuffer.glsl         — GeometryMaterialBuffer read/write
//   lib/buffers/diffuse_buffer.glsl  — DiffuseBuffer read/write + samplePathGuide
//   lib/buffers/specular_buffer.glsl — Reflect/Refract buffer read/write
//   lib/lighting/maxent_encode.glsl   — MaxEntEncoding + encode/decode/project
//   lib/buffers/diffuse_io.glsl      — Diffuse data structs + load/fetch/write
//   lib/buffers/specular_io.glsl     — Specular data structs + fetch/write
// ===========================================================================

#include "/lib/common/pack_half.glsl"
#include "/lib/buffers/addr.glsl"
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/buffers/diffuse_buffer.glsl"
#include "/lib/buffers/specular_buffer.glsl"
#include "/lib/lighting/maxent_encode.glsl"
#include "/lib/buffers/diffuse_io.glsl"
#include "/lib/buffers/specular_io.glsl"
