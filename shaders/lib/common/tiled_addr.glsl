#ifndef TILED_ADDR_GLSL
#define TILED_ADDR_GLSL

// ===========================================================================
// 8×8 Tiled Linear Address Encoding / Decoding
// ===========================================================================
// 将 2D 图像像素按 8×8 瓦片（tile）组织，瓦片内部按行优先线性排列，
// 瓦片之间也按行优先排列。相比朴素的线性地址，8×8 瓦片编码显著改善了
// 2D 访问模式的 GPU 缓存局部性（cache locality）。
//
// 多层级（N = 0, 1, 2, …）共用同一 w×h，各层级在缓冲区中连续存放：
//   N=0 占 tiledSize(w,h) 像素，N=1 紧随其后，依此类推。
//
// 对于非 8 倍对齐的图像尺寸（如 13×10），右侧和底部的瓦片自然截断：
//   ┌──────────┬─────┐
//   │ Tile 0   │ T 1 │  ← 右侧残瓦：仅前 5 列有效
//   │ (8×8)    │(5×8)│
//   ├──────────┼─────┤
//   │ Tile 2   │ T 3 │  ← 右下角残瓦：仅 5×2 有效
//   │ (8×2)    │(5×2)│
//   └──────────┴─────┘
// 残瓦内的无效槽位会分配地址但调用方不应访问（x≥w 或 y≥h）。
// ===========================================================================

// ---------------------------------------------------------------------------
// tiledSize8x8 — 单层 8×8 瓦片存储尺寸（像素数）
//
//   = ceil(w/8) × ceil(h/8) × 64
//
//   用途：分配总缓冲区大小 = 层数 × tiledSize8x8(w, h)
// ---------------------------------------------------------------------------
uint tiledSize8x8(uint w, uint h) {
    uint tilesX = (w + 7u) >> 3;
    uint tilesY = (h + 7u) >> 3;
    return (tilesX * tilesY) << 6;
}

// ---------------------------------------------------------------------------
// tiledAddr8x8 — 8×8 瓦片线性地址编码
//
//   参数:
//     N — 图像层级索引 (0, 1, 2, …)。所有层级共用同一 w×h，
//         N=0 从 offset 0 开始，后续层级依次紧接。
//     w — 图像宽度（像素）
//     h — 图像高度（像素）
//     x — 像素列坐标  (0 ≤ x < w)
//     y — 像素行坐标  (0 ≤ y < h)
//
//   返回: 全局线性索引
//
//   性能提示: 若在循环中反复调用且 w/h 不变，
//   可在循环外预计算 stride = tiledSize8x8(w,h)，
//   然后直接用 N*stride + tiledAddr8x8(0, w, h, x, y)。
// ---------------------------------------------------------------------------
uint tiledAddr8x8(uint N, uint w, uint h, uint x, uint y) {
    uint tileX = x >> 3;
    uint tileY = y >> 3;
    uint localX = x & 7u;
    uint localY = y & 7u;

    uint tilesPerRow = (w + 7u) >> 3;

    uint tileIndex = tileY * tilesPerRow + tileX;
    uint localIndex = (localY << 3) + localX;

    uint levelStride = (tilesPerRow * ((h + 7u) >> 3)) << 6;

    return N * levelStride + (tileIndex << 6) + localIndex;
}

// ---------------------------------------------------------------------------
// tiledAddrInverse8x8 — 瓦片地址反解（单层内）
//
//   参数:
//     index — 层内瓦片地址 (0 .. tiledSize8x8(w,h)-1)
//     w     — 图像宽度（像素）
//     h     — 图像高度（像素） [保留以保持接口对称]
//     x, y  — [out] 还原的像素坐标
//
//   注意: 反解对残瓦中的无效槽位也能正确还原坐标，
//         调用方需自行判断 x < w && y < h。
// ---------------------------------------------------------------------------
void tiledAddrInverse8x8(uint index, uint w, uint h, out uint x, out uint y) {
    uint tilesPerRow = (w + 7u) >> 3;

    uint tileIndex = index >> 6;
    uint localIndex = index & 63u;

    uint tileY = tileIndex / tilesPerRow;
    uint tileX = tileIndex - tileY * tilesPerRow;

    uint localY = localIndex >> 3;
    uint localX = localIndex & 7u;

    x = (tileX << 3) + localX;
    y = (tileY << 3) + localY;
}

#endif // TILED_ADDR_GLSL
