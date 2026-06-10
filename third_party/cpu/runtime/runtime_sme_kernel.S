// SME2 SMOPA micro-kernel. Preprocessed assembly (.S): SYM() bridges the
// Mach-O leading-underscore vs ELF symbol naming so the same source
// assembles on macOS and Linux arm64. NOTE: Apple's arm64 assembler treats
// ';' as a comment character, so directives must stay one-per-line (no
// multi-statement macros).
#ifdef __APPLE__
#  define SYM(name) _##name
#else
#  define SYM(name) name
#endif

.arch armv9-a+sme2
.text
// void sme_tile_16x64(const int8_t* Ap, const int8_t* Bp, int32_t* C, int64_t K4, int64_t ldc_elems)
//   Ap:[K4][16][4]  Bp:[K4][4][16][4]  C: tile top-left, rows strided by ldc_elems int32
//   x0=Ap x1=Bp x2=C x3=K4 x4=ldc_elems
.globl SYM(sme_tile_16x64)
#ifndef __APPLE__
.type sme_tile_16x64, %function
#endif
.p2align 2
SYM(sme_tile_16x64):
    stp     d8,  d9,  [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    smstart
    ptrue   p0.b
    zero    {za}
    lsl     x4, x4, #2          // ldc bytes = ldc_elems*4
    cbz     x3, 2f
    mov     x7, x3
1:  ld1b    {z0.b}, p0/z, [x0]
    ld1b    {z1.b}, p0/z, [x1]
    ld1b    {z2.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z3.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z4.b}, p0/z, [x1, #3, mul vl]
    smopa   za0.s, p0/m, p0/m, z0.b, z1.b
    smopa   za1.s, p0/m, p0/m, z0.b, z2.b
    smopa   za2.s, p0/m, p0/m, z0.b, z3.b
    smopa   za3.s, p0/m, p0/m, z0.b, z4.b
    add     x0, x0, #64
    add     x1, x1, #256
    subs    x7, x7, #1
    b.ne    1b
2:  mov     w12, #0
    mov     x9,  #16
    mov     x10, #32
    mov     x11, #48
3:  st1w    {za0h.s[w12, 0]}, p0, [x2]
    st1w    {za1h.s[w12, 0]}, p0, [x2, x9,  lsl #2]
    st1w    {za2h.s[w12, 0]}, p0, [x2, x10, lsl #2]
    st1w    {za3h.s[w12, 0]}, p0, [x2, x11, lsl #2]
    add     x2, x2, x4
    add     w12, w12, #1
    cmp     w12, #16
    b.ne    3b
    smstop
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8,  d9,  [sp], #64
    ret
#ifndef __APPLE__
.size sme_tile_16x64, . - sme_tile_16x64
.section .note.GNU-stack,"",%progbits
#endif
