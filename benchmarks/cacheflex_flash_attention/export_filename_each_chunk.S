// Called instead of the export loop's inline pseudo-op. Preserve x0-x2;
// materialize the filename anew for every chunk. Only post-ROI execution.
.text
.global filename_each_chunk
filename_each_chunk:
    sub sp, sp, #64
    mov x9, #0
    mov x10, sp
1:
    ldrb w11, [x3, x9]
    strb w11, [x10, x9]
    cbz w11, 2f
    add x9, x9, #1
    cmp x9, #64
    b.lo 1b
    mov x0, #0
    b 3f
2:
    dsb sy
    mov x3, sp
    .inst 0xff4f0110
3:
    add sp, sp, #64
    ret
