// Post-ROI adapter: make the filename visible through drained timing stores.
// x0/x1/x2 (data/length/offset) are unchanged. Only caller-saved regs used.
.text
.global screen_filename_hook
screen_filename_hook:
    sub sp, sp, #64
    mov x9, x3
    mov x3, sp
    mov x10, #0
1:
    ldrb w11, [x9, x10]
    strb w11, [x3, x10]
    add x10, x10, #1
    cbz w11, 2f
    cmp x10, #64
    b.lo 1b
    mov x0, #0
    add sp, sp, #64
    ret
2:
    dsb sy
    .inst 0xff4f0110
    add sp, sp, #64
    ret
