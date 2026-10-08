// Diagnostic/repaired post-ROI exporter. Refresh BOTH filename and payload
// through timing accesses before every functional pseudo-operation.
.text
.global cf_export_hook
cf_export_hook:
    cbz x1, 9f
    tst x1, #7
    b.ne 9f
    stp x19, x20, [sp, #-224]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x29, x30, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x1
1:
    mov x24, #64
    cmp x20, x24
    csel x24, x20, x24, lo
    add x10, sp, #96
    mov x9, #0
2:
    ldr x11, [x19, x9]
    str x11, [x10, x9]
    add x9, x9, #8
    cmp x9, x24
    b.lo 2b
    add x12, sp, #160
    mov x9, #0
3:
    ldrb w11, [x22, x9]
    strb w11, [x12, x9]
    cbz w11, 6f
    add x9, x9, #1
    cmp x9, #64
    b.lo 3b
    b 4f
6:
    dsb sy
    mov x0, x10
    mov x1, x24
    mov x2, x21
    mov x3, x12
    .inst 0xff4f0110
    cmp x0, x24
    b.ne 4f
    add x19, x19, x24
    add x21, x21, x24
    subs x20, x20, x24
    b.ne 1b
    mov x0, x23
    b 5f
4:
    mov x0, #0
5:
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x29, x30, [sp, #48]
    ldp x19, x20, [sp], #224
    ret
9:
    .inst 0xff4f0110
    ret
