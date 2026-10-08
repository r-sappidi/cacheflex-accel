// Four full SVE registers carry one bounded 1024-byte diagnostic chunk.
// Payload loads/stores use the coherent timing path; no functional memory.
static void ncc_input(const void* src,size_t bytes) {
 ncc_sum(0x4e434350,10,bytes,0,0,0);
 for(size_t off=0;off<bytes;off+=1024){
  register uint64_t x0 asm("x0")=0x4e434350,x1 asm("x1")=11,x2 asm("x2")=off;
  register uint64_t x3 asm("x3")=0,x4 asm("x4")=0,x5 asm("x5")=0;
  auto* ptr=static_cast<const char*>(src)+off;
  asm volatile("ptrue p0.b\nld1b {z28.b}, p0/z, [%6]\nld1b {z29.b}, p0/z, [%6, #1, mul vl]\nld1b {z30.b}, p0/z, [%6, #2, mul vl]\nld1b {z31.b}, p0/z, [%6, #3, mul vl]\ndsb sy\nisb\n.inst 0xff230110\nisb\n"
   : "+r"(x0),"+r"(x1),"+r"(x2),"+r"(x3),"+r"(x4),"+r"(x5):"r"(ptr):"p0","z28","z29","z30","z31","memory");
 }
}
static void ncc_output(void* dst,size_t bytes) {
 while(!ncc_sum(0x4e434350,4,0,0,0,0))asm volatile("yield":::"memory");
 for(size_t off=0;off<bytes;off+=1024){
  register uint64_t x0 asm("x0")=0x4e434350,x1 asm("x1")=12,x2 asm("x2")=off;
  register uint64_t x3 asm("x3")=0,x4 asm("x4")=0,x5 asm("x5")=0;
  auto* ptr=static_cast<char*>(dst)+off;
  asm volatile(".inst 0xff230110\nisb\nptrue p0.b\nst1b {z28.b}, p0, [%6]\nst1b {z29.b}, p0, [%6, #1, mul vl]\nst1b {z30.b}, p0, [%6, #2, mul vl]\nst1b {z31.b}, p0, [%6, #3, mul vl]\ndsb sy\nisb\n"
   : "+r"(x0),"+r"(x1),"+r"(x2),"+r"(x3),"+r"(x4),"+r"(x5):"r"(ptr):"p0","z28","z29","z30","z31","memory");
 }
}
