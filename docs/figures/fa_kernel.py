import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42})
INK='#1f2328'; MUTE='#57606a'; LINE='#8c959f'
COMP='#eef1f4'; MEM='#ddeaf7'; SPM='#d3efe9'; SPMK='#0f7b6c'; CACHE='#fbe7d3'; CACHEK='#b45309'
fig=plt.figure(figsize=(7.2,6.15)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,72); ax.set_ylim(0,61.5); ax.axis('off')
def box(x,y,w,h,fc,ec=LINE,lw=0.7,r=0.6,ls='-',z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,ls=ls,zorder=z))
def step(n,x,y,w,h,title,body,fc=COMP,ec=LINE,tc=INK):
    box(x,y,w,h,fc,ec)
    ax.add_patch(plt.Circle((x+1.35,y+h-1.3),0.95,fc=INK,ec='none',zorder=4))
    ax.text(x+1.35,y+h-1.32,str(n),color='white',ha='center',va='center',fontsize=6.6,fontweight='bold',zorder=5)
    ax.text(x+2.8,y+h-1.3,title,ha='left',va='center',fontweight='bold',color=tc,fontsize=7.4,zorder=5)
    ax.text(x+2.8,y+h-2.75,body,ha='left',va='top',color=MUTE,fontsize=6.5,linespacing=1.35,zorder=5)
def arrow(x1,y1,x2,y2,c=LINE,lw=0.8):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18',color=c,lw=lw,shrinkA=0,shrinkB=0),zorder=3)
def panel(x,y,s,t): ax.text(x,y,s,fontweight='bold',fontsize=8.6,color=INK,va='top'); ax.text(x+2.0,y,t,fontsize=8.2,color=INK,va='top')

# ---------------- (a) execution flow ----------------
panel(1,60.6,'(a)','Per-core execution of one causal FA pass (identical code on all 8 cores)')
X=2; W=40
step(1,X,53.0,W,5.2,'Pack K  (all cores, inside ROI)',r'K for 8 KV heads $\rightarrow$ 128-key panels (32 per head); barrier')
arrow(X+W/2,53.0,X+W/2,51.9)
step(2,X,46.5,W,5.4,'Assign tasks',r'32 Q heads $\times$ 16 groups of $R$=256 rows = 512 tasks;'+'\n'+r'core $c$ takes every 8th (64/core), group order rotated per head')
arrow(X+W/2,46.5,X+W/2,45.4)
# task loop frame
box(1.2,5.0,W+1.6,40.1,'none',ec=MUTE,lw=0.8,ls=(0,(3,2)),z=1)
ax.text(2.0,44.6,r'for each task (64 per core)',fontsize=6.6,color=MUTE,style='italic',va='top')
step(3,X+1,38.0,W-1.5,5.4,'Initialise task',r'pack Q (256$\times$64, FP16); $m\!=\!-\infty$, $\ell\!=\!0$;'+'\n'+r'$O\!=\!0$ (256$\times$64 FP32 = 64 KiB, coherent)')
arrow(X+W/2,38.0,X+W/2,36.9)
# KV-block loop frame
box(2.6,10.0,W-1.0,26.6,'none',ec=MUTE,lw=0.8,ls=(0,(1.2,1.6)),z=1)
ax.text(3.3,36.1,r'for each KV block $j$ ($B_c$=256 keys) up to the causal limit',fontsize=6.6,color=MUTE,style='italic',va='top')
box(X+2,29.5,W-4.0,5.5,SPM,ec=SPMK,lw=0.9)
ax.add_patch(plt.Circle((X+3.35,33.7),0.95,fc=INK,ec='none',zorder=4)); ax.text(X+3.35,33.68,'4',color='white',ha='center',va='center',fontsize=6.6,fontweight='bold',zorder=5)
ax.text(X+4.8,33.7,'Stage V block  ',fontweight='bold',fontsize=7.4,va='center',zorder=5)
ax.text(X+18.2,33.7,'SPM only',fontweight='bold',fontsize=6.6,color=SPMK,va='center',zorder=5)
ax.text(X+4.8,32.25,r'SPMCP: 256 rows $\times$ 2 copies = 32 KiB of V $\rightarrow$ scratchpad'+'\n'+'cache backend: no copy; V is loaded on demand in step 8',color=MUTE,fontsize=6.5,va='top',linespacing=1.35,zorder=5)
arrow(X+W/2,29.5,X+W/2,28.4)
# microtile loop
box(4.0,10.9,W-3.8,17.1,'none',ec=MUTE,lw=0.8,ls=(0,(0.6,1.2)),z=1)
ax.text(4.7,27.6,r'for each microtile of $B_r$=16 query rows ($\times$16; V block reused by all 16)',fontsize=6.4,color=MUTE,style='italic',va='top')
sw=(W-6.4)/2; y1=20.2; y2=11.6; h=5.4; h2=7.6
step(5,4.8,y1,sw-0.4,h,'QK',r'$S = QK^{\top}$  (16$\times$256)'+'\n'+'SVE FP16 FMA, packed K')
step(6,4.8+sw+0.4,y1,sw-0.4,h,'Causal mask',r'diagonal block only:'+'\n'+r'future keys $\rightarrow -\infty$')
step(7,4.8,y2,sw-0.4,h2,'Online softmax',r"$m'=\max(m,\ \mathrm{rowmax}(S)/8)$"+'\n'+r"$\alpha=e^{m-m'}$,  $P=e^{S/8-m'}$"+'\n'+r"$\ell\leftarrow\alpha\ell+\sum P$;  P stored FP16")
step(8,4.8+sw+0.4,y2,sw-0.4,h2,'PV + rescale',r'$O \leftarrow \alpha O + P\,V$ (FP32)'+'\n'+'V source: scratchpad (SPM)'+'\n'+'or L1D/L2 loads (cache)',fc=MEM)
arrow(4.8+sw-0.4,y1+h/2,4.8+sw+0.4,y1+h/2)
ax.annotate('',xy=(4.8+(sw-0.4)/2,y2+h2),xytext=(4.8+sw+0.4+(sw-0.4)/2,y1),arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18',color=LINE,lw=0.8,connectionstyle='arc3,rad=0.0'),zorder=3)
arrow(4.8+sw-0.4,y2+h2/2,4.8+sw+0.4,y2+h2/2)
step(9,X+1,5.8,W-1.5,3.9,'Normalise and write',r'$O/\ell \rightarrow$ FP16 output; next task')
step(10,X,0.6,W,3.9,'Finish',r'SPM: release scratchpad slots; barrier; end of ROI')
arrow(X+W/2,5.0,X+W/2,4.5)

# ---------------- (b) memory hierarchy ----------------
RX=45.5; RW=25.5
panel(RX-0.5,60.6,'(b)','Simulated system (per core, 8 cores)')
def level(y,h,label,detail,fc=MEM,ec=LINE):
    box(RX,y,RW,h,fc,ec); ax.text(RX+1,y+h-1.1,label,fontweight='bold',fontsize=7.0,va='center'); ax.text(RX+1,y+h-2.35,detail,color=MUTE,fontsize=6.3,va='top',linespacing=1.3)
level(53.2,4.6,'O3 core, 2.5 GHz, 2048-bit SVE','ROB 128, LQ/SQ 32/48')
arrow(RX+RW/2,53.2,RX+RW/2,52.4)
level(48.0,4.4,'L1D 64 KiB, 4-way','8 outstanding-miss credits',fc=CACHE,ec=CACHEK)
arrow(RX+RW/2,48.0,RX+RW/2,47.2)
box(RX,39.6,RW,7.6,MEM,LINE); ax.text(RX+1,46.1,'Private L2 512 KiB, 8-way (20 miss credits)',fontweight='bold',fontsize=7.0,va='center')
for i in range(8):
    c=SPM if i<6 else '#ffffff'; ax.add_patch(Rectangle((RX+1+i*2.9,41.6),2.6,2.3,fc=c,ec=SPMK if i<6 else LINE,lw=0.6,zorder=3))
ax.text(RX+1,44.75,'SPM backend: 6 of 8 ways = 384 KiB scratchpad',color=SPMK,fontsize=6.3,va='center')
ax.text(RX+1+3*2.9-0.15,40.55,'6 scratchpad ways',color=SPMK,fontsize=5.8,ha='center',va='center',zorder=4)
ax.text(RX+1+7*2.9-0.15,40.55,'2 cache ways',color=MUTE,fontsize=5.8,ha='center',va='center',zorder=4)
arrow(RX+RW/2,39.6,RX+RW/2,38.8)
level(34.6,4.2,r'Shared L3: 8 banks $\times$ 512 KiB','MESI three-level coherence (Ruby)')
arrow(RX+RW/2,34.6,RX+RW/2,33.8)
level(30.6,3.2,'DRAM','')

# ---------------- (c) workload + result ----------------
panel(RX-0.5,27.6,'(c)','Workload, tiling and outcome')
txt=(r'Llama-3.2-1B attention: 32 Q / 8 KV heads ($d$=64),'+'\n'+r'$T$=4096, batch 1, causal; FP16 GEMMs, FP32 state'+'\n'
     r'Tiling: $R$=256, $B_c$=256, $B_r$=16 (best one-step'+'\n'+r'neighbour for both backends)')
ax.text(RX,25.2,txt,fontsize=6.5,va='top',linespacing=1.4,color=INK)
# bars
ax.text(RX,17.3,'Full-ROI latency, 8 L1D miss credits',fontsize=6.6,fontweight='bold',va='center')
maxv=23499.38; L=RW-8
for i,(lab,v,fc,ec) in enumerate([('Cache',23499.38,CACHE,CACHEK),('SPM',21296.35,SPM,SPMK)]):
    yy=14.6-i*2.9; ax.text(RX,yy,lab,fontsize=6.5,va='center')
    ax.add_patch(Rectangle((RX+4.2,yy-0.95),L*v/maxv,1.9,fc=fc,ec=ec,lw=0.7))
    ax.text(RX+4.2+L*v/maxv+0.4,yy,f'{v/1000:.1f} ms',fontsize=6.4,va='center')
ax.text(RX,9.1,r'SPM speedup 1.10$\times$. Only V placement differs:'+'\n'+r'V is staged into and read from the scratchpad'+'\n'+r'(steps 4, 8) instead of demand-loaded via caches.'+'\n'+r'L1D miss-credit-full events: 300.5 M (cache)'+'\n'+r'vs 154.2 M (SPM).',fontsize=6.3,va='top',linespacing=1.4,color=INK)
ax.text(RX,1.5,'gem5 simulation, synthetic inputs (FA-41).',fontsize=5.9,color=MUTE,va='center',style='italic')
for ext in ('png','pdf','svg'): fig.savefig(f'fa_kernel.{ext}',dpi=300 if ext=='png' else None,facecolor='white')
