import matplotlib.pyplot as plt
import matplotlib
import os
from six.moves import range


matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42

fig = plt.figure(figsize=(9,3))
#fig = plt.figure(frameon=False)
ax = fig.add_subplot(1,1,1)

def frange(start, stop, step=1.0):
    f = start
    while f < stop:
        f += step
        yield f

yticks = []
xticks_labels = ['1/32','1/16','1/8','1/4','1/2','1','2','4','8','16','32','64', '128']
xticks = [2.**i for i in range(-5, 8)]

ax.set_xlabel('Arithmetic Intensity [GFLOP/Byte]')
ax.set_ylabel('Performance [GFLOP/s]')


x_o = list(frange(min(xticks), max(xticks), 0.001))



sp_flops = 473.6
dp_flops = 236.8
ddr_bw = 34.1
edram_bw = 102.4


# Draw DP_DDR
bw = float(ddr_bw)
flops = dp_flops
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='b')
ax.text(12, flops*0.55, 'DP='+str(flops)+'GFlop/s')
ax.text(0.1, 0.1*bw*1.2, 'DDR='+str(bw)+'GB/s',rotation=24)

# Draw DP_eDRAM
bw = float(edram_bw)
flops = dp_flops
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='b')

# Draw SP_DDR
bw = float(ddr_bw)
flops = sp_flops
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='b')

# Draw SP_eDRAM
bw = float(edram_bw)
flops = sp_flops
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='g')


ax.plot([float(dp_flops)/float(edram_bw), xticks[-1] ],[dp_flops,dp_flops], lw=4, color='c')
ax.plot([float(sp_flops)/float(edram_bw), xticks[-1] ],[sp_flops,sp_flops], lw=4, color='m')


ax.text(12, flops*1.1, 'SP='+str(flops)+'GFlop/s')
ax.text(0.1, 0.1*bw*1.2, 'eDRAM='+str(bw)+'GB/s',rotation=24)


ax.text(0.04,500,"eDRAM in Broadwell", fontweight='bold')

#plt.title("Roofline for Intel Core i7-5775c", fontweight="bold")

#============ Stencil ==================
ai = 7.625
ax.plot(ai, 236.8, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,236.8],'r--', lw=2)
ax.text(6.8,1.1,'Stencil',color='r',rotation=90,ha='center')

##============ Stream ==================
ai = 0.0625
ax.plot(ai, 6.8, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,6.8],'r--', lw=2)
ax.text(0.056,1.1,'Stream',color='r',rotation=90,ha='center')

#============ DGEMM ==================
ai = 64
ax.plot(ai, 236.8, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,236.8],'r--', lw=2)
ax.text(59,1.1,'Dgemm',color='r',rotation=90,ha='center')

#============ Cholesky ==================
ai = 42.67
ax.plot(ai, 236.8, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,236.8],'r--', lw=2)
ax.text(38,1.1,'Cholesky',color='r',rotation=90,ha='center')

#============ FFT ==================
ai = 1.0417
ax.plot(ai, 110, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,110],'r--', lw=2)
ax.text(0.94,1.1,'FFT',color='r',rotation=90,ha='center')

#============ SpMV ==================
ai = 0.0835
ax.plot(ai, 9, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,9],'r--', lw=2)
ax.text(0.077,1.1,'SpMV',color='r',rotation=90,ha='center')

#============ SpTrans ==================
ai = 0.4124
ax.plot(ai, 45, 'r+', markersize=12, markeredgewidth=4)
ax.plot([ai,ai],[1,45],'r--', lw=2)
ax.text(0.38,1.1,'SpTrans',color='r',rotation=90,ha='center')







ax.set_xscale('log', basex=2)
ax.set_yscale('log', basex=2)
ax.set_xlim(min(xticks), max(xticks))
# ax.set_yticks([perf, float(max_flops)])
# ax.set_xticks(xticks+[arith_intensity])
ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels)
#ax.grid(axis='x', alpha=0.7, linestyle='--')
fig.savefig('out.eps',format='eps', dpi=500, bbox_inches='tight')
os.system("epstopdf out.eps")
plt.show()


