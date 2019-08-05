# Script to plot Roofline Chart

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
xticks_labels = ['1/32','1/16','1/8','1/4','1/2','1','2','4','8','16','32','64', '128', '256']
xticks = [2.**i for i in range(-5, 7)]

ax.set_xlabel('Arithmetic Intensity [#FLOPs/#Bytes]')
ax.set_ylabel('Performance [GFLOPS]')


x_o = list(frange(min(xticks), max(xticks), 0.001))


#Set the values
peak_sp = 14900
ert_flops = 7826
ert_dram = 791
peak_dram = 900
ert_llc = 3332

# Draw the Ceilings
bw = ert_dram
flops = ert_flops
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='b')
ax.text(10.8, flops*0.45, 'ERT '+str(flops)+' GFLOPS')
ax.text(0.75, 0.1*bw*16, 'ERT DRAM '+str(bw)+' GB/s',rotation=15)

bw = float(peak_dram)
flops = ert_flops
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='b')


bw = float(ert_dram)
flops = peak_sp
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='b')


bw = float(peak_dram)
flops = peak_sp
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='g')


bw = float(ert_llc)
flops = peak_sp
ax.plot(x_o, [min(bw*x, float(flops)) for x in x_o], lw=4, color='#FF8C00')

bw = float(peak_dram)
ax.plot([float(ert_flops)/float(ert_llc), xticks[-1] ],[ert_flops,ert_flops], lw=4, color='c')
ax.plot([float(peak_sp)/float(ert_llc), xticks[-1] ],[peak_sp,peak_sp], lw=4, color='m')

#Set the labels
ax.text(7.2, flops*1.32, 'Peak SP '+str(flops)+' GFLOPS')
bw = peak_dram
ax.text(0.15, 0.1*bw*13.5, 'Peak DRAM '+str(bw)+' GB/s',rotation=15)
bw = ert_llc
ax.text(0.058, 0.1*bw*5.9, 'ERT LLC (L2) '+str(ert_llc)+' GB/s',rotation=14)

#Graph title
ax.text(1,41000,"Empirical Roofline Graph - DGX-1V", fontweight='bold', ha='center')

#plt.title("Roofline for Intel Core i7-5775c", fontweight="bold")

#============ TEW ==================
ai = 0.08333
ax.plot(ai, 60, 'r+', markersize=12, markeredgewidth=2.75)
ax.plot([ai,ai],[1,60],'r--', lw=2)
ax.text(0.094,8,'TEW',color='r',rotation=90,ha='center')

#============ TS ==================
ai = 0.125
ax.plot(ai, 85, 'r+', markersize=12, markeredgewidth=2.75)
ax.plot([ai,ai],[1,85],'r--', lw=2)
ax.text(0.143,8,'TS',color='r',rotation=90,ha='center')

#============ TTV, MTTKRP ==================
ai = 0.25
ax.plot(ai, 170, 'r+', markersize=12, markeredgewidth=2.75)
ax.plot([ai,ai],[1,170],'r--', lw=2)
ax.text(0.29,70,'TTV, MTTKRP',color='r',rotation=90,ha='center')

#============ TTM ==================
ai = 0.5
ax.plot(ai, 337, 'r+', markersize=12, markeredgewidth=2.75)
ax.plot([ai,ai],[1,337],'r--', lw=2)
ax.text(0.58,13,'TTM',color='r',rotation=90,ha='center')


#plt.rc('axes', axisbelow=True)
ax.set_axisbelow(True)
plt.grid(b=True, which='both', axis='both', linestyle='-', color='0.85')




ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(min(xticks), max(xticks))
ax.set_ylim(0,35000)
# ax.set_yticks([perf, float(max_flops)])
# ax.set_xticks(xticks+[arith_intensity])
ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels)
#ax.grid(axis='x', alpha=0.7, linestyle='--')
fig.savefig('volta_graph.eps',format='eps', dpi=500, bbox_inches='tight')
fig.savefig('volta_graph.png',format='png', dpi=500, bbox_inches='tight')
#os.system("epstopdf cori_graph.eps")
plt.show()


