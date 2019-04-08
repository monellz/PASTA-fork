import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator, ScalarFormatter
import numpy as np

mplib.rcParams['ps.useafm'] = True
mplib.rcParams['pdf.use14corefonts'] = True
mplib.rcParams['text.usetex'] = True

apps= np.array(["bt","cg","ep","ft","IS","mg"])

#turntile
all10 = np.array([84.23, 25.21, 18.64, 10.26, 50.00, 10.85])
all30 = np.array([13.22,71.77, 33.91, 46.89,50,54.00])
all50 = np.array([2.00,2.26,29.90,25.59,0,34.93])
all80 = np.array([0.53,0.74,17.53,17.24,0,0.20])

def geometricMean(listOfNumbers):
    """Calculates the geometric mean of a list of numbers."""
    return  listOfNumbers

def alpha(ipc):
    return 4/ipc
#def slowdown(a, f):
#    return (a-1)*f +1
def slowdown(a, f):
    return f +1

def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height =rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.00*(rect.get_y()+height), '%d'%int(height),
                ha='center', va='bottom', fontsize=20)

def autolabel_low(rects, ax):
    # attach some text labels
    for rect in rects:
        height= rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 0.979*(rect.get_y()+height), '%d'%int(height),
                ha='center', va='bottom', fontsize=20)
print apps
#runtime=np.append(runtime, [geometricMean(runtime)])
#runtime_no_vcf=np.append(runtime_no_vcf, [geometricMean(runtime_no_vcf)])
#runtime_full=np.append(runtime_full, [geometricMean(runtime_full)])
#runtime_full_no_vcf=np.append(runtime_full_no_vcf, [geometricMean(runtime_full_no_vcf)])

#print runtime
#insts=np.append(insts, [geometricMean(insts)])
#insts_no_vcf=np.append(insts_no_vcf, [geometricMean(insts_no_vcf)])
#insts_full=np.append(insts_full, [geometricMean(insts_full)])
#insts_full_no_vcf=np.append(insts_full_no_vcf, [geometricMean(insts_full_no_vcf)])
#print insts
#overhead15=np.append(overhead15, [geometricMean(overhead15)])
#print overhead15
#def f(t,s,f):
#    return (np.sqrt(s/(2*2*t)))*100*f
#def f2(t,s,f):
#    return (np.sqrt(s/(2*4*t)))*100*f
#def thresDet(thres,s):
#    return np.ceil((thres*s)/0.0005)
#def thresDet2(thres,s):
#    return np.ceil((thres*s)/0.0005)*4
N=6
#ind = np.arange(0, 35.1, 3.5)  # the x locations for the groups
#ind = np.arange(0, 18.1, 1.8)  # the x locations for the groups
#ind = np.arange(0, 12.7, 1.8)  # the x locations for the groups
ytk = np.arange(0, 101, 10)  # the x locations for the groups
ind= np.arange(N)
width = 0.35      # the width of the bars

def plot_gragh(ax, o1, o2,o3,o4):
    rects1 = ax.bar(ind,o1, width, color='b')
    rects2 = ax.bar(ind,o2, width, color='y', bottom=o1)
    rects3 = ax.bar(ind,o3, width, color='g',bottom=(o1+o2)) 
    rects4 = ax.bar(ind,o4, width, color='r',bottom=(o1+o2+o3))
   # rects1 = ax.bar(ind, o1, width, color='black')
   # rects2 = ax.bar(ind, o2, width, color='y')
 #   rects3 = ax.bar(ind, o3, width, color='g')
   # rects4 = ax.bar(ind, o4, width, color='r',bottom=o3)


#    rects3 = ax.bar(ind+width+width, o3, width, color='y')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Distribution%)', fontsize=22)
#    ax.set_title('Simulation overhead for partial DMR with idempotent register renaming', fontsize=26)
#    ax.set_title('(a)' , horizontalalignment='center',
#                 verticalalignment='bottom',y=-0.16,fontsize=26)
    #ax.set_xticks(ind+width)
    ax.set_yticks(ytk)
    #ax.set_ylim(ymin=0)
    zed = [tick.label.set_fontsize(16) for tick in ax.yaxis.get_major_ticks()]
    ax.set_xticks(ind)
    ax.set_xticklabels(apps, fontsize=16, rotation=30)
    #ax.legend( (rects1[0], rects2[0]), ('runtime', 'insts'), loc='upper right', shadow=True, fontsize=26)
    #ax.legend( (rects1[0], rects2[0]), ('runtime', 'insts'), loc='upper center', shadow=True, fontsize=26)
    #ax.legend( (rects1[0], rects2[0]), ('partial-runtime','full-runtime', 'partial-insts', 'full-insts'), shadow=True, fontsize=26)
#    ax.legend( (rects1[0], rects2[0]), ('partial-runtime','full-runtime'),loc='upper center', shadow=True, fontsize=26)
#    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('FullDMR-w-recovery', 'FullDMR', 'Clover', 'Clover-w/o-recovery'),loc='upper left', shadow=True)
    #ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Recovery-w/-FullDMR', 'FullDMR', 'Recovery-w/-TailDMR', 'TailDMR'),loc='upper center', shadow=True)
    #ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('Idem-w/-FullDMR', 'FullDMR', 'Idem-w/-TailDMR', 'TailDMR'),loc='upper center', shadow=True, bbox_to_anchor=(0.48, 1))
   # ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('10', '30', '50', '80'),loc='upper center', shadow=True)
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('10', '40','70','100'),loc='right', shadow=True)
 #   ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('10', '30', '50', '80'),loc='right', shadow=False)
    autolabel_low(rects1, ax)
    autolabel_low(rects2, ax)
    autolabel_low(rects3, ax)
    autolabel(rects4, ax)

def plot_gragh2(ax, o1, o2,o3, o4):
    rects1 = ax.bar(ind, o1, width)
    rects2 = ax.bar(ind, o2, width, color='r')
    rects3 = ax.bar(ind+width, o3, width, color='black')
    rects4 = ax.bar(ind+width, o4, width, color='y')
#    rects3 = ax.bar(ind+width+width, o3, width, color='y')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('dynamic instrucions ratio(dmr/origin%)', fontsize=20)
#    ax.set_title('Simulation overhead for partial DMR with idempotent register renaming', fontsize=26)
    ax.set_title('(b)' , horizontalalignment='center',
                 verticalalignment='bottom',y=-0.16,fontsize=26)
    #ax.set_xticks(ind+width)
    ax.set_xticks(ind)
    ax.set_xticklabels(apps, fontsize=16, rotation=30)
    #ax.legend( (rects1[0], rects2[0]), ('runtime', 'insts'), loc='upper right', shadow=True, fontsize=26)
    #ax.legend( (rects1[0], rects2[0]), ('runtime', 'insts'), loc='upper center', shadow=True, fontsize=26)
    #ax.legend( (rects1[0], rects2[0]), ('partial-runtime','full-runtime', 'partial-insts', 'full-insts'), shadow=True, fontsize=26)
#    ax.legend( (rects1[0], rects2[0]), ('partial-insts','full-insts'),loc='upper center', shadow=True, fontsize=26)
    #ax.legend( (rects1[0], rects2[0],rects3[0], rects4[0]), ('partial-insts','partial-insts-w/o-rename','full-insts', 'full-insts-w/o-rename'),loc='upper center', shadow=True)
    #ax.legend( (rects1[0], rects2[0],rects3[0], rects4[0]), ('full-insts', 'full-insts-w/o-rename', 'partial-insts','partial-insts-w/o-rename'),loc=(10, 220), shadow=True)
    ax.legend( (rects1[0], rects2[0],rects3[0], rects4[0]), ('full-insts', 'full-insts-w/o-rename', 'partial-insts','partial-insts-w/o-rename'),loc='upper center', shadow=True)
    autolabel(rects1, ax)
    autolabel_low(rects2, ax)
    autolabel(rects3, ax)
    autolabel_low(rects4, ax)


#fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(nrows=2, ncols=4)
#fig, ((ax0, ax1, ax2),(ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11)) = plt.subplots(nrows=4, ncols=3)
#fig, (ax0,ax1) = plt.subplots(nrows=2, ncols=1)
fig, (ax0) = plt.subplots(nrows=1, ncols=1)

# plot_gragh(ax0,arange(1,1001,1), 0.45, ax0ticks, 'a')
# plot_gragh(ax0, runtime_full, runtime_full_no_vcf, runtime, runtime_no_vcf)
plot_gragh(ax0,all10, all30, all50, all80)

#plot_gragh2(ax1,insts, insts_no_vcf, insts_full,insts_full_no_vcf)

plt.show()

#import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#
#maj`orLocator   = MultipleLocator(20)
#majorFormatter = FormatStrFormatter('%d')
#minorLocator   = MultipleLocator(5)
#
#
#t = np.arange(0.0, 100.0, 0.1)
#s = np.sin(0.1*np.pi*t)*np.exp(-t*0.01)
#
#fig, ax = plt.subplots()
#plt.plot(t,s)
#
#ax.xaxis.set_major_locator(majorLocator)
#ax.xaxis.set_major_formatter(majorFormatter)
#
##for the minor ticks, use no labels; default NullFormatter
#ax.xaxis.set_minor_locator(minorLocator)
#
#plt.show()
