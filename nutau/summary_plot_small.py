import numpy as np
import sys
import matplotlib as mpl
from collections import OrderedDict
# headless mode
mpl.use('Agg')
# fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fsize1 = 13
fsize2 = 12


def make_rel_bar(data, ax, title='', offset=0,color='b',maxm=0.3):
    #ax = plt.gca()
    for i,name in enumerate(data.keys()):
        x = i + offset
        val = data[name]
        y_68 = np.mean([(val[2]-val[1])/val[2], (val[3] - val[2])/val[2]])
        y_90 = np.mean([(val[2]-val[0])/val[2], (val[4] - val[2])/val[2]])
        #if 'Phase 1' in name: color = 'gray'
        textc = 'k'
        if 'Phase 1' in name:
            name = r'$\blacktriangleleft$ ' + name
            textc = 'red'
        ax.bar(x, y_68, facecolor=color, edgecolor='white',alpha=0.7)
        ax.bar(x, max(y_68,y_90), facecolor=color, edgecolor='white',alpha=0.3)
        #if 'Phase' in name:
        #    ax.bar(x, max(y_68,y_90), facecolor='None',
        #            edgecolor='white',alpha=1.0, hatch='//')
        #    ax.bar(x, max(y_68,y_90), facecolor='None',
        #            edgecolor=color,alpha=1.0)
        ax.text(x+0.4, max(y_68,y_90), ' '+name,ha='center', va= 'bottom',fontsize=fsize2,
                rotation='vertical',color=textc)
    ax.set_ylim((0,maxm))
    ax.text(len(data)/2.+offset, -0.01*maxm, title, ha='center', va=
            'top',fontsize=fsize1)

def make_year_bar(data, ax, title='',offset=0,color='b',maxm=0.3):
    #ax = plt.gca()
    for i,name in enumerate(data.keys()):
        x = i + offset
        val = data[name]
        textc = 'k'
        if 'Phase 1' in name:
            name = r'$\blacktriangleleft$ ' + name
            textc = 'red'
        ax.bar(x, val[1], facecolor=color, edgecolor='white',alpha=0.3)
        ax.bar(x, val[0], facecolor=color, edgecolor='white',alpha=0.7)
        #if 'Phase' in name:
        #    ax.bar(x, max(val), facecolor='None',
        #            edgecolor='white',alpha=1.0, hatch='////')
        #    ax.bar(x, max(val), facecolor='None',
        #            edgecolor=color,alpha=1.0)
        ax.text(x+0.4, val[1], ' ' + name, ha='center', va= 'bottom',fontsize=fsize2,
                rotation='vertical', color=textc)
    ax.set_ylim((0,maxm))
    ax.text(len(data)/2.+offset, -0.05*maxm, title, ha='center', va=
            'top',fontsize=fsize1)

#fig = plt.figure(figsize=(20,10))
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((1,4), (0,0), colspan=4)
#ax1 = plt.subplot2grid((1,7), (0,0), colspan=4)
#ax3 = plt.subplot2grid((1,7), (0,4), colspan=1)
#ax2 = plt.subplot2grid((1,7), (0,5), colspan=2)


ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax2.yaxis.set_ticks_position('left')
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax3.yaxis.set_ticks_position('left')
#ax3.spines['right'].set_visible(False)
#ax3.spines['top'].set_visible(False)

# DATA

#deltam2
dm2 = OrderedDict()
dm2['NOvA'] = [2.42, 2.49, 2.67, 2.85, 2.92]
dm2['T2K'] = [2.365, 2.41, 2.545, 2.66,2.72]
dm2['DeepCore 3y'] = [2.32, 2.37, 2.5,2.61,2.67] 
#dm2['NOvA 6y'] = [2.258, 2.282, 2.35, 2.421,2.452 ]
dm2['NOvA 6y'] = [0, 2.282, 2.35, 2.421,0]
dm2['T2K 2021'] = [2.42, 0, 2.51,0, 2.6]
dm2['T2K 2026'] = [2.46, 0, 2.51,0, 2.56]
dm2['Phase 1 3y'] = [2.6, 2.62, 2.67,2.72 ,2.74]


#theta23 @ 0.5 (maximal)
t23_max = OrderedDict()
t23_max['DeepCore 3y'] = [0.385, 0.405, 0.52, 0.6, 0.62] 
t23_max['T2k'] = [0.428, 0.448 , 0.532, 0.594 , 0.611]
t23_max['NOvA 6y'] = [0, 0.441, 0.5, 0.559, 0] 
t23_max['T2K 2021'] = [0.447, 0, 0.5, 0, 0.582]
t23_max['T2K 2026'] = [0.464, 0, 0.5, 0, 0.562]
t23_max['Phase 1 3y'] = [0.438,0.455,0.532, 0.58, 0.593]

#theta23 @ ~0.4 (off-maximal)
t23_low = OrderedDict()
t23_low['DeepCore 3y'] = [0,0.36,0.4,0.46,0]
t23_low['NOvA'] = [0.352,0.367,0.4,0.448,0.472]
t23_low['NOvA 6y'] = [0, 0.364, 0.388, 0.416,0]
t23_low['T2K 2021'] = [0.403, 0, 0.43, 0, 0.475]
t23_low['T2K 2026'] = [0.415, 0, 0.43, 0, 0.45 ]
t23_low['Phase 1 3y'] = [0.37, 0.38,0.4, 0.423, 0.433]

# octant exclusion at 2teta23 = 0.95, =0.39
# sigma best & worst case
octant = OrderedDict()
octant['DUNE 35kt 5+5y'] = [5.5, 6.5]
octant['NOvA 6y'] = [1.34, 2.53]
octant['Phase 1 3y'] = [1.55, 3.1]

# sterile Ut4 (Um4=0)
#Ut4 = OrderedDict()
#Ut4['Super-K']
#Ut4['DeepCore 3y']
#Ut4['Phase 1']

# NMO years until 3 sigma
# best & worst case
NMO_trueIO = OrderedDict()
#NMO_trueIO['DUNE 10kt'] = [2.1, 4.6]
NMO_trueIO['DUNE 35kt'] = [0.6, 1.31]
#NMO_trueIO['ORCA'] = [2.7, 2.9]
NMO_trueIO['ORCA / PINGU'] = [2.7, 5.6]
NMO_trueIO['JUNO'] = [4., 7.9]
NMO_trueIO['Phase 1'] = [5.5, 7.1]

NMO_trueNO = OrderedDict()
#NMO_trueNO['DUNE 10kt'] = [1.6, 2.9]
NMO_trueNO['DUNE 35kt'] = [0.46, 0.83]
#NMO_trueNO['ORCA'] = [0.6,2.5]
NMO_trueNO['ORCA / PINGU'] = [0.6,4.8]
NMO_trueNO['JUNO'] = [4.5, 8.7]
NMO_trueNO['Phase 1'] = [2.8, 8.1]

make_rel_bar(dm2,ax1,r'$\Delta m_{atm}^2$',1,'green',0.3)
make_rel_bar(t23_max,ax1,r'$\sin^2(\theta_{23})$'+'\n(at maximal)',9,'firebrick',0.3)
make_rel_bar(t23_low,ax1,r'$\sin^2(\theta_{23})$'+'\n(off maximal)',16,'orange',0.3)

#make_year_bar(NMO_trueNO,ax2,'true NO',1,'steelblue',10)
#make_year_bar(NMO_trueIO,ax2,'true IO',6,'teal',10)

#make_year_bar(octant,ax3,'octant sensitivity',1,'purple',8)

ax1.get_xaxis().set_visible(False)
#ax2.get_xaxis().set_visible(False)
#ax3.get_xaxis().set_visible(False)
ax1.set_ylabel('relative uncertainty (68%, 90%)',fontsize=fsize1)
#ax2.set_ylabel(r'livetime to $3\sigma$ sensitivity (best, worst case)',fontsize=fsize1)
#ax3.set_ylabel(r'sensitivity $(\sigma)$ to exclude wrong octant (worst, best case)',fontsize=fsize1)
ax1.set_ylim(0,0.3)
ax1.set_xlim(0,23)
#ax2.set_xlim(0,11)
#ax3.set_xlim(0,5)
plt.show()

plt.savefig('summary_small.pdf')
plt.savefig('summary_small.png')
sys.exit()


#legs = []
#
## nutau
#X = np.array([0,1,2,3])
#Y_68p = np.array([0.,   0.218, 0.17, 0.08])
#Y_90p = np.array([1.,   0.,    0.32, 0.14])
#Y_68m = np.array([0.,   0.218, 0.15, 0.07])
#Y_90m = np.array([0.61, 0.,    0.27, 0.12])
#names = ['Opera', 'SuperK','DC 10y', 'Phase1 3y']
#plt.bar(X, +Y_68p, facecolor='b', edgecolor='white',alpha=0.7)
#plt.bar(X, +Y_90p, facecolor='b', edgecolor='white', alpha=0.3)
#plt.bar(X, -Y_68m, facecolor='b', edgecolor='white', alpha=0.7)
#plt.bar(X, -Y_90m, facecolor='b', edgecolor='white', alpha=0.3)
#for x,y,n in zip(X,np.maximum(Y_68p,Y_90p),names):
#    plt.text(x+0.4, y+0.05, n, ha='center', va= 'bottom',fontsize=10,
#            rotation='vertical')
#legs.append(mpatches.Patch(color='b', label=r'$\nu_\tau$ norm',
#            alpha=0.7))
#
## deltam
#X = np.array([6,7,8,9])
#Y_68p = np.array([0.2,   0.218, 0.17, 0.08])
#Y_90p = np.array([0.3,   0.,    0.32, 0.14])
#Y_68m = np.array([0.3,   0.3, 0.15, 0.07])
#Y_90m = np.array([0.4, 0.5,    0.27, 0.12])
#names = ['T2K', 'NOvA','DC 10y', 'Phase1 3y']
#plt.bar(X, +Y_68p, facecolor='r', edgecolor='white',alpha=0.7)
#plt.bar(X, +Y_90p, facecolor='r', edgecolor='white', alpha=0.3)
#plt.bar(X, -Y_68m, facecolor='r', edgecolor='white', alpha=0.7)
#plt.bar(X, -Y_90m, facecolor='r', edgecolor='white', alpha=0.3)
#for x,y,n in zip(X,np.maximum(Y_68p,Y_90p),names):
#    plt.text(x+0.4, y+0.05, n, ha='center', va= 'bottom',fontsize=10,
#            rotation='vertical')
#legs.append(mpatches.Patch(color='r', label=r'$\Delta m^2_{atm}$', alpha=0.7))
#
## deltam
#X = np.array([11,12,13,14])
#Y_68p = np.array([0.2,   0.218, 0.17, 0.08])
#Y_90p = np.array([0.3,   0.,    0.32, 0.14])
#Y_68m = np.array([0.3,   0.3, 0.15, 0.07])
#Y_90m = np.array([0.4, 0.5,    0.27, 0.12])
#names = ['T2K', 'NOvA','DC 10y', 'Phase1 3y']
#plt.bar(X, +Y_68p, facecolor='g', edgecolor='white',alpha=0.7)
#plt.bar(X, +Y_90p, facecolor='g', edgecolor='white', alpha=0.3)
#plt.bar(X, -Y_68m, facecolor='g', edgecolor='white', alpha=0.7)
#plt.bar(X, -Y_90m, facecolor='g', edgecolor='white', alpha=0.3)
#for x,y,n in zip(X,np.maximum(Y_68p,Y_90p),names):
#    plt.text(x+0.4, y+0.05, n, ha='center', va= 'bottom',fontsize=10,
#            rotation='vertical')
#legs.append(mpatches.Patch(color='g', label=r'$\sin^2(\theta_{23})$', alpha=0.7))
#
#
#ylim = np.array([-1,1.5])
#f_second = 3.
#
## NMO
#X = np.array([16,17,18,19])
#Y = np.array([1.2,   1.5, 3, 3.5])/f_second
#names = ['Dune', 'JUNO','Orca', 'Phase1']
#plt.bar(X, +Y, facecolor='orange', edgecolor='white',alpha=0.7)
#for x,y,n in zip(X,Y,names):
#    plt.text(x+0.4, y+0.05, n, ha='center', va= 'bottom',fontsize=10,
#            rotation='vertical')
#legs.append(mpatches.Patch(color='orange', label=r'NMO', alpha=0.7))
#
#plt.ylim(ylim)
#plt.ylabel('relative uncertainty (68%, 90%)')
#plt.gca().get_xaxis().set_visible(False)
#
#ax2 = plt.gca().twinx()
#ax2.set_ylim(ylim*f_second)
#ax2.set_ylabel('years to 3 sigma')
#
#plt.legend(handles=legs,loc='lower right',ncol=1,
#        frameon=False,numpoints=1,fontsize=10)
#plt.xlim(-1,21)
#plt.axhline(0,c='k')
