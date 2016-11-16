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


def make_err_bar(data, ax, offset=0,color='b',maxm=0.3):
    #ax = plt.gca()
    for i,name in enumerate(data.keys()):
        x = i + offset + 0.4
        val = data[name]
        textc = 'k'
        if 'Phase 1' in name:
            name = r'$\blacktriangleleft$ ' + name
            textc = 'red'
        ax.errorbar(np.array([x]),np.array([val[2]]),yerr=np.array([[val[2]-val[1]],[val[3]-val[2]]]),fmt='.',color='navy')
        eb = ax.errorbar(np.array([x]),np.array([val[2]]),yerr=np.array([[val[2]-val[0]],[val[4]-val[2]]]),fmt='.',color='navy')
        eb[-1][0].set_linestyle(':')
        ax.text(x+0.4, max(val[3],val[4]), ' '+name,ha='center', va= 'bottom',fontsize=fsize2,
                rotation='vertical',color=textc)
    ax.set_ylim((0,maxm))
    ax.text(len(data)/2.+offset, -0.05*maxm, title, ha='center', va=
            'bottom',fontsize=fsize1)

def make_bar(data, ax,offset=0,color='b',maxm=0.3):
    #ax = plt.gca()
    for i,name in enumerate(data.keys()):
        x = i + offset
        val = data[name]
        textc = 'k'
        if 'Phase 1' in name:
            name = r'$\blacktriangleleft$ ' + name
            textc = 'red'
        ax.bar(x, val, facecolor=color, edgecolor='white',alpha=0.7)
    ax.set_ylim((0,maxm))
    ax.text(len(data)/2.+offset, -0.05*maxm, title, ha='center', va=
            'bottom',fontsize=fsize1)

fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace=0)
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0))


#ax1.yaxis.set_ticks_position('left')
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
#ax2.yaxis.set_ticks_position('left')
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax3.yaxis.set_ticks_position('left')
#ax3.spines['right'].set_visible(False)
#ax3.spines['top'].set_visible(False)

# DATA

tau = OrderedDict()
sigma = OrderedDict()
tau['Opera'] = [0.7, 0, 1.8, 0, 3.6]
tau['SuperK'] = [0, 1.15, 1.47, 1.79, 0]
tau['DeepCore 10y'] = [0.684020164917, 0.845601047987, 1.0, 1.17198675026, 1.37258047105]
#tau['Phase1 0.1 y'] = [0.346, 0.659, 1.000, 1.387, 1.856]
#tau['Phase1 0.2 y'] = [0.504, 0.744, 1.000, 1.285, 1.610]
#tau['Phase1 0.3 y'] = [0.613, 0.801, 1.000, 1.212, 1.442]
#tau['Phase1 0.5 y'] = [0.673, 0.833, 1.000, 1.178, 1.371]
#tau['Phase1 0.7 y'] = [0.707, 0.850, 1.000, 1.157, 1.326]
tau['Phase1 1 y'] = [0.756, 0.878, 1.000, 1.130, 1.269]
#tau['Phase1 1.5 y'] = [0.792, 0.896, 1.000, 1.107, 1.222]
tau['Phase1 2 y'] = [0.817, 0.911, 1.000, 1.093, 1.197]
#tau['Phase1 2.5 y'] = [0.834, 0.921, 1.000, 1.081, 1.176]
tau['Phase1 3 y'] = [0.846, 0.930, 1.000, 1.075, 1.161]
#tau['Phase1 4.0 y'] = [0.865, 0.939, 1.000, 1.062, 1.141]

sigma['Opera'] = 5.1
sigma['SuperK'] = 4.6
sigma['DeepCore 10y'] = 7.2
sigma['Phase1 1y'] = 8.73965189
sigma['Phase1 2y'] = 11.82911535
sigma['Phase1 3y'] = 14.18633765


make_err_bar(tau,ax1,1,'green',4)
make_bar(sigma,ax2,1,'green',16)

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax1.set_ylabel(r'$\nu_\tau$ normalization (68%, 90%)',fontsize=fsize1)
ax2.set_ylabel(r'significance ecluding no appearance $(\sigma)$',fontsize=fsize1)
ax1.set_xlim(0.8,7)
ax2.set_xlim(0.8,7)
plt.show()

plt.savefig('nutau.pdf')
plt.savefig('nutau.png')
sys.exit()

