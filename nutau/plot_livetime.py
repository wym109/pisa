#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from scipy.stats import chi2
from scipy import optimize
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from scipy.ndimage import zoom
from cycler import cycler
import collections
from pisa.utils.fileio import from_file

def plot(name,hypos,asimovs,x_var='nutau_cc_norm',dir='.'):

    fig = plt.figure()
    plt.grid()
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111)
    #a_text = AnchoredText(r'DeepCore $\nu_\tau$ appearance', loc=2, frameon=False)
    a_text = AnchoredText(r'PINGU v39 $\nu_\tau$ appearance', loc=2, frameon=False)
    ax.add_artist(a_text)
    sigmas = []
    years = []
    minorLocator = MultipleLocator(0.1)
    for i,asimov in enumerate(asimovs): 
        years.append(sorted(asimov.keys()))
        times = []
        best = []
        best_ms = []
        best_m2s = []
        best_ps = []
        best_p2s = []
        #sigmas.append([[]]*len(hypos))
        #print sigmas
        sigmas.append([])
        for time in years[-1]:
            data = np.array(asimov[time][name])
            best_idx = np.argmin(data)
            sigmas[i].append(np.sqrt(data))
            #for j,hypo in enumerate(hypos):
            #    print hypo
            #    print np.sqrt(data[j])
            #    sigmas[i][j].append(np.sqrt(data[j]))
            best.append(hypos[best_idx])
            best_ms.append(np.interp(1,data[best_idx::-1],hypos[best_idx::-1]))
            best_m2s.append(np.interp(4.,data[best_idx::-1],hypos[best_idx::-1]))
            #best_m2s.append(np.interp(2.71,data[best_idx::-1],hypos[best_idx::-1]))
            best_ps.append(np.interp(1,data[best_idx:],hypos[best_idx:]))
            #best_p2s.append(np.interp(2.71,data[best_idx:],hypos[best_idx:]))
            best_p2s.append(np.interp(4.,data[best_idx:],hypos[best_idx:]))
        if i == 0:
            #ax.fill_between(years[-1],best_m2s,best_p2s,facecolor='g', linewidth=0, alpha=0.15, label='90% range improved sys')
            #ax.fill_between(years[-1],best_ms,best_ps,facecolor='g', linewidth=0, alpha=0.3, label='68% range improved sys')
            ax.fill_between(years[-1],best_m2s,best_p2s,facecolor='g', linewidth=0, alpha=0.15, label='95% range')
            ax.fill_between(years[-1],best_ms,best_ps,facecolor='g', linewidth=0, alpha=0.3, label='68% range')
            ax.text(max(years[-1])*1.01,0.98*best_ms[-1],' %.2f'%best_ms[-1],fontsize=10,color='g')
            ax.text(max(years[-1])*1.01,0.98*best_ps[-1],' %.2f'%best_ps[-1],fontsize=10,color='g')
            ax.text(max(years[-1])*1.01,0.98*best_m2s[-1],'(%.2f)'%best_m2s[-1],fontsize=10,color='g')
            ax.text(max(years[-1])*1.01,0.98*best_p2s[-1],'(%.2f)'%best_p2s[-1],fontsize=10,color='g')
        else:
            ax.plot(years[-1],best_m2s, color='k', linewidth=1, linestyle=':', label='95% range baseline')
            ax.plot(years[-1],best_p2s, color='k', linewidth=1, linestyle=':')
            ax.plot(years[-1],best_ms,  color='k', linewidth=1, label='68% range baseline')
            ax.plot(years[-1],best_ps,  color='k', linewidth=1)
        #ax.set_xlabel('livetime (years)')
        ax.set_xlabel('livetime (months)')
        ax.set_ylabel(r'$\nu_\tau$ normalization')
        ax.set_ylim([0,2])
        #ax.set_yticks(np.linspace(0,2,21))
        ax.yaxis.set_minor_locator(minorLocator)

        ax.set_xlim([years[0][0],years[0][-1]])
        ax.plot(years[-1],best, color='g', linewidth=1)
    
    ax.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10)

    plt.show()
    plt.savefig('%s/nutaunorm.png'%(dir), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('%s/nutaunorm.pdf'%(dir), facecolor=fig.get_facecolor(), edgecolor='none')


    plt.clf()
    plt.grid()
    ax = fig.add_subplot(111)
    num = 10
    ax.set_xlim([years[0][0],years[0][-1]])
    sigma = np.array(sigmas[0]).T
    colormap = plt.cm.Spectral 
    colors = [colormap(i) for i in np.linspace(0, 1,num)]
    for j,c in enumerate(colors):
        ax.plot(years[0],sigma[j], color=c,label=r'rejection $\nu_\tau=$ %s'%hypos[j])
    #ax.plot(years[0],sigmas[0], color='b', label='baseline')
    #ax.plot(years[1],sigmas[1], color='g', label='improved sys')
    #ax.set_xlabel('livetime (years)')
    ax.set_xlabel('livetime (months)')
    ax.set_ylabel(r'$\nu_\tau$ appearance significance $(\sigma)$')
    #a_text = AnchoredText(r'DeepCore $\nu_\tau$ appearance', loc=2, frameon=False)
    a_text = AnchoredText(r'PINGU v39 $\nu_\tau$ appearance', loc=2, frameon=False)
    ax.add_artist(a_text)
    ax.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10)

    plt.show()
    plt.savefig('%s/nutausigma.png'%(dir), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('%s/nutausigma.pdf'%(dir), facecolor=fig.get_facecolor(), edgecolor='none')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('-d1','--dir1',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('-x','--x-var',help='variable to plot against', default='nutau_cc_norm') 
    parser.add_argument('--dist',action='store_true') 
    parser.add_argument('--asimov',action='store_true') 
    args = parser.parse_args()

    #asimovs = [{},{}]
    asimovs = [{}]

    for i,dir in enumerate([args.dir]):
        # get llh denominators for q for each seed
        for filename in os.listdir(dir):
            if filename.endswith('.json'):
                file = from_file(dir +'/'+filename)
                cond = file[0][0]
                glob = file[0][1]
                if file[0][0].has_key('llh'): metric = 'llh'
                elif file[0][0].has_key('conv_llh'): metric = 'conv_llh'
                elif file[0][0].has_key('chi2'): metric = 'chi2'
                elif file[0][0].has_key('mod_chi2'): metric = 'mod_chi2'
                elif file[0][0].has_key('barlow_llh'): metric = 'barlow_llh'
                else: continue
                plot_dict = {}
                flag = False
                name = filename.rstrip('.json')
                year = int(name.split('_')[0])
                asimovs[i][year] = {}
                for key,val in cond.items():
                    if key == 'warnflag':
                        flag = any(val)
                    elif key == metric:
                        if 'chi2' in metric:
                            plot_dict['llh'] = (np.array(val) - glob[metric])
                        else:
                            plot_dict['llh'] = 2*(np.array(val) - glob[metric])
                    elif key == args.x_var:
                        hypos = val[0]
                        asimovs[i][year]['hypos'] = hypos
                    else:
                         continue
                    for key,val in plot_dict.items():
                        asimovs[i][year][key] = [x for x in val]
    plot('llh',hypos,asimovs,args.x_var,args.dir)
