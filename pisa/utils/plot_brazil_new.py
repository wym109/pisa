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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys
from scipy.stats import chi2
from scipy.optimize as optimize
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from scipy.ndimage import zoom
from cycler import cycler
import collections
from pisa.utils.fileio import from_file

def plot(name,data,hypos,trials):
    
    median = np.array([])
    sigmam = np.array([])
    sigmap = np.array([])
    sigmam2 = np.array([])
    sigmap2 = np.array([])
    for datum in data:
        median = np.append(median,np.percentile(datum,50))
        sigmam = np.append(sigmam,np.percentile(datum,16))
        sigmam2 = np.append(sigmam2,np.percentile(datum,5))
        sigmap = np.append(sigmap,np.percentile(datum,84))
        sigmap2 = np.append(sigmap2,np.percentile(datum,95))
    if name == 'llh':
        h0 = '%.2f'%(np.sqrt(median[0]))
        h0_up = '%.2f'%(np.sqrt(sigmap[0])-np.sqrt(median[0]))
        h0_down = '%.2f'%(np.sqrt(median[0])-np.sqrt(sigmam[0]))
        print 'Significance for excluding H0: %s + %s - %s'%(h0, h0_up, h0_down) 
    
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y','c','m','k']*2) +
                           cycler('linestyle', ['-']*7+['--']*7)))


    fig = plt.figure()
    fig.patch.set_facecolor('none')
    if name == 'llh':
        ax = plt.subplot2grid((6,1), (0,0), rowspan=5)
    else:
        ax = fig.add_subplot(111)
    ax.fill_between(hypos,sigmam2,sigmap2,facecolor='b', linewidth=0, alpha=0.15, label='90% range')
    ax.fill_between(hypos,sigmam,sigmap,facecolor='b', linewidth=0, alpha=0.3, label='68% range')
    ax.plot(hypos,median, color='k', label='median')
    ax.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10)
    ax.set_xlabel(r'$\nu_{\tau}$ normalization')
    ax.set_xlim(min(hypos),max(hypos))
    if name == 'llh':
        tex= r'$H_0$ at %s $\sigma ^{+%s}_{-%s}$'%(h0, h0_up, h0_down)
        print tex
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\n'%(2.5,trials)+'Rejection of '+ tex, loc=2, frameon=False)
    else:
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\n'%(2.5,trials)+'nuisnace pulls', loc=2, frameon=False)
    ax.add_artist(a_text)
    #ax.patch.set_facecolor('white')
    #ax.set_axis_bgcolor('white') 
    #ax.set_frame_on(False)
    if name == 'llh':
        best_idx = np.argmin(median)
        best = hypos[best_idx]
        best_ms = np.interp(1,median[best_idx::-1],hypos[best_idx::-1])
        best_m2s = np.interp(4,median[best_idx::-1],hypos[best_idx::-1])
        best_ps = np.interp(1,median[best_idx:],hypos[best_idx:])
        best_p2s = np.interp(4,median[best_idx:],hypos[best_idx:])
        print best_m2s,best_ms,best,best_ps,best_p2s
        ax2 = plt.subplot2grid((6,1), (5,0),sharex=ax)
        ax2.errorbar(np.array([1.47]),np.array([1.]),xerr=np.array([[0.32],[0.32]]),fmt='.',color='forestgreen')
        ax2.text(0.05,0.75,r'Super-K 2016 (68%)',size=8)
        ax2.errorbar(np.array([1.8]),np.array([2.]),xerr=np.array([[1.1],[1.8]]),fmt='.',color='sienna')
        ax2.text(0.05,1.75,r'Opera 2015 (90%)',size=8)
        ax2.errorbar(np.array([best]),np.array([3.]),xerr=np.array([[best-best_ms],[best_ps-best]]),fmt='.',color='mediumblue')
        ax2.errorbar(np.array([best]),np.array([3.]),xerr=np.array([[best-best_m2s],[best_p2s-best]]),fmt='.',color='mediumblue')
        ax2.text(0.05,2.75,r'Expected (68%, 95%)',size=8)
        ax2.set_ylim(0,4)
        ax2.set_xlim(0,2)
        ax2.get_yaxis().set_visible(False)
        fig.subplots_adjust(hspace=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_xlabel(r'$\nu_{\tau}$ CC normalization')
        for i in [0.5,1,1.5]:
            ax2.axvline(i,color='k', linestyle='-',alpha=0.2)
    if name == 'llh':
        ax.set_ylabel(r'$-2\Delta LLH$')
        ax.set_ylim([0,30])
        for i in [1,4,9,16,25,36]:
            ax.axhline(i,color='k', linestyle='-',alpha=0.2)
            ax.text(2.02,i-0.2,r'$%i\ \sigma$'%np.sqrt(i))
    else:
        ax.set_ylabel(name)
        #if params.has_key(name):
        #    params_value = params[name]['value']
        #    ax.axhline(params_value, color='g')
        #    ax.text(1.80,params_value,r'nominal',color='g',size=10)
        #    if params[name]['prior'].has_key('sigma'):
        #        params_sigma = params[name]['prior']['sigma']
        #        params_sigma = params[name]['prior']['sigma']
        #        ymin = params_value - params_sigma
        #        ymax = params_value + params_sigma
        #        ax.axhline(ymin, color='g', linestyle='--')
        #        ax.axhline(ymax, color='g', linestyle='--')
        #        ax.text(1.88,ymin,r'$-1\sigma$',color='g')
        #        ax.text(1.88,ymax,r'$+1\sigma$',color='g')
        #        ax.set_ylim(ymin-0.4*(params_value-ymin), ymax+0.4*(ymax-params_value))
        #    else:
        #        delta = ax.get_ylim()[1]-ax.get_ylim()[0]
        #        ax.set_ylim(ax.get_ylim()[0]-0.4*delta, ax.get_ylim()[1]+0.4*delta)
    for i in [0.5,1,1.5]:
        ax.axvline(i,color='k', linestyle='-',alpha=0.2)
    plt.show()
    plt.savefig('q1_%s.png'%name, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('q1_%s.pdf'%name, facecolor=fig.get_facecolor(), edgecolor='none')

def dist(data,name,hypos, asimov_hypos, params,trials):

    for hypo in hypos:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data['%.1f'%hypo],20,alpha=0.5)
        ax.axvline(np.median(data['%.1f'%hypo]),color='r',linewidth=2)
        ax.patch.set_facecolor('white')
        if params.has_key(name):
            if params[name].has_key('value'):
                params_value = params[name]['value']
                ax.axvline(params_value, color='g',linewidth=2)
        ax.set_xlabel(name)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
        ax.set_title('profile likelihood, %s years, %s trials'%(params['livetime']['value'],trials))
        plt.show()
        plt.savefig('q%.1f_%s.png'%(hypo,name),transparent=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('--dist',action='store_true') 
    args = parser.parse_args()

    x_var = 'nutau_cc_norm'

    total = 0
    hypos = []
    data = {}

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_file(args.dir +'/'+filename)
            cond = file[0][0]
            glob = file[0][1]
            plot_dict = {}
            flag = False
            for key,val in cond.items():
                if key == 'warnflag':
                    flag = any(val)
                elif key == 'llh':
                    plot_dict[key] = 2*(np.array(val) - glob[key])
                elif key == x_var:
                    hypos = val[0]
                else:
                    plot_dict[key] = val[0]
            if flag:
                print 'skipping file %s'%filename
            else:
                total += 1
                for key,val in plot_dict.items():
                    if data.has_key(key):
                        [x.append(y) for x,y in zip(data[key],val)]
                    else:
                        data[key] = [[x] for x in val]

            #params = file['template_settings']['params']
            #for trial in file['trials']:
            #    ts = trial['test_statistics']
            #    #if ts == 'asimov':
            #    #    asimov_hypos[name] = trial['nutau_norm']
            #    #    asimov_results[name] = trial['fit_results'][0]
            #    #    asimov_results[name]['llh'] = trial['q']
            #    #    if name == 'nufit_prior' or name =='no_prior' or name == 'nominal':
            #    #        sys = asimov_results[name].keys()
            #    elif ts == 'profile':
            #        data['hypos'] = trial['nutau_norm']
            #        for key in trial['fit_results'][0]:
            #            if key == 'llh':
            #                val = trial['q']
            #            else:
            #                val = trial['fit_results'][0][key]
            #            if data.has_key(key):
            #                [x.append(y) for x,y in zip(data[key],val)]
            #            else:
            #                data[key] = [[x] for x in val]
            #        total += 1

    for key,val in data.items():
        plot(key,val,hypos,total)
    #if args.dist:
    #    for s in syslist:
    #        dist(results, asimov_results,hypos, asimov_hypos, params, total)
    #else:
    #    for s in data.keys():
    #        if s != 'hypos':
    #            plot(s,data[s], asimov_results,data['hypos'], asimov_hypos, params, total)

