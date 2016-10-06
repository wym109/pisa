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
from scipy import optimize
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
from scipy.ndimage import zoom
from cycler import cycler
import collections
from pisa.utils.fileio import from_file
from pisa.utils.parse_config import parse_config
from pisa.core.param import ParamSet

def plot(name,data,hypos,asimov,trials,x_var='nutau_cc_norm',dir='.'):
   
    if len(data) > 0: 
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
        if name in ['llh', 'conv_llh', 'chi2', 'mod_chi2']:
            h0 = '%.2f'%(np.sqrt(median[0]))
            h0_up = '%.2f'%(np.sqrt(sigmap[0])-np.sqrt(median[0]))
            h0_down = '%.2f'%(np.sqrt(median[0])-np.sqrt(sigmam[0]))
            print 'Significance for excluding H0: %s + %s - %s'%(h0, h0_up, h0_down) 
    
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y','c','m','k']*2) +
                           cycler('linestyle', ['-']*7+['--']*7)))


    fig = plt.figure()
    fig.patch.set_facecolor('none')
    if name in ['llh', 'conv_llh', 'chi2', 'mod_chi2']:
        ax = plt.subplot2grid((6,1), (0,0), rowspan=5)
    else:
        ax = fig.add_subplot(111)
    if len(data) > 0: 
        ax.fill_between(hypos,sigmam2,sigmap2,facecolor='b', linewidth=0, alpha=0.15, label='90% range')
        ax.fill_between(hypos,sigmam,sigmap,facecolor='b', linewidth=0, alpha=0.3, label='68% range')
        ax.plot(hypos,median, color='k', label='median')
    for fname, asi in asimov.items():
        ax.plot(asi['hypos'], asi[name], label=fname)
    ax.legend(loc='upper right',ncol=1, frameon=False,numpoints=1,fontsize=10)
    ax.set_xlabel(r'$\nu_{\tau}$ CC normalization')
    ax.set_xlim(min(hypos),max(hypos))
    if name in ['llh', 'conv_llh', 'chi2', 'mod_chi2'] and x_var == 'nutau_cc_norm' and len(data)>0:
        tex= r'$H_0$ at %s $\sigma ^{+%s}_{-%s}$'%(h0, h0_up, h0_down)
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\n'%(2.5,trials)+'Rejection of '+ tex, loc=2, frameon=False)
    else:
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s years, %s trials\n'%(2.5,trials)+'nuisnace pulls', loc=2, frameon=False)
    ax.add_artist(a_text)
    #ax.patch.set_facecolor('white')
    #ax.set_axis_bgcolor('white') 
    #ax.set_frame_on(False)
    if name in ['llh', 'conv_llh', 'chi2', 'mod_chi2'] and len(data) > 0:
        best_idx = np.argmin(median)
        best = hypos[best_idx]
        best_ms = np.interp(1,median[best_idx::-1],hypos[best_idx::-1])
        best_m2s = np.interp(4,median[best_idx::-1],hypos[best_idx::-1])
        best_ps = np.interp(1,median[best_idx:],hypos[best_idx:])
        best_p2s = np.interp(4,median[best_idx:],hypos[best_idx:])
        if x_var == 'nutau_cc_norm':
            ax2 = plt.subplot2grid((6,1), (5,0),sharex=ax)
            ax2.errorbar(np.array([1.47]),np.array([1.]),xerr=np.array([[0.32],[0.32]]),fmt='.',color='forestgreen')
            ax2.text(0.05,0.75,r'Super-K 2016 (68%)',size=8)
            ax2.errorbar(np.array([1.8]),np.array([2.]),xerr=np.array([[1.1],[1.8]]),fmt='.',color='sienna')
            ax2.text(0.05,1.75,r'Opera 2015 (90%)',size=8)
            #ax2.errorbar(np.array([best]),np.array([3.]),xerr=np.array([[best-best_ms],[best_ps-best]]),fmt='.',color='mediumblue')
            ax2.errorbar(np.array([best]),np.array([3.]),xerr=np.array([[best-best_ms],[best_ps-best]]),color='mediumblue')
            #ax2.errorbar(np.array([best]),np.array([3.]),xerr=np.array([[best-best_m2s],[best_p2s-best]]),fmt='.',color='mediumblue')
            ax2.errorbar(np.array([best]),np.array([3.]),xerr=np.array([[best-best_m2s],[best_p2s-best]]),color='mediumblue')
            ax2.text(0.05,2.75,r'Expected (68%, 95%)',size=8)
            ax2.set_ylim(0,4)
            ax2.set_xlim(min(hypos),max(hypos))
            ax2.get_yaxis().set_visible(False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            fig.subplots_adjust(hspace=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax2.set_xlabel(r'$\nu_{\tau}$ CC normalization')
            for i in [0.5,1,1.5]:
                ax2.axvline(i,color='k', linestyle='-',alpha=0.2)
        elif x_var == 'deltacp':
            ax.set_xlabel(r'$\Delta_{CP}$')
    if name in ['llh', 'conv_llh', 'chi2', 'mod_chi2']:
        if 'chi2' in name: 
            ax.set_ylabel(r'$\Delta \chi^2$')
        else:
            ax.set_ylabel(r'$-2\Delta LLH$')
        if x_var == 'nutau_cc_norm':
            ax.set_ylim([0,25])
            #ax.set_ylim([0,9])
        elif x_var == 'deltacp':
            ax.set_ylim([0,0.02])
        for i in [1,4,9,16,25,36]:
            ax.axhline(i,color='k', linestyle='-',alpha=0.2)
            ax.text(2.02,i-0.2,r'$%i\ \sigma$'%np.sqrt(i))
    else:
        ax.set_ylabel(name)
    for i in [0.5,1,1.5]:
        ax.axvline(i,color='k', linestyle='-',alpha=0.2)
    plt.show()
    plt.savefig('%s/q1_%s.png'%(dir,name), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('%s/q1_%s.pdf'%(dir,name), facecolor=fig.get_facecolor(), edgecolor='none')

def dist(name,data, asimov, params, trials, dir):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    h,b,p = ax.hist(data,20, linewidth=2, histtype='step', color='k',normed=True)
    ax.ticklabel_format(useOffset=False)
    median = np.median(data)
    mean = np.mean(data)
    std = np.std(data)
    uncert = std/np.sqrt(trials)
    ax.axvline(median,color='r',linewidth=2, label='median')
    ax.axvline(mean,color='b',linewidth=2, label='mean')
    ax.axvspan(mean-uncert, mean+uncert, alpha=0.2, color='b', linewidth=0, label='uncert. on mean')
    ax.patch.set_facecolor('white')
    if name in params.names:
        params_value = params[name].value.m
        sigma = 0
        if params[name].prior.kind == 'gaussian':
            sigma = params[name].prior.stddev.m
        ax.axvline(params_value, color='g',linewidth=2, label='injected')
        if sigma > 0:
            ax.axvline(params_value - sigma, color='g',linewidth=2, linestyle='--', label = r'prior $\pm \sigma$' )
            ax.axvline(params_value + sigma, color='g',linewidth=2, linestyle='--')
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s trials\ninjected = %.2f\nmedian = %.2f\nmean = %.2f +/- %.2f'%(trials, params_value, median, mean, uncert), loc=2, frameon=False)
        if sigma > 0:
            ax.set_xlim(params_value - 1.2*sigma, params_value + 1.2*sigma)
    else:
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s trials\nmedian = %.2f\nmean = %.2f +/- %.2f'%(trials, median, np.mean(data), uncert), loc=2, frameon=False)
    if name in ['llh','conv_llh','barlow_llh', 'chi2', 'mod_chi2', 'funny_llh']:
        p = chi2.fit(data,floc=0, scale=1)
        x = np.linspace(b[0], b[-1], 100)
        f = chi2.pdf(x, *p)
        ax.plot(x,f, color='r')
        a_text = AnchoredText(r'$\nu_\tau$ appearance'+'\n%s trials\nmedian = %.2f\nmean = %.2f +/- %.2f\nd.o.f. = %.1f'%(trials, median, np.mean(data), uncert, p[0]), loc=2, frameon=False)
    ax.add_artist(a_text)
    ax.set_xlabel(name)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
    plt.show()
    plt.gca().legend(loc='upper right',ncol=2, frameon=False,numpoints=1)
    plt.savefig('%s/%s.png'%(dir,name),facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('%s/%s.pdf'%(dir,name),facecolor=fig.get_facecolor(), edgecolor='none')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dir',metavar='dir',help='directory containg output json files', default='.') 
    parser.add_argument('--dist',action='store_true') 
    parser.add_argument('--corr',action='store_true') 
    parser.add_argument('-t', '--template-settings',
        metavar='configfile', default=None,
        action='append',
        help='''settings for the template generation''')
    args = parser.parse_args()

    
    params = ParamSet()
    if args.template_settings is not None:
        for ts in args.template_settings:
            cfg = parse_config(ts)
            for name,stage in cfg.items():
                if stage.has_key('params'):
                    params.update(stage['params'])

    total = 0
    hypos = []
    data = {}
    asimov = {}

    # get llh denominators for q for each seed
    for filename in os.listdir(args.dir):
        if filename.endswith('.json'):
            file = from_file(args.dir +'/'+filename)
            fit = file[0]
	    if file[0].has_key('llh'): metric = 'llh'
	    elif file[0].has_key('conv_llh'): metric = 'conv_llh'
	    elif file[0].has_key('chi2'): metric = 'chi2'
	    elif file[0].has_key('mod_chi2'): metric = 'mod_chi2'
	    elif file[0].has_key('barlow_llh'): metric = 'barlow_llh'
	    else: continue
            plot_dict = {}
            flag = False
            for key,val in fit.items():
                if key == 'warnflag':
                    flag = val
                elif key == 'all_metrics':
                    for m,v in val.items():
                        if 'llh' in m:
                    	    plot_dict[m] = - 2*np.array(v)
                        else:
                    	    plot_dict[m] = np.array(v)
                elif key == metric:
                        pass
                else:
                    plot_dict[key] = val[0]
            if flag:
                print 'skipping file %s'%filename
            elif 'asimov' in filename:
                name = filename.rstrip('.json')
                asimov[name] = {}
                for key,val in plot_dict.items():
                    asimov[name][key] = [[x] for x in val]
                
            else:
                total += 1
                for key,val in plot_dict.items():
                    if data.has_key(key):
                        data[key].append(val)
                    else:
                        data[key] = [val]
    for key,val in data.items():
        data[key] = np.array(val)

    if args.dist:
        if len(data) > 0:
            for key,val in data.items():
                dist(key,val,asimov, params,total,args.dir)
        elif len(asimov) > 0:
            for key in sorted(asimov[asimov.keys()[0]].keys()):
                dist(key,[],asimov,params,total,args.dir)

    if args.corr:
        import pandas as pd
        import seaborn as sns
        from scipy import stats
        sns.set(style="ticks", color_codes=True)
        # pairplot
        #columns = []
        columns = {}
        #columns['norm'] = ['atm_muon_scale', 'aeff_scale', 'nu_nc_norm', 'nutau_cc_norm']
        #columns['osc'] = ['sin2(2theta23)', 'theta13', 'deltam31', 'deltacp', 'nutau_cc_norm']
        columns['flux'] = list(set(data.keys()).intersection(['Barr_uphor_ratio', 'Barr_nu_nubar_ratio', 'delta_index', 'nue_numu_ratio', 'nutau_cc_norm']))
        columns['det'] =  list(set(data.keys()).intersection(['dom_eff', 'hole_ice', 'hole_ice_fwd', 'reco_cz_res', 'nutau_cc_norm']))
        columns['pull'] = list(set(data.keys()).intersection(['aeff_scale', 'atm_muon_scale', 'Barr_uphor_ratio', 'nutau_cc_norm']))
        #for key,val in data.items():
        #    if key not in ['llh','conv_llh','barlow_llh', 'chi2', 'mod_chi2']:
        #        columns.append(key)
        if data.has_key('theta23'):
            data['sin2(2theta23)'] = np.square(np.sin(2*data['theta23']* np.pi / 180.))
        if data.has_key('deltam31'):
            data['deltam31'] = 1000*data['deltam31']
        df = pd.DataFrame(data)

	def corrfunc(x, y, **kws):
	    r, p = stats.pearsonr(x, y)
	    ax = plt.gca()
	    ax.annotate("r = %.2f\np0 = %.1e"%(r,p),
			xy=(.1, .8), xycoords=ax.transAxes)
        def add_lines(*args, **kwargs):
	    ax = plt.gca()
            nominal = []
            mean = []
            median = []
            uncert = []
            prior = []
            for d in args:
                sigma = 0
                if d.name in params.names:
                    nom = params[d.name].value.m
                    if d.name == 'deltam31':
                        nom*=1000
                    if params[d.name].prior.kind == 'gaussian':
                        sigma = params[d.name].prior.stddev.m
                elif d.name == 'sin2(2theta23)':
                    nom = np.square(np.sin(2*params['theta23'].value.m* np.pi / 180.)) 
                m = np.mean(d)
                med = np.median(d)
                unc = np.std(d)/np.sqrt(len(d))
                nominal.append(nom)
                median.append(med)
                prior.append(sigma)
                uncert.append(unc)
                mean.append(m)
            for i in range(len(nominal)):
                if i == 0:
                    ax.axvline(nominal[0], color='r', linewidth=1)
                    if prior[0] > 0:
                        ax.axvline(nominal[0] - prior[0], color='r', linewidth=1, linestyle='--')
                        ax.axvline(nominal[0] + prior[0], color='r', linewidth=1, linestyle='--')
                        xmin, xmax = ax.get_xlim()
                        if xmin > nominal[0] - 1.2*prior[0]:
                            xmin = nominal[0] - 1.2*prior[0]
                        if xmax < nominal[0] + 1.2*prior[0]:
                            xmax = nominal[0] + 1.2*prior[0]
                        ax.set_xlim((xmin, xmax))
                    ax.axvline(median[0], color='b', linewidth=1)
                    ax.axvline(mean[0], color='g', linewidth=1)
                    ax.axvspan(mean[0]-uncert[0], mean[0]+uncert[0], alpha=0.2, color='g', linewidth=0)
                elif i == 1:
                    ax.axhline(nominal[1], color='r', linewidth=1)
                    if prior[1] > 0:
                        ax.axhline(nominal[1] - prior[1], color='r', linewidth=1, linestyle='--')
                        ax.axhline(nominal[1] + prior[1], color='r', linewidth=1, linestyle='--')
                        ymin, ymax = ax.get_ylim()
                        if ymin > nominal[1] - 1.2*prior[1]:
                            ymin = nominal[1] - 1.2*prior[1]
                        if ymax < nominal[1] + 1.2*prior[1]:
                            ymax = nominal[1] + 1.2*prior[1]
                        ax.set_ylim((ymin, ymax))
                    ax.axhline(median[1], color='b', linewidth=1)
                    ax.axhline(mean[1], color='g', linewidth=1)
                    ax.axhspan(mean[1]-uncert[1], mean[1]+uncert[1], alpha=0.2, color='g', linewidth=0)

        def add_text(x,**kwargs):
	    ax = plt.gca()
            m = np.mean(x)
            med = np.median(x)
            unc = np.std(x)/np.sqrt(len(x))
            if x.name in params.names:
                nom = params[x.name].value.m
                if x.name == 'deltam31':
                    nom*=1000
                if params[x.name].prior.kind == 'gaussian':
                    sigma = params[x.name].prior.stddev.m
            elif x.name == 'sin2(2theta23)':
                nom = np.square(np.sin(2*params['theta23'].value.m* np.pi / 180.)) 
            a_text = AnchoredText('injected = %.2f\nmedian = %.2f\nmean = %.2f +/- %.2f'%(nom, med, m, unc), loc=2, frameon=False)
            ax.add_artist(a_text)

        dot_size = max(min(7,int(1000/len(df.index))),1)
        for key, val in columns.items():
            cols = [col for col in val if col in data.keys()]
            if len(cols) > 0:
                g = sns.PairGrid(df[cols])
                g.map_upper(sns.regplot, scatter_kws={'s':dot_size})
                g.map_lower(sns.kdeplot, shade=True, cmap='Blues', shade_lowest=False)
                g.map_upper(corrfunc)
                g.map_lower(add_lines)
                g.map_diag(plt.hist)
                g.map_diag(add_lines)
                g.map_diag(add_text)
                g.savefig('%s/%s_corr.pdf'%(args.dir,key))
                g.savefig('%s/%s_corr.png'%(args.dir,key))
