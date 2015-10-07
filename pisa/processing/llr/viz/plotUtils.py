#
# plotUtils.py
# A set of plotting functions for handling the processed LLR data.
#

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import re
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy.stats import norm

from pisa.analysis.stats.LLHStatistics import get_binwise_llh
from pisa.utils.log import logging
from pisa.utils.params import select_hierarchy, get_free_params, get_param_values
from pisa.utils.utils import get_bin_centers


def validate_key(key):
    valid_keys = ['true_NH', 'true_IH']
    if key not in valid_keys:
        raise ValueError("Key: %s is unknown! Please use one of %s"
                         %(key, valid_keys))


# Functions for gaussian fit:
gauss = lambda x, amp, loc, width: amp*np.exp(-(x-loc)**2/(2.*width**2))

def do_gauss(xvals, yvals, **kwargs):
    f, c = curve_fit(gauss, xvals, yvals, **kwargs)
    return f, np.sqrt(np.diag(c))

def plot_gauss(xvals, fit, **kwargs):
    plt.plot(xvals, gauss(xvals, *fit), **kwargs)
    return


def plot_llr_distribution(llr_cur, tkey, bins, color='b', **kwargs):
    """Plots LLR distributions

    \Params:
      * llr_cur    - pandas Series object of llr for current value of MH
      * tkey       - must be in ['true_NH','true_IH']
      * bins       - bins for histogramming the llr distributions
      * color      - color of llr distribution
    """

    validate_key(tkey)
    llr_cur.hist(
        bins=bins, histtype='step', lw=2, color=color,**kwargs)

    hist_vals, bincen = plot_error(llr_cur, bins, fmt='.'+color, lw=2)

    fit_gauss = plot_gauss_fit(llr_cur, hist_vals, bincen, color=color, lw=2)
    logging.info("    mu = %.4f, sigma = %.4f"%(fit_gauss[1],fit_gauss[2]))
    #print "median: ",llr_cur.median()
    logging.info("    num_trials: %d",len(llr_cur))
    return hist_vals, bincen, fit_gauss


def plot_posterior_params(frames, template_settings, plot_llh=True,
                          plot_param_info=True, pbins=20, mctrue=False,
                          **kwargs):
    """Plot posterior parameter distributions, and related data"""

    good_columns = get_free_params(
        select_hierarchy(template_settings, normal_hierarchy=True)).keys()

    #good_columns = [col for col in frames[0].columns
    #                if col not in ['hypo','mctrue']]
    if plot_llh: good_columns.append('llh')
    print "good_columns: \n",good_columns

    max_plots_per_fig = 4
    nfigs = (len(good_columns)-1)/max_plots_per_fig + 1
    logging.info("len(good_cols): %d, nfigs: %d"%(len(good_columns),nfigs))

    figs = []
    fig_names = []
    colors = ['b','r','g','k','c','m']
    for frame in frames:
        ifig = 0
        true_key = frame['mctrue'][0]
        hypo_key = frame['hypo'][0]

        for icol,col_name in enumerate(good_columns):
            column = frame[col_name]
            # Create new fig if needed:
            if (icol%max_plots_per_fig) == 0:
                ifig += 1
                fig = plt.figure(figsize=(10,10))
                fig_names.append(true_key+"_"+hypo_key+"_"+str(ifig)+".png")
                figs.append(fig)
                fig.suptitle('Posteriors for %s, %s'%(true_key,hypo_key))
                             #fontsize='large')

            # Why is this not adding subplot?...
            subplot = (icol%max_plots_per_fig + 1)
            color = 'k' if plot_param_info else colors[icol%len(colors)]

            plot_column(
                true_key, hypo_key, subplot, column, template_settings,
                color,plot_param_info=plot_param_info,pbins=pbins,
                mctrue=mctrue)

    return figs,fig_names


def make_scatter_plot(frame, name, **kwargs):
    """
    Makes a scatter plot of column name in frame.
    """

    column_x = frame[name]
    if name == 'deltam31': column_x*=100.0

    params = []
    exclude = set(['hypo','llh','mctrue'])
    params = list(set(frame.columns).difference(exclude))

    figs = []
    # Plot correlation scatter plot for all other systematics
    for p in params:
        if p == name: continue
        column_y = frame[p]
        if p == 'deltam31': column_y*=100.0
        if 'theta' in p: column_y = np.rad2deg(column_y)

        with sns.axes_style("whitegrid"):
            sns.jointplot(column_x, column_y, size=8, color='b',
                          **kwargs)
            plt.tight_layout()
            figs.append(plt.gcf())

    return figs



def plot_error(llr,bins,**kwargs):
    """Given llr distribution Series, calculates the error bars and plots
    them """

    hist_vals,xbins = np.histogram(llr,bins=bins)
    bincen = get_bin_centers(xbins)
    plt.errorbar(bincen,hist_vals,yerr=np.sqrt(hist_vals),**kwargs)
    return hist_vals,bincen

def plot_gauss_fit(llr,hist_vals,bincen,**kwargs):
    """Plots gaussian fit over the llr distributions."""

    guess = [np.max(hist_vals), np.mean(llr), np.std(llr)]
    fit, cov = do_gauss(bincen,hist_vals, p0=guess)
    plot_gauss(bincen,fit,**kwargs)

    return fit


def plot_asimov_line(llh_dict, tkey, max_yval, **kwargs):
    """
    llh_dict  - dictionary of llh data
    tkey      - key of the true hierarchy (from asimov or pseudo data set)
    max_yval  - maximum yvalue for asimov line.
    """

    validate_key(tkey)

    asimov_data = llh_dict[tkey]['asimov_data']
    asimov_data_null = llh_dict[tkey]['asimov_data_null']

    llh_asimov = get_binwise_llh(asimov_data,asimov_data)
    llh_null = -llh_dict[tkey]['llh_null']['llh'][-1]

    logging.info("  >> llh_asimov: %.4f"%llh_asimov)
    logging.info("  >> llh null: %.4f"%llh_null)
    logging.info("Null hypothesis: ")
    for k,v in llh_dict[tkey]['llh_null'].items():
        logging.info("  >> %s: %f"%(k,v[-1]))

    asimov_llr = (llh_null - llh_asimov if 'true_N' in tkey
                  else llh_asimov - llh_null)
    vline = plt.vlines(
        asimov_llr, 0.1, max_yval ,colors='k',**kwargs)

    return asimov_llr

def plot_fill(llr_cur, tkey, asimov_llr, hist_vals, bincen, fit_gauss, **kwargs):
    """
    Plots fill between the asimov llr value and the histogram values
    which represent an LLR distribution.
    """
    validate_key(tkey)

    expr = 'bincen < asimov_llr' if 'true_N' in tkey else 'bincen > asimov_llr'

    plt.fill_betweenx(
        hist_vals, bincen, x2=asimov_llr, where=eval(expr), **kwargs)

    pvalue = (1.0 - float(np.sum(llr_cur > asimov_llr))/len(llr_cur)
              if 'true_N' in tkey else
              (1.0 - float(np.sum(llr_cur < asimov_llr))/len(llr_cur)))

    sigma_fit = np.fabs(asimov_llr - fit_gauss[1])/fit_gauss[2]
    #logging.info(
    #    "  For tkey: %s, gaussian computed mean (of alt MH): %.3f and sigma: %.3f"
    #    %(tkey,fit_gauss[1],fit_gauss[2]))
    pval_gauss = 1.0 - norm.cdf(sigma_fit)
    sigma_1side = np.sqrt(2.0)*erfinv(1.0 - pval_gauss)

    mctrue_row = [tkey,asimov_llr,llr_cur.mean(),pvalue,pval_gauss,sigma_fit,
                  sigma_1side]

    return mctrue_row



def plot_mean_std(mean_val, std_val, ymax,ax):
    """Plot the mean value as a vertical line """

    vline = plt.vlines(mean_val,1,ymax,colors='b',linewidth=3,label="mean")

    xfill = np.linspace(mean_val-std_val,mean_val+std_val,10)
    ax.fill_between(xfill,0.0,ymax*0.15,alpha=0.5,hatch='x',
                    facecolor='g')
    plt.plot(xfill,np.zeros_like(xfill),lw=3,color='g',alpha=0.8,label="st dev")

    return

def plot_injected_val(injected_val,ymax):

    vline = plt.vlines(injected_val,1,ymax,colors='r',linewidth=2,
                       alpha=1.0,label="injected")
    return

def plot_prior(prior,value,ymax,ax):

    if prior is None: return
    else:
        xfill = np.linspace(value-prior,value+prior,10)
        ax.fill_between(xfill,0.0,ymax*0.1,alpha=0.4,facecolor='k')
        plt.plot(xfill,np.zeros_like(xfill),lw=3,color='k',alpha=0.4,label='prior')

    return

def plot_bound(range_bound,ymax,ax):

    xfill = np.linspace(range_bound[0],range_bound[1],10)
    ax.fill_between(xfill,0.0,ymax*0.05,alpha=0.8,hatch='xx',facecolor='y')
    plt.plot(xfill,np.zeros_like(xfill),lw=3,color='y',alpha=0.8,label='bound')

    return

def get_col_info(col_name, tkey, hkey, template_settings, mctrue=False):

    # REMEMBER: Since the "true_NH" corresponds to "pseudo data IH",
    # the normal hierarchy input is opposite!
    # If mctrue --> This means that the LLR distributions are of the TH,
    #   rather than the AH, null hypothesis.

    #print "mctrue: ",mctrue

    if 'true_N' in tkey:
        mh = True if mctrue else False
        injected_vals = select_hierarchy(template_settings,
                                         normal_hierarchy=mh)
    else:
        mh = False if mctrue else True
        injected_vals = select_hierarchy(template_settings,
                                         normal_hierarchy=mh)
    if 'hypo_N' in hkey:
        mh = True if mctrue else False
        fit_vals = select_hierarchy(template_settings,
                                    normal_hierarchy=mh)
    else:
        mh = False if mctrue else True
        fit_vals = select_hierarchy(template_settings,
                                    normal_hierarchy=mh)

    value = injected_vals[col_name]['value']
    scale = injected_vals[col_name]['scale']
    prange = fit_vals[col_name]['range']

    #prior = fit_vals[col_name]['prior']
    if injected_vals[col_name]['prior']['kind'] == "gaussian":
        prior_val = injected_vals[col_name]['prior']["sigma"]
    else:
        prior_val = None

    return prior_val, value, prange, scale

def plot_column(tkey,hkey, subplot, column, template_settings, color,
                plot_param_info=True,pbins=20,mctrue=False):
    """Plot column information"""


    #
    # NOTE: Fix prior implementation here. If prior['kind'] == 'gaussian',
    # then get prior['sigma']!
    #
    # I don't think I need to check for theta, etc...
    #

    col_name = column.name
    if 'llh' not in col_name:
        prior, inj_value, prange, scale = get_col_info(
            col_name, tkey, hkey, template_settings,mctrue=mctrue)
        column = scale*column

    if bool(re.match('^theta',col_name)):
        column = np.rad2deg(column)
        if prior is not None: prior = np.rad2deg(prior)
        inj_value = np.rad2deg(inj_value)
        prange = np.rad2deg(prange)

    std = column.std()
    mean = column.mean()

    ax = plt.subplot(2,2,subplot)
    logging.debug("Processing column: %s"%col_name)

    hist,xbins,patches = plt.hist(column,histtype='step',lw=2,color=color,
                                  bins=pbins)
    plt.title(col_name)#,fontsize='large')
    plt.grid(True)

    # Plot extra info about priors, injected val, mean, range, etc.
    if plot_param_info:
        ylim = ax.get_ylim()
        ymax = ylim[1]

        # First, plot mean and std dev:
        plot_mean_std(mean,std,ymax,ax)

        # Next: plot injected_val, prior, and bound
        if col_name != 'llh':
            plot_injected_val(scale*inj_value,ymax)
            if prior is not None:
                plot_prior(scale*prior,scale*inj_value, ymax,ax)

            # Finally, plot bound:
            plot_bound(scale*prange,ymax,ax)

        if bool(re.match('^theta23',col_name)):
            ax.set_xlim([prange[0],prange[1]])
        else:
            ax.set_xlim([mean-5.0*std,mean+5.0*std])
        ax.set_ylim([ylim[0],ymax*1.2])

        plt.legend(loc='best',framealpha=0.5)#,fontsize='large')

    return
