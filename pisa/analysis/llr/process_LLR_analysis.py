#! /usr/bin/env python
#
# process_LLR_analysis.py - Process an analysis result of an LLR
# Analysis run, and plot all posterior parameter information if
# desired.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   31 March 2015
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import h5py

from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy.stats import norm

from pisa.utils.log import logging, set_verbosity
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers, Timer
from pisa.utils.fileio import from_file
from pisa.utils.params import select_hierarchy
from pisa.utils.hdf import from_hdf


# Functions for gaussian fit:
gauss = lambda x, amp, loc, width: amp*np.exp(-(x-loc)**2/(2.*width**2))


def do_gauss(xvals, yvals, **kwargs):
    f, c = curve_fit(gauss, xvals, yvals, **kwargs)
    return f, np.sqrt(np.diag(c))


def plot_gauss(xvals, fit, **kwargs):
    plt.plot(xvals, gauss(xvals, *fit), **kwargs)
    return


def get_data_frames(llh_file):
    """
    Loads data from stored hdf5 file into a data frame for each
    combination of 'pseudo_data | hypo'
    """

    fh = h5py.File(llh_file,'r')
    #fh = from_file(llh_file)
    data_frames = []
    for dFlag in ['data_NMH','data_IMH']:
        for hFlag in ['hypo_NMH','hypo_IMH']:

            keys = fh['trials'][dFlag][hFlag].keys()
            entries = len(fh['trials'][dFlag][hFlag][keys[0]])

            data = {key: np.array(fh['trials'][dFlag][hFlag][key]) for key in keys }
            data['pseudo_data'] = np.empty_like(data[keys[0]],dtype='|S16')
            data['pseudo_data'][:] = dFlag
            data['hypo'] = np.empty_like(data[keys[0]],dtype='|S16')
            data['hypo'][:] = hFlag

            df = DataFrame(data)
            data_frames.append(df)

    fh.close()

    return data_frames


def show_frame(df):
    pd.set_option('display.max_columns', len(df))
    pd.set_option('expand_frame_repr', False)
    pd.set_option('max_rows',20)
    logging.debug("df:\n%s"%df)

    return


def get_template_settings(llh_file):
    datafile = from_hdf(llh_file)
    return datafile['template_settings']['params']


def plot_llr_distributions(llr_nmh,llr_imh,nbins,plot_gauss=True,
                           imh_true=False):
    """Plots LLR distributions-expects llr_nmh and llr_imh to be type
    Series. Also plots vertical line signifying the mean of the
    hierarchy assumed to be given in nature, and the percentage of
    trials from the opposite hierarchy with LLR beyond the mean.
    """

    label_text = r'$\log ( \mathcal{L}(data: IH|IH) / \mathcal{L}( data: IH|NH) )$'
    llr_imh.hist(bins=nbins,histtype='step',lw=2,color='b',label=label_text)
    hist_vals_imh,bincen_imh = plot_error(llr_imh,nbins,fmt='.b',lw=2)
    if plot_gauss: plot_gauss_fit(llr_imh,hist_vals_imh,bincen_imh,color='b',lw=2)

    label_text = r'$\log ( \mathcal{L}(data: NH|IH) / \mathcal{L}(data: NH|NH) )$'
    llr_nmh.hist(bins=nbins,histtype='step',lw=2,color='r',label=label_text)
    hist_vals_nmh,bincen_nmh = plot_error(llr_nmh,nbins,fmt='.r',lw=2)
    if plot_gauss: plot_gauss_fit(llr_nmh,hist_vals_nmh,bincen_nmh,color='r',lw=2)

    if imh_true:
        mean_val = llr_nmh.mean()
        pvalue = 1.0 - float(np.sum(llr_imh > mean_val))/len(llr_imh)
    else:
        mean_val = llr_imh.mean()
        pvalue = float(np.sum(llr_nmh > mean_val))/len(llr_nmh)

    ymax = max(hist_vals_nmh) if imh_true else max(hist_vals_imh)
    bincen = bincen_imh if imh_true else bincen_nmh
    vline = plt.vlines(mean_val,1,ymax,colors='k',linewidth=2,
                       label=("pval = %.4f"%pvalue))

    sigma_1side = np.sqrt(2.0)*erfinv(1.0 - pvalue)
    sigma_2side = norm.isf(pvalue)
    print "  pvalue: %.4f"%pvalue
    print "  sigma 1 sided (erfinv): %.4f"%sigma_1side
    print "  sigma 2 sided (isf)   : %.4f"%sigma_2side

    if imh_true:
        plt.fill_betweenx(hist_vals_imh,bincen,x2=mean_val,where=bincen < mean_val,
                          alpha=0.5,hatch='xx')
    else:
        plt.fill_betweenx(hist_vals_nmh,bincen,x2=mean_val,where=bincen>mean_val,
                          alpha=0.5,hatch='xx')

    plt.ylabel('# Trials',fontsize='x-large')
    plt.xlabel('LLR value',fontsize='x-large')

    return


def plot_error(llr,nbins,**kwargs):
    """Given llr distribution Series, calculates the error bars and plots
    them """
    hist_vals,xbins = np.histogram(llr,bins=nbins)
    bincen = get_bin_centers(xbins)
    plt.errorbar(bincen,hist_vals,yerr=np.sqrt(hist_vals),**kwargs)
    return hist_vals,bincen


def plot_gauss_fit(llr,hist_vals,bincen,**kwargs):
    """Plots gaussian fit over the llr distributions."""

    fit_imh, cov = do_gauss(bincen,hist_vals,
                            p0=[np.max(hist_vals),llr.mean(),llr.std()])
    plot_gauss(bincen,fit_imh,**kwargs)

    return


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


def plot_posterior_params(frames, template_settings, plot_param_info=True,pbins=20,
                          save_fig=False,**kwargs):
    """Plot posterior parameter distributions, and related data"""

    good_columns = [col for col in frames[0].columns if col not in ['hypo','pseudo_data']]

    max_plots_per_fig = 4
    nfigs = len(good_columns)/max_plots_per_fig
    logging.info("len(good_cols): %d, nfigs: %d"%(len(good_columns),nfigs))

    figs = []
    fig_names = []
    colors = ['b','r','g','k','c','m']
    for frame in frames:
        data = frame['pseudo_data'][0]
        hypo = frame['hypo'][0]
        icol = 0
        for i in xrange(nfigs):

            fig = plt.figure(figsize=(10,10))
            figs.append(fig)
            fig.suptitle('Posteriors for %s, %s'%(data,hypo),fontsize='large')

            # determine how many columns to put on each figure:
            nsubplots = int(float(len(good_columns))/float(nfigs) + 0.5)
            for j in range(nsubplots):
                ax = plt.subplot(2,2,j+1)
                color = 'k' if plot_param_info else colors[icol%len(colors)]

                # Rescale the column data by 'scale'-so that these are
                # the values the optimizer actually sees, and makes it
                # easier to display on plots
                col_name = good_columns[icol]
                column = frame[col_name]

                if data == 'data_NMH': injected_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                else: injected_vals = select_hierarchy(template_settings,normal_hierarchy=False)

                if 'llh' not in col_name: scale = injected_vals[col_name]['scale']
                column = scale*column

                hist,xbins,patches = plt.hist(column,histtype='step',
                                              lw=2,color=color,bins=pbins)
                plt.title(good_columns[icol],fontsize='large')
                plt.grid(True)

                # Plot extra info about priors, injected val, mean, range, etc.
                if plot_param_info:
                    ylim = ax.get_ylim()
                    ymax = ylim[1]
                    std = column.std()
                    mean = column.mean()
                    key = column.name
                    logging.debug("Processing column: %s"%key)

                    # First, plot mean and std dev:
                    plot_mean_std(mean,std,ymax,ax)

                    # Next plot: injected val
                    if key != 'llh':
                        scale = injected_vals[key]['scale']
                        plot_injected_val(scale*injected_vals[key]['value'],ymax)

                        if key == 'theta23':
                            print "injected: ",scale*injected_vals[key]['value']
                            print "mean: ",mean


                        if hypo == 'hypo_NMH': fit_vals = select_hierarchy(template_settings,normal_hierarchy=True)
                        else: fit_vals = select_hierarchy(template_settings,normal_hierarchy=False)
                        # Now plot prior:
                        #if injected_vals[key]['prior'] is not None:
                        plot_prior(scale*fit_vals[key]['prior'],
                                   scale*fit_vals[key]['value'], ymax,ax)

                        # Finally, plot bound:
                        plot_bound(scale*fit_vals[key]['range'],ymax,ax)

                    ax.set_xlim([mean-5.0*std,mean+5.0*std])
                    ax.set_ylim([ylim[0],ymax*1.2])

                    plt.legend(loc='best',fontsize='large',framealpha=0.5)
                icol+=1

            if save_fig:
                filestem=args.llh_file.split('/')[-1]
                filename=(filestem.split('.')[0]+'_'+data+'_'+hypo+
                          '_'+str(i)+'.png')
                fig_names.append(filename)

    return figs,fig_names


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'llh_file',
        type=str,
        help="Processed LLH files to analyze"
    )
    parser.add_argument(
        '--nbins',
        type=int, default=50,
        help="Number of bins in x axis."
    )
    parser.add_argument(
        '--yscale',
        type=float, default=1.2,
        help='Factor to scale ymax by.'
    )
    parser.add_argument(
        '--no_gauss',
        action='store_true', default=False,
        help="Do not plot gaussian fit overlaid on distributions"
    )
    parser.add_argument(
        '--imh_true',
        action='store_true', default=False,
        help="Assumes IMH is the case nature gives us (rather than NMH)."
    )
    
    # Parameter posterior arguments
    parser.add_argument(
        '--no_params',
        action='store_true', default=False,
        help="Do not plot any posterior parameter information."
    )
    parser.add_argument(
        '--pbins',
        type=int, default=20,
        help="Number of bins in x axis for posteriors."
    )
    parser.add_argument(
        '--plot_param_extra',
        action='store_true', default=False,
        help='Plot extra parameter information on posteriors'
    )
    parser.add_argument(
        '-s', '--save_fig',
        action='store_true', default=False,
        help='Save all figures'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help='set verbosity level'
    )
    args = parser.parse_args()
    set_verbosity(args.verbose)
    
    df_dNMH_hNMH, df_dNMH_hIMH, df_dIMH_hNMH, df_dIMH_hIMH = get_data_frames(args.llh_file)
    template_settings = get_template_settings(args.llh_file)
    
    if args.verbose > 1:
        show_frame(df_dNMH_hNMH)
    
    # Apply this function to columns of data frame to get LLR:
    get_llh_ratio = lambda hIMH,hNMH: -(hIMH['llh'] - hNMH['llh'])
    llr_dNMH = get_llh_ratio(df_dNMH_hIMH, df_dNMH_hNMH)
    llr_dIMH = get_llh_ratio(df_dIMH_hIMH, df_dIMH_hNMH)
    
    logging.info("nmh mean: %f"%llr_dNMH.mean())
    logging.info("imh mean: %f"%llr_dIMH.mean())
    
    plot_llr_distributions(
        llr_dNMH, llr_dIMH, args.nbins,
        plot_gauss=(not args.no_gauss),
        imh_true=args.imh_true
    )
    
    # Configure plotting:
    ylim = plt.gca().get_ylim()
    ylim = [ylim[0], ylim[1]*args.yscale]
    plt.gca().set_ylim(ylim)
    plt.legend(loc='best', fontsize='medium')
    if args.save_fig:
        filestem=args.llh_file.split('/')[-1]
        filename=(filestem.split('.')[0]+'_LLR.png')
        logging.info('Saving to file: %s'%filename)
        plt.savefig(filename, dpi=150)
    
    # Next plot posterior parameter distributions and related plots:
    frames = [df_dNMH_hNMH, df_dNMH_hIMH, df_dIMH_hNMH, df_dIMH_hIMH]
    figs, fignames = plot_posterior_params(
        frames, template_settings, plot_param_info=args.plot_param_extra,
        save_fig=args.save_fig, pbins=args.pbins
    )
    
    if args.save_fig:
        for i,fig in enumerate(figs):
            fig.savefig(fignames[i], dpi=160)
    else:
        plt.show()
