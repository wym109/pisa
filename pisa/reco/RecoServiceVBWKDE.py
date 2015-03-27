#
# RecoServiceKDE.py
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   November 5, 2014
#


import sys

import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import interp1d
import h5py

from copy import deepcopy as copy

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
import pisa.utils.kde as kde
from pisa.utils.utils import get_bin_centers, get_bin_sizes, is_linear
from pisa.utils.log import logging, physics

class RecoServiceKDE(RecoServiceBase):
    """
    Creates reconstruction kernels using Kernel Density Estimation (KDE).
    
    NOTE: (tca) 22 Jan 2015 - For some reason V36 (noisy) is not
    being well characterized by the KDE automatically, like V15
    was. It would be nice to get to the bottom of why this is,
    but I haven't been able to do so yet. For now, it's recommended
    to either:
    
    1) Use the notebook in $PISA/pisa/i3utils to find the KDEs
    and convert them to kernelfiles
    
    or
    
    2) Use the parameterization method rather than KDEs.
    """
    
    def __init__(self, ebins, czbins, reco_kde_file=None, **kwargs):
        """
        Parameters needed to instantiate a reconstruction service with
        parametrizations:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        """
        self.kernels = None
        self.duplicate_nu_bar = False
        self.kde_dict = {}
        
        RecoServiceBase.__init__(self, ebins, czbins, reco_kde_file=reco_kde_file,
                                 **kwargs)
    
    
    def _get_reco_kernels(self, reco_kde_file=None,e_reco_scale=None,
                          cz_reco_scale=None,**kwargs):
        
        if reco_kde_file is None:
            logging.warn("Did not receive a reco_kde_file-> returning WITHOUT constructing kernels.")
            return
        
        if self.kernels is not None:
            return self.kernels
        
        # This is the step that is analagous to finding
        # parameterizations and putting the lambda functions into the
        # settings file.
        logging.info('Constructing KDEs from file: %s'%reco_kde_file)
        self.kde_dict = self.construct_1DKDE_dict(reco_kde_file,
                                                  remove_sim_downgoing=True)
        
        logging.info('Creating reconstruction kernels')
        self.kernels = self.calculate_kernels(kde_dict=self.kde_dict)
        
        return self.kernels
    
    
    def calculate_kernels(self, kde_dict=None, flipback=True):
        '''
        Calculates the 4D reco kernels in (true energy, true coszen,
        reco energy, reco coszen) for each flavour and interaction
        type.
        '''
        
        # get binning information
        evals, esizes = get_bin_centers(self.ebins), get_bin_sizes(self.ebins)
        czvals, czsizes = get_bin_centers(self.czbins), get_bin_sizes(self.czbins)
        n_e, n_cz = len(evals), len(czvals)
        
        # prepare for folding back at lower edge
        if flipback:
            if is_linear(self.czbins):
                czvals = np.append(czvals-(self.czbins[-1]-self.czbins[0]),
                                   czvals)
                czsizes = np.append(czsizes, czsizes)
            else:
                logging.warn("cos(zenith) bins have different "
                             "sizes! Unable to fold around edge "
                             "of histogram, will not do that.")
                flipback = False
        
        
        kernel_dict = {key:{'cc':None,'nc':None} for key in kde_dict.keys()}
        #self.kernels1D = {key:{'cc':None, 'nc':None} for key in kde_dict.keys()}
        
        flavours = kde_dict.keys()
        # If nu_bar are duplicates, remove from keys to loop over and
        # simply initialize them to their non-bar counterparts.
        if self.duplicate_nu_bar:
            flavours = [flav for flav in flavours if '_bar' not in flav]
        for flavour in flavours:
            for int_type in ['cc','nc']:
                logging.debug('Calculating KDE based reconstruction kernel for %s %s'
                              %(flavour, int_type))
                
                # create empty kernel
                kernel = np.zeros((n_e, n_cz, n_e, n_cz))
                #self.kernels1D[flavour][int_type] = np.zeros((n_e,n_cz))
                
                # loop over every bin in true (energy, coszen)
                for i in range(n_e):
                    energy = evals[i]
                    kvals = evals - energy
                    e_kern = kde_dict[flavour][int_type]['energy'][i].evaluate(kvals)
                    
                    e_kern_int = np.sum(e_kern*esizes)
                    
                    for j in range(n_cz):
                        offset = n_cz if flipback else 0
                        
                        kvals = czvals - czvals[j+offset]
                        cz_kern = kde_dict[flavour][int_type]['coszen'][i].evaluate(kvals)
                        
                        cz_kern_int = np.sum(cz_kern*czsizes)
                        
                        if flipback:
                            cz_kern = cz_kern[:len(czvals)/2][::-1] + cz_kern[len(czvals)/2:]
                        
                        kernel[i,j] = np.outer(e_kern, cz_kern)
                        # normalize correctly:
                        kernel[i,j]*=e_kern_int*cz_kern_int/np.sum(kernel[i,j])
                
                
                kernel_dict[flavour][int_type] = copy(kernel)
                if self.duplicate_nu_bar:
                    flav_bar = flavour+'_bar'
                    logging.debug('Duplicating reco kernel of %s/%s = %s/%s'
                                  %(flav_bar, int_type,flavour,int_type))
                    kernel_dict[flav_bar][int_type] = copy(kernel)
        kernel_dict['ebins'] = self.ebins
        kernel_dict['czbins'] = self.czbins
        
        return kernel_dict
    
    
    def construct_1DKDE_dict(self, kdefile, remove_sim_downgoing=True):
        """
        Constructs the 1D energy and coszen KDEs from the data in
        kdefile, and stores them in self.kde_dict. These resulting
        1D KDEs can be used to create the full 4D parameterized
        kernels in calculate_kernels()
        """
        
        try:
            kde_fh = h5py.File(find_resource(kdefile),'r')
        except IOError, e:
            logging.error("Unable to open KDE file %s"%kdefile)
            logging.error(e)
            sys.exit(1)
        
        flavours = ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']
        int_types = ['cc','nc']
        self.duplicate_nu_bar = self._check_duplicate_nu_bar(kde_fh)
        if self.duplicate_nu_bar:
            flavours = ['nue','numu','nutau']
        
        kde_dict = {}
        for flavour in flavours:
            flavour_dict = {}
            for int_type in int_types:
                logging.debug("Working on %s/%s kernels"%(flavour,int_type))
                true_energy = np.array(kde_fh[flavour+'/'+int_type+'/true_energy'])
                true_coszen = np.array(kde_fh[flavour+'/'+int_type+'/true_coszen'])
                reco_energy = np.array(kde_fh[flavour+'/'+int_type+'/reco_energy'])
                reco_coszen = np.array(kde_fh[flavour+'/'+int_type+'/reco_coszen'])
                
                # Cut on simulated downgoing:
                if remove_sim_downgoing:
                    logging.debug("Removing simulated downgoing events in KDE construction.")
                    cuts = np.alltrue(np.array([true_coszen < 0.0]),axis=0)
                    true_energy = true_energy[cuts]
                    true_coszen = true_coszen[cuts]
                    reco_energy = reco_energy[cuts]
                    reco_coszen = reco_coszen[cuts]
                
                flavour_dict[int_type] = self._get_KDEs(true_energy, true_coszen,
                                                        reco_energy, reco_coszen)
            kde_dict[flavour] = copy(flavour_dict)
            if self.duplicate_nu_bar:
                flavour += '_bar'
                logging.debug("   >Copying into %s kernels"%flavour)
                kde_dict[flavour]  = copy(flavour_dict)
        
        return kde_dict
    
    
    def _get_KDEs(self, e_true, cz_true, e_reco, cz_reco, tgt_num_events=300,
                  min_num_events=100, epsilon=1e-10, make_plots=False,
                  save_pdf=False):
        '''
        For the set of true/reco data, form the 1D energy/coszen
        KDE vs. energy at discrete energy bins of self.ebins. Due
        to decreasing statistics at higher energies, the bin size with
        which to characterize the KDE will be varied, so that a min.
        number of counts will be present in each determination of the
        KDE at the given bin.
        
        returns - dictionary of 1D energy/coszen KDEs in the format of
          'energy': [pdf_eres(E1),pdf_eres(E2),...,pdf_eres(En)]
          'coszen': [pdf_czres(E1),pdf_czres(E2),...,pdf_czres(En)]
        where self.ebins = [E1,E2,...,En]
        '''
        # TODO: make sure logging is done correctly
        # TODO: test!
        # TODO: pull in kde code; integrate findCI code in that same module, too
        assert ( np.min(np.diff(self.ebins )) > 0 ), logging.error("Energy bin edges must be monotonically increasing, but are not.")
        assert ( np.min(np.diff(self.czbins)) > 0 ), logging.error("cos-zenith bin edges must be monotonically increasing, but are not.")
        
        if not make_plots:
            save_pdf = False
            save_png = False
            pdf_fname = None
        
        if make_plots and save_pdf:
            pp = PdfPages(pdf_fname)
            
        ENERGY_RANGE = [0,501]
        
        # Histogram parameters
        N_HBINS = 15
        
        # Plot parameters
        TOP = 0.925
        BOTTOM = 0.05
        RIGHT = 0.97
        LEFT = 0.07
        HSPACE = 0.12
        LABELPAD = 0.058
        
        AXISBG = (0.5, 0.5, 0.5)
        
        VBWKDE_LAB = r'\mathrm{VBW KDE}'
        DIFFUS_PP  = dict(color=(0.0, 0.0, 0.0), linestyle='-',  marker=None, alpha=0.6, linewidth=4.0,
                          label=r'$' + VBWKDE_LAB + r'$')
        RUG_PP     = dict(color=(1.0, 1.0, 1.0), linewidth=0.7, alpha=0.3)
        HIST_PP    = dict(facecolor=(1,0.5,0.5), edgecolor=DARK_RED, histtype='stepfilled', alpha=0.7,
                          label=r'$\mathrm{Histogram}$')
        
        LEGFNTCOL  = (1,1,1)
        LEGFACECOL = (0.2,0.2,0.2)
        GRIDCOL    = (0.4, 0.4, 0.4)
        ebin_edges = np.array(self.ebins)
        left_ebin_edges = ebin_edges[0:-1]
        right_ebin_edges = npin_edges[1:]
        ebin_centers = (left_ebin_edges+right_ebin_edges)/2.0
        n_ebins = len(ebin_centers)

        czbin_edges = np.array(self.czbins)
        left_czbin_edges = czbin_edges[0:-1]
        right_czbin_edges = czbin_edges[1:]
        czbin_centers = (left_czbin_edges+right_czbin_edges)/2.0
        n_czbins = len(czbin_centers)

        n_events = len(e_true)

        if min_num_events > n_events:
            min_num_events = n_events
        if tgt_num_events > n_events:
            tgt_num_events = n_events
        
        # Object with which to store the 4D kernels: np 4D array
        kernels = np.zeros((n_ebins, n_czbins, n_ebins, n_czbins))
        
        # Object with which to store the 2D "aggregate_map": the total number of
        # events reconstructed into a given (E, CZ) bin, used for sanity checks
        aggregate_map = np.zeros((n_ebins, n_czbins))
        for ebin_n in range(n_ebins):
            ebin_min = left_ebin_edges[ebin_n]
            ebin_max = right_ebin_edges[ebin_n]
            ebin_mid = (ebin_min+ebin_max)/2.0
            ebin_wid = ebin_max-ebin_min
            
            logging.debug('  processing true-energy bin_n=' + str(ebin_n) + ' of ' +
                    str(n_ebins-1) + ', E_{nu,true} in [' +
                    numFmt(ebin_min,threeSpacing=0) + ', ' +
                    numFmt(ebin_max,threeSpacing=0) + '] ...')
            
            # Absolute distance from these events' re-centered reco energies to the
            # center of this energy bin
            abs_enu_dist = np.abs(e_true - ebin_mid)
            
            # Sort in ascending-distance order
            abs_enu_dist.sort()
            
            # Grab the distance the number-"tgt_num_events" event is from the bin
            # center
            tgt_thresh_enu_dist = abs_enu_dist.iloc[tgt_num_events-1]
            
            # Grab the distance the number-"min_num_events" event is from the bin
            # center
            min_thresh_enu_dist = abs_enu_dist.iloc[min_num_events-1]
            
            # Make threshold distance (which is half the total width) no MORE than
            # 4x the true-energy-bin width and no LESS than the actual distance
            # from this midpoint to the bin edges (i.e., the bin should at least be
            # as wide as the stated bin width). But if there are fewer than
            # min_num_events, then take as wide a bin as necessary to fill that
            # many events.
            thresh_enu_dist = max( min( max(tgt_thresh_enu_dist,ebin_wid/2), 4*ebin_wid ), min_thresh_enu_dist )
            
            # Grab all events within the threshold distance
            in_ebin_ind = np.where(abs_enu_dist <= thresh_enu_dist)[0]
            n_in_bin = len(in_ebin_ind)
            
            # Record lowest/highest energies that are *actually* used in
            # resolution computations for this bin; not letting the bin edge
            # fall outside the binned region
            #
            # TODO: seems like we should allow ourselves to pull true-energies even
            # OUTSIDE the binned region... so eliminate the max(min({e,cz}bins)
            # bits here?
            actual_left_ebin_edge = max(min(ebin_edges), ebin_mid-thresh_enu_dist)
            actual_right_ebin_edge = min(max(ebin_edges), ebin_mid+thresh_enu_dist)
            
            # Extract just the neutrino-energy/coszen error columns' values for succinctness
            enu_err = e_reco[in_ebin_ind] - e_true[in_ebin_ind]
            cz_err = cz_reco[in_ebin_ind] - cz_true[in_ebin_ind]
            
            #========================================================================
            # Neutrino energy resolution
            #========================================================================
            dmin = min(enu_err)
            dmax = max(enu_err)
            drange = dmax-dmin
            
            hbin_max = ebin_centers[ebin_n]
            hbins = np.linspace(dmin-0.02*drange, dmax+0.02*drange, N_HBINS*np.round(drange/hbin_max))
            bw = hbins[1] - hbins[0]
            
            hbin_centers = (hbins[0:-1]+hbins[1:])/2.0 
            
            min_val = min(ENERGY_RANGE[0]-ebin_mid*1.5, dmin-drange*0.5)
            max_val = max((ebin_overall_roi_max-ebin_mid)*1.5, dmax+drange*0.5)
            err_E_nu_calc_x_lims = np.array([min_val, max_val])
            err_E_nu_calc_N = int(max(2**10, 2**np.ceil(np.log2((err_E_nu_calc_x_lims[1]-err_E_nu_calc_x_lims[0])/(min(np.diff(ebin_edges))/2.0)))))
            logging.debug(' Nevts=' + str(n_in_bin) +
                    ' taken from [' + str(ebin_mid-thresh_enu_dist) + ', ' + str(ebin_mid+thresh_enu_dist) + ']' +
                    ', KDE x_lims=' + str(err_E_nu_calc_x_lims) + ', KDE_N: ' + str(err_E_nu_calc_N))
            
            # Compute variable-bandwidth KDEs
            enu_vbwkde_bw, enu_vbwkde_mesh, enu_vbwkde_pdf = kde.vbw_kde(
                data           = enu_err,
                overfit_factor = OVERFIT_FACTOR,
                MIN            = err_E_nu_calc_x_lims[0],
                MAX            = err_E_nu_calc_x_lims[1],
                N              = err_E_nu_calc_N
            )
            
            if np.min(enu_vbwkde_pdf) < 0:
                # Only issue warning if the most-negative value is negative beyond
                # specified acceptable-numerical-precision threshold (epsilon)
                if np.min(enu_vbwkde_pdf) <= -epsilon:
                    warnings.warn("np.min(enu_vbwkde_pdf) < 0: Minimum value is " +
                                  str(np.min(enu_vbwkde_pdf)) +
                                  "; forcing all negative values to 0.")
                # Otherwise, just quietly clip the negative values at 0
                enu_vbwkde_pdf = np.clip(a=enu_vbwkde_pdf, a_min=0, a_max=np.inf)
            
            assert ( np.min(enu_vbwkde_pdf) >= 0 ), logging.error(str(np.min(enu_vbwkde_pdf)))
            
            # Re-center distribution on the center of the energy bin for which
            # errors were computed
            offset_enu_vbwkde_mesh = enu_vbwkde_mesh+ebin_mid
            offset_enu_vbwkde_pdf = enu_vbwkde_pdf
             
            # Get reference area under the PDF, for checking after interpolated
            # values are added.
            #
            # NOTE There should be NO normalization because any events lost due to
            # cuting off tails outside the binned region are actually going to be
            # lost, and so should penalize the total area.
            int_val0 = np.trapz(y=offset_enu_vbwkde_pdf, x=offset_enu_vbwkde_mesh)
            
            # Create linear interpolator for the PDF
            interp = interp1d(x=offset_enu_vbwkde_mesh, y=offset_enu_vbwkde_pdf,
                              kind='linear',
                              copy=True,
                              bounds_error=True,
                              fill_value=np.nan,
                              assume_sorted=True) 
            
            # Insert all bin edges' exact locations into the mesh (For accurate
            # accounting of area in each bin, must include values out to bin
            # edges!)
            edge_locs = [ be for be in np.concatenate((left_ebin_edges,right_ebin_edges)) if not(be in offset_enu_vbwkde_mesh) ]
            edge_locs.sort()
            edge_pdfs = interp(edge_locs)
            insert_ind = np.searchsorted(offset_enu_vbwkde_mesh, edge_locs)
            offset_enu_vbwkde_mesh = np.insert(offset_enu_vbwkde_mesh, insert_ind, edge_locs)
            offset_enu_vbwkde_pdf = np.insert(offset_enu_vbwkde_pdf, insert_ind, edge_pdfs)
            
            int_val = np.trapz(y=offset_enu_vbwkde_pdf, x=offset_enu_vbwkde_mesh)
            
            assert ( np.abs(int_val - int_val0) < epsilon )
            
            # Chop off distribution at extrema of energy bins
            valid_ind = np.where((offset_enu_vbwkde_mesh >= ebin_overall_roi_min) & (offset_enu_vbwkde_mesh <= ebin_overall_roi_max))[0]
            offset_enu_vbwkde_mesh = offset_enu_vbwkde_mesh[valid_ind]
            offset_enu_vbwkde_pdf = offset_enu_vbwkde_pdf[valid_ind]
            
            # Check that there are no negative density values (after inserts)
            assert ( np.min(offset_enu_vbwkde_pdf) > 0-epsilon ), logging.error(str(np.min(offset_enu_vbwkde_pdf)))
            
            # Record the integrated area after removing outside binned range
            tot_ebin_area0 = np.trapz(y=offset_enu_vbwkde_pdf, x=offset_enu_vbwkde_mesh)
            
            # Check that it integrates to <= 1, sanity check
            assert ( tot_ebin_area0 < 1+epsilon ), logging.error(str(int_val))
            
            # Identify indices encapsulating the defined energy bins' ranges, and
            # find the area of each bin
            lbinds = np.searchsorted(offset_enu_vbwkde_mesh, left_ebin_edges)
            rbinds = np.searchsorted(offset_enu_vbwkde_mesh, right_ebin_edges)
            bininds = zip(lbinds, rbinds)
            ebin_areas = [ np.trapz(y=offset_enu_vbwkde_pdf[l:r+1], x=offset_enu_vbwkde_mesh[l:r+1]) for (l,r) in bininds ]
            
            # Check that no bins have negative areas
            assert ( np.min(ebin_areas) >= 0 )
            
            # Sum the individual bins' areas
            tot_ebin_area = np.sum(ebin_areas)
            
            # Check that this total of all the bins is equal to the total area
            # under the curve (i.e., check there is no overlap between or gaps
            # between bins)
            assert ( np.abs(tot_ebin_area-tot_ebin_area0) < epsilon ), logging.error('tot_ebin_area=' + str(tot_ebin_area) + ' should equal tot_ebin_area0=' + str(tot_ebin_area0))
            
            if make_plots:
                fig1 = plt.figure(fignum_offset+1, figsize=(8,10), dpi=90)
                fig1.clf()
                ax1 = fig1.add_subplot(211, axisbg=AXISBG)
                
                fci = kde.FindCI(x=enu_diffus_mesh, y=enu_diffus_pdf)
                for conf in np.logspace(np.log10(0.999),np.log10(0.95),50):
                    try:
                        lb, ub, yopt, r = fci.findCI_lin(conf=conf)
                    except:
                        pass
                    else:
                        break
                
                # TODO: is this a good range to set?
                xlims = (min(-ebin_mid*1.5, lb), max(min(ub, 6*ebin_mid),2*ebin_mid))
                
                lw = 2
                hvals, hbins, hpatches = ax1.hist(enu_err,
                                                  bins=hbins,
                                                  normed=True,
                                                  linewidth=lw,
                                                  **HIST_PP)
                
                ax1.plot(enu_diffus_mesh, enu_diffus_pdf, **DIFFUS_PP)
                
                axlims = ax1.axis('tight')
                ax1.set_xlim(xlims)
                ymax = axlims[3]*1.05
                ax1.set_ylim(0, ymax)
                
                # Grey-out region outside binned region, so it's clear what part of
                # tail(s) will be thrown away
                width = -ebin_mid+ebin_overall_roi_min-xlims[0]
                unbinned_region_tex = r'$\mathrm{Unbinned}$'
                if width > 0:
                    ax1.add_patch(
                        Rectangle(
                            (xlims[0],0), width, ymax, #zorder=-1,
                            alpha=0.30, facecolor=(0.0 ,0.0, 0.0),
                            fill=True,
                            ec='none'
                        )
                    )
                    ax1.text(xlims[0]+(xlims[1]-xlims[0])/40., ymax/10.,
                             unbinned_region_tex,  fontsize=14, ha='left',
                             va='bottom', rotation=90, color='k')
                
                width = xlims[1] - (ebin_overall_roi_max-ebin_mid)
                if width > 0:
                    ax1.add_patch(
                        Rectangle(
                            (xlims[1]-width,0), width, ymax, #zorder=-1,
                            alpha=0.30, facecolor=(0.0 ,0.0, 0.0), fill=True,
                            ec='none'
                        )
                    )
                    ax1.text(xlims[1]-(xlims[1]-xlims[0])/40., ymax/10.,
                             unbinned_region_tex,  fontsize=14, ha='right',
                             va='bottom', rotation=90, color='k')
                
                ylim = ax1.get_ylim()
                dy = ylim[1] - ylim[0]
                ruglines = rugplot(enu_err, y0=ylim[1], dy=-dy/40., ax=ax1, **RUG_PP)
                ruglines[-1].set_label(r'$\mathrm{Rug\,plot,\,MC\,evts}$')
                
                leg_title_tex = r'\mathrm{Normalized}\,E_\nu\mathrm{-err.\,distr.}'
                errinfo = {'tex': r'E_{\nu,\mathrm{reco}}-E_{\nu,\mathrm{true}}',
                           'units': r'(\mathrm{GeV})'}
                
                x1lab = ax1.set_xlabel(
                    r'$' + r'\,'.join([errinfo['tex'], errinfo['units']]) + r'$',
                    labelpad=LABELPAD
                )
                ax1.xaxis.set_label_coords(0.9, -LABELPAD)
                leg = ax1.legend(loc='upper right',
                                 title=r'$' + leg_title_tex + r'$',
                                 frameon=True, framealpha=0.8, fancybox=True,
                                 bbox_to_anchor=[1,0.975])
                ax1.xaxis.grid(color=GRIDCOL)
                ax1.yaxis.grid(color=GRIDCOL)
                leg.get_title().set_fontsize(16)
                leg.get_title().set_color(LEGFNTCOL)
                [ t.set_color(LEGFNTCOL) for t in leg.get_texts() ]
                frame = leg.get_frame()
                frame.set_facecolor(LEGFACECOL)
                frame.set_edgecolor(None)
            
            #========================================================================
            # Neutrino coszen resolution
            #========================================================================
            dmin = min(cz_err)
            dmax = max(cz_err)
            drange = dmax-dmin
            
            if make_plots:
                hbins = np.linspace(dmin-0.02*drange, dmax+0.02*drange, N_HBINS*5)
                bw = hbins[1] - hbins[0]
                
                hbin_centers = (hbins[0:-1]+hbins[1:])/2.0 
            
            if make_plots and comp_kde:
                err_cz_calc_x_lims = np.array([-2, 2])
                x_kde = np.linspace(err_cz_calc_x_lims[0], err_cz_calc_x_lims[1], 1000)
                
                # Compute KDE and evaluate at x_kde
                kde_cz = gaussian_kde(cz_err, bw_method='silverman')
                pdf_cz = kde_cz.evaluate(x_kde)
                
            #
            # Compute variable-bandwidth KDEs for coszen
            #
            
            # VBW KDE
            # 
            # NOTE the limits are 1 less than / 1 greater than the limits that the
            # error will actually take on, so as to allow for any smooth roll-off
            # at edges of data. The calculation of areas below captures all of the
            # area, though, by reflecting bins defined in [-1, 1] about the points
            # -1 and 1, thereby capturing any densities in the range [-3, +3]. This
            # is not necessarily accurate, but it's better than throwing that info
            # out entirely.
            #
            # NOTE also that since reco events as of now are only in range -1 to 0,
            # though, that there are "gaps" in the capture range, but this is due
            # to densities being in the upper-hemisphere which we are intentionally
            # ignoring, rather than the code here not taking them into account.
            # Normalization is based upon *all* events, whether or not they fall
            # within a bin specified above.
            
            # Number of points in the mesh used for VBWKDE; must be large enough to
            # capture fast changes in the data but the larger the number, the
            # longer it takes to compute the densities at all the points
            N_cz_mesh = 2**9
            
            # Data range for VBWKDE to consider
            cz_gaus_kde_min = -3
            cz_gaus_kde_max = +2
            
            cz_gaus_kde_failed = False
            previous_fail = False
            for n in xrange(3):
                try:
                    cz_diffus_bw, cz_diffus_mesh, cz_diffus_pdf = kde.vbw_kde(
                        data           = cz_err,
                        overfit_factor = OVERFIT_FACTOR,
                        MIN            = cz_gaus_kde_min,
                        MAX            = cz_gaus_kde_max,
                        N              = N_cz_mesh
                    )
                    #print cz_diffus_bw
                except:
                    cz_gaus_kde_failed = True
                    if n == 0:
                        logging.debug('(cz diffus KDE ')
                    logging.debug('fail, ')
                    # If failure occurred in vbw_kde, expand the data range it
                    # takes into account; this usually helps
                    cz_gaus_kde_min -= 1
                    cz_gaus_kde_max += 1
                else:
                    if cz_gaus_kde_failed:
                        previous_fail = True
                        logging.debug('success!')
                    cz_gaus_kde_failed = False
                finally:
                    if previous_fail:
                        logging.debug(')')
                    previous_fail = False
                    if not cz_gaus_kde_failed:
                        break
            
            if cz_gaus_kde_failed:
                logging.error('Failed to fit diffus. KDE!')
                continue
            
            if make_plots and comp_diffusfbw:
                cz_diffus_bw0, cz_diffus_mesh0, cz_diffus_pdf0 = kde.kde(
                    data           = cz_err,
                    overfit_factor = 1.0,
                    MIN            = cz_gaus_kde_min,
                    MAX            = cz_gaus_kde_max,
                    N              = N_cz_mesh
                )
            
            if np.min(cz_diffus_pdf) < 0:
                warnings.warn("np.min(cz_diffus_pdf) < 0: Minimum value is " + str(np.min(cz_diffus_pdf)) + "; forcing all negative values to 0.")
                np.clip(a=cz_diffus_mesh, a_min=0, a_max=np.inf)
            
            assert ( np.min(cz_diffus_pdf) >= -epsilon ), logging.error(str(np.min(cz_diffus_pdf)))
            
            for czbin_n in range(n_czbins):
                czbin_min = left_czbin_edges[czbin_n]
                czbin_max = right_czbin_edges[czbin_n]
                czbin_mid = czbin_centers[czbin_n]
                
                # Re-center distribution on the center of the current coszen bin
                offset_cz_diffus_mesh = cz_diffus_mesh + czbin_mid
                 
                # Create interpolation object, used to fill in bin edge values
                interp = interp1d(x=offset_cz_diffus_mesh, y=cz_diffus_pdf,
                                  kind='linear',
                                  copy=True,
                                  bounds_error=False,
                                  fill_value=0,
                                  assume_sorted=True)
                
                # Figure out where all bin edges lie in this re-centered
                # distribution (some bins may be repeated since bins in [-1,0] and
                # err in [-2,1]:
                #
                # 1. Find limits of mesh values..
                mmin = offset_cz_diffus_mesh[0]
                mmax = offset_cz_diffus_mesh[-1]
                
                # 2. Map all bin edges into the full mesh-value range, reflecting
                # about -1 and +1. If the reflected edge is outside the mesh
                # range, use the exceeded limit of the mesh range as the bin edge
                # instead.
                #
                # Logically, this maps every bin edge {i} to 3 new edges, indexed
                # new_edges[i][{0,1,2}]. Bins are formed by adjacent indices and
                # same-subindices, so what started as, e.g., bin 3 now is described
                # by (left, right) edges at
                #   (new_edges[3][0], new_edges[4][0]),
                #   (new_edges[3][1], new_edges[4][1]), and
                #   (new_edges[3][2], new_edges[4][2]).
                
                # NOTE / TODO: It's tempting to dynamically set the number of
                # reflections to minimize computation time, but I think it breaks
                # the code. Just set to a reasonably large number for now and
                # accept the performance penalty. ALSO: if you change the parity
                # of the number of reflections, the code below that has either
                # (wrap_n % 2 == 0) or (wrap_n+1 % 2 == 0) must be swapped!!!
                n_left_reflections = 4
                n_right_reflections = 4
                #n_left_reflections = int(np.ceil((np.min(offset_cz_diffus_mesh) - -1) / (-2.)))
                #n_right_reflections = int(np.ceil((np.max(offset_cz_diffus_mesh) - +1) / (2.)))
                
                new_czbin_edges = []
                for edge in czbin_edges:
                    edges_refl_left = []
                    for n in xrange(n_left_reflections):
                        edge_refl_left = reflect(edge, -1-(2*n))
                        if edge_refl_left < mmin: edge_refl_left = mmin
                        edges_refl_left.append( edge_refl_left )
                    edges_refl_right = []
                    for n in xrange(n_right_reflections):
                        edge_refl_right = reflect(edge, +1+(2*n))
                        if edge_refl_right > mmax: edge_refl_right = mmax
                        edges_refl_right.append(edge_refl_right)
                    # Include all left-reflected versions of this bin edge, in
                    # increasing-x order + this bin edge + right-reflected versions
                    # of this bin edge
                    new_czbin_edges.append( edges_refl_left[::-1] + [edge] + edges_refl_right )
                
                # Record all unique bin edges
                edge_locs = set()
                [ edge_locs.update(edges) for edges in new_czbin_edges ]
                
                # Throw away bin edges that are already in the mesh
                [ edge_locs.remove(edge) for edge in list(edge_locs) if edge in offset_cz_diffus_mesh ]
                
                # Make into sorted list
                edge_locs = sorted(edge_locs)
                
                # Record the total area under the curve
                int_val0 = np.trapz(y=cz_diffus_pdf, x=offset_cz_diffus_mesh)
                
                # Insert the missing bin edge locations & pdf-values into
                # the mesh & pdf, respectively
                edge_pdfs = interp(edge_locs)
                insert_ind = np.searchsorted(offset_cz_diffus_mesh, edge_locs)
                offset_cz_diffus_mesh = np.insert(offset_cz_diffus_mesh, insert_ind, edge_locs)
                offset_cz_diffus_pdf = np.insert(cz_diffus_pdf, insert_ind, edge_pdfs)
                assert ( np.min(offset_cz_diffus_pdf) > 0-epsilon )
                
                # Check that this total of all the bins is equal to the total area
                # under the curve (i.e., check there is no overlap between or gaps
                # between bins)
                int_val = np.trapz(y=offset_cz_diffus_pdf, x=offset_cz_diffus_mesh)
                assert ( np.abs(int_val-1) < epsilon )
                
                # Renormalize if it's not exactly 1
                if int_val != 1.0:
                    offset_cz_diffus_pdf = offset_cz_diffus_pdf / int_val
                
                # Add up the area in the bin and areas that are "reflected" into
                # this bin
                new_czbin_edges = np.array(new_czbin_edges)
                czbin_areas = np.zeros(np.shape(new_czbin_edges)[0]-1)
                for wrap_n in range(np.shape(new_czbin_edges)[1]):
                    bin_edge_inds = np.searchsorted(offset_cz_diffus_mesh, new_czbin_edges[:,wrap_n])
                    lbinds = bin_edge_inds[0:-1]
                    rbinds = bin_edge_inds[1:]
                    # Make sure indices that appear first are less than indices
                    # that appear second in a pair of bin indices
                    if (wrap_n+1) % 2 == 0:
                        bininds = zip(rbinds, lbinds)
                    else:
                        bininds = zip(lbinds, rbinds)
                    tmp_areas = []
                    for (this_bin_left_edge, this_bin_right_edge) in bininds:
                        if this_bin_left_edge == this_bin_right_edge:
                            tmp_areas.append(0)
                            continue
                        this_bin_area = np.array(
                            np.trapz(
                                y=offset_cz_diffus_pdf[this_bin_left_edge:this_bin_right_edge+1],
                                x=offset_cz_diffus_mesh[this_bin_left_edge:this_bin_right_edge+1]
                            )
                        )
                        tmp_areas.append(this_bin_area)
                    czbin_areas += np.array(tmp_areas)
                
                assert ( np.min(czbin_areas) > -epsilon )
                
                tot_czbin_area = np.sum(czbin_areas)
                assert ( tot_czbin_area < int_val + epsilon )
                
                kernels[ebin_n, czbin_n] = np.outer(ebin_areas, czbin_areas)
                assert ( (np.sum(kernels[ebin_n, czbin_n]) - tot_ebin_area*tot_czbin_area) < epsilon )
                
                aggregate_map += kernels[ebin_n, czbin_n]
                
            if make_plots:
                ax2 = fig1.add_subplot(212, axisbg=AXISBG)
                hvals, hbins, hpatches = ax2.hist(cz_err,
                                                  bins=hbins,
                                                  normed=True,
                                                  lw=2,
                                                  **HIST_PP)
                
                ax2.plot(cz_diffus_mesh, cz_diffus_pdf, **DIFFUS_PP)
                
                # TODO: set xlims to something consistent vs. dynamic for
                # easier comparisons?
                fci = kde.FindCI(x=cz_diffus_mesh, y=cz_diffus_pdf)
                for conf in np.logspace(np.log10(0.999),np.log10(0.95),50):
                    try:
                        lb, ub, yopt, r = fci.findCI_lin(conf=conf)
                    except:
                        pass
                    else:
                        break
                
                axlims = ax2.axis('tight')
                ax2.set_xlim(lb, ub)
                ax2.set_ylim(0,axlims[3]*1.05)
                
                ylim = ax2.get_ylim()
                dy = ylim[1] - ylim[0]
                ruglines = rugplot(cz_err, y0=ylim[1], dy=-dy/40., ax=ax2, **RUG_PP)
                ruglines[-1].set_label(r'$\mathrm{Rug\,plot,\,MC\,evts}$')
                
                errinfo = {'tex': r'\cos\vartheta_{\mathrm{track,reco}}-\cos\vartheta_{\mu,\mathrm{true}}',
                           'units': r''}
                x2lab = ax2.set_xlabel(
                    r'$' + r'\,'.join([errinfo['tex'], errinfo['units']]) + r'$',
                    labelpad=LABELPAD
                )
                ax2.xaxis.set_label_coords(0.9, -LABELPAD)
                ax2.xaxis.grid(color=GRIDCOL)
                ax2.yaxis.grid(color=GRIDCOL)
                
                leg_title_tex = r'$\mathrm{Normalized}\,\cos\vartheta\mathrm{-err.\,distr.}$'
                leg = ax2.legend(loc='upper right', title=leg_title_tex,
                                 frameon=True, framealpha=0.8, fancybox=True,
                                 bbox_to_anchor=[1,0.975])
                leg.get_title().set_fontsize(16)
                leg.get_title().set_color(LEGFNTCOL)
                [ t.set_color(LEGFNTCOL) for t in leg.get_texts() ]
                frame = leg.get_frame()
                frame.set_facecolor(LEGFACECOL)
                frame.set_edgecolor(None)
                
                stt = r'\mathrm{Resolutions}'
                if suptitle_tex and len(suptitle_tex) > 0:
                        stt = stt + r',\,' + suptitle_tex
                
                actual_bin_tex = ''
                if (actual_left_ebin_edge != ebin_min) or (actual_right_ebin_edge != ebin_max):
                    actual_bin_tex = r'E_{\nu,\mathrm{true}}\in [' + \
                            numFmt(actual_left_ebin_edge,keepAllSigFigs=1) + r',\,' + \
                            numFmt(actual_right_ebin_edge,keepAllSigFigs=1) + r'] \mapsto '
                
                stt = r'$' + stt + r'$' + '\n' + \
                        r'$' + actual_bin_tex + r'\mathrm{Bin}_{' + numFmt(ebin_n) + r'}\equiv E_{\nu,\mathrm{true}}\in [' + numFmt(ebin_min,keepAllSigFigs=1) + \
                        r',\,' + numFmt(ebin_max,keepAllSigFigs=1) + r']\,\mathrm{GeV}' + \
                        r',\,N_\mathrm{MC\,evts}=' + numFmt(true_enu_bin_num_allevents) + r'$'
                
                fig1.subplots_adjust(top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT, hspace=HSPACE)
                suptitle = fig1.suptitle(stt)
                suptitle.set_fontsize(16)
                suptitle.set_position((0.5,0.99))
                
                if save_pdf:
                    fig1.savefig(pp, format='pdf')
                
            logging.debug(' done.')
        
        check_areas = np.array([ [np.sum(kernels[en,czn]) for czn in range(n_czbins)] for en in range(n_ebins) ])
        
        assert ( np.max(check_areas) < 1 + epsilon )
        assert ( np.min(check_areas) > 0 - epsilon )
        
        if make_plots:
            fig = plt.figure(fignum_offset+2, figsize=(8,10), dpi=90)
            fig.clf()
            ax = fig.add_subplot(111)
            X, Y = np.meshgrid(range(n_czbins), range(n_ebins))
            #cm = mpl.cm.Oranges_r
            cm = mpl.cm.Paired_r
            cm.set_over((1,1,1), 1)
            cm.set_under((0,0,0), 1)
            plt.pcolor(X, Y, check_areas, vmin=0+epsilon, vmax=1.0, shading='faceted', cmap=cm)
            #plt.yscale('log')
            plt.colorbar(ticks=np.linspace(0,1,11))
            ax.grid(0)
            ax.axis('tight')
            ax.set_xlabel(r'$\cos\vartheta\mathrm{\,bin\,num.}$')
            ax.set_ylabel(r'$E_\nu\mathrm{\,bin\,num.}$')
            ax.set_title(
                r'$\mathrm{Areas\,for\,each\,2D\,sub-hist.,\,i.e.,\,kernel\,at\,coord.}\,(E_{\nu,\mathrm{true}},\,\cos\vartheta_\mathrm{true})$'+
                '\n' + r'$\mathrm{None\,should\,be>1\,(shown\,white);\,no\,event\,bins\,are\,black;\,avg.}=' + numFmt(np.mean(check_areas)) + r'$')
            fig.tight_layout()
            
            if save_pdf:
                fig.savefig(pp, format='pdf')
        
        if make_plots and save_pdf:
            pp.close()
        
        return kernels
















































        ecen = get_bin_centers(self.ebins)
        kde_dict = {'energy': [],
                    'coszen': []}
        min_events = 500
        min_bin_width = 1.0
        for ie,energy in enumerate(ecen):
            min_edge = self.ebins[ie]
            max_edge = self.ebins[ie+1]
            bin_width = (max_edge - min_edge)
            in_bin = np.alltrue(np.array([e_true >= min_edge,
                                          e_true < max_edge]),axis=0)
            nevents = np.sum(in_bin)
            # TESTING:
            logging.trace("working on energy %.2f, nevents: %.2f "%(energy,nevents))
            if ((nevents < min_events) or (bin_width < min_bin_width) ):
                logging.trace("  Increasing bin size for-> energy: %.2f, nevents: %.2d"
                              %(energy,nevents))
                lo_indx = ie
                hi_indx = ie+1
                while( (nevents < min_events) or (bin_width < min_bin_width) ):
                    
                    # Decrement lower bin edge if not at lowest:
                    if lo_indx > 0: lo_indx -= 1
                    min_edge = self.ebins[lo_indx]
                    
                    # Try increasing upper bin edge if not at max bin:
                    hi_indx += 1
                    try:
                        max_edge = self.ebins[hi_indx]
                    except:
                        hi_indx -= 1
                        max_edge = self.ebins[hi_indx]
                    
                    bin_width = (max_edge - min_edge)
                    in_bin = np.alltrue(np.array([e_true >= min_edge,
                                                  e_true < max_edge]),axis=0)
                    nevents = np.sum(in_bin)
                
                logging.trace("    Using %d bins for nevents: %d and width: %.2f"
                              %((hi_indx - lo_indx),nevents,bin_width))
            
            e_res_data = e_reco[in_bin] - e_true[in_bin]
            cz_res_data = cz_reco[in_bin] - cz_true[in_bin]
            egy_kde = gaussian_kde(e_res_data)
            
            kde_dict['energy'].append(egy_kde)
            kde_dict['coszen'].append(cz_kde)
        
        return kde_dict
    
    def _check_duplicate_nu_bar(self,kde_fh):
        '''
        If nu<flav> and nu<flav>_bar have duplicate fields, then don't
        duplicate the KDE construction, but use the same KDEs for each
        (saves time and increases statistics).
        '''
        
        if (np.all(np.array(kde_fh['nue/cc/reco_energy'])==
            np.array(kde_fh['nue_bar/cc/reco_energy'])) and
            np.all(np.array(kde_fh['numu/cc/reco_energy'])==
            np.array(kde_fh['numu_bar/cc/reco_energy'])) and
            np.all(np.array(kde_fh['nutau/cc/reco_energy'])==
            np.array(kde_fh['nutau_bar/cc/reco_energy']))):
            return True
        else:
            return False
