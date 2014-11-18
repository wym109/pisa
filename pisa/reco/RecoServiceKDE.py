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
import h5py

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers, get_bin_sizes, is_linear
from pisa.utils.log import logging, physics

class RecoServiceKDE(RecoServiceBase):
    """
    Creates reconstruction kernels using Kernel Density Estimation (KDE).
    """

    def __init__(self, ebins, czbins, kdefile=None, **kwargs):
        """
        Parameters needed to instantiate a reconstruction service with
        parametrizations:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * kdefile: HDF5 file containing the MC events.
        """
        self.kdefile = kdefile
        RecoServiceBase.__init__(self, ebins, czbins, kdefile=kdefile, **kwargs)


    def _get_reco_kernels(self, kdefile=None, remove_sim_downgoing=True,
                         **kwargs):
        
        # Check for invalid reco scales
        for reco_scale in ['e_reco_scale', 'cz_reco_scale']:
            if reco_scale in kwargs:
                if not kwargs[reco_scale]==1:
                    raise ValueError('%s = %.2f not valid for RecoServiceMC!'
                                     %(reco_scale, kwargs[reco_scale]))

        if not kdefile in [self.kdefile, None]:
            logging.info('Reconstruction from non-default MC file %s!'%kdefile)
            temp_kde_dict = self.construct_KDEs(kdefile,remove_sim_downgoing=remove_sim_downgoing)
            return self.calculate_kernels(kde_dict=temp_kde_dict)
        
        if not hasattr(self, 'kde_dict'):
            logging.info('Constructing KDEs from file: %s'%kdefile)
            self.kde_dict = self.construct_KDEs(kdefile,remove_sim_downgoing=remove_sim_downgoing)

        if not hasattr(self, 'kernels'):
            logging.info('Creating reconstruction kernels')
            self.kernels = self.calculate_kernels(kde_dict=self.kde_dict)

        #TODO: future: if reco scales != 1, recalculate

        return self.kernels


    def calculate_kernels(self, kde_dict=None, flipback=True):
        
        #TODO: implement reco scales here
        
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
        for flavour in kde_dict.keys():
            for int_type in ['cc','nc']:
                logging.debug('Calculating KDE based reconstruction kernel for %s %s'
                              %(flavour, int_type))

                # create empty kernel
                kernel = np.zeros((n_e, n_cz, n_e, n_cz))

                # loop over every bin in true (energy, coszen)
                for i in range(n_e):
                    energy = evals[i]
                    kvals = evals - energy
                    e_kern = self._get_1D_kernel(flavour,int_type,energy,kvals,
                                                 kde_dict=kde_dict,ktype="energy")

                    for j in range(n_cz):
                        offset = n_cz if flipback else 0
                        #print "energy: %.2f coszen: %.2f"%(evals[i],czvals[j+offset])
                        kvals = czvals - czvals[j+offset]
                        cz_kern = self._get_1D_kernel(flavour,int_type,energy,kvals,
                                                      kde_dict=kde_dict,ktype="coszen")

                        if flipback:
                            # fold back
                            cz_kern = cz_kern[:len(czvals)/2][::-1] + cz_kern[len(czvals)/2:]

                        kernel[i,j] = np.outer(e_kern, cz_kern)

                        if np.any(np.isnan(kernel[i,j])):
                            print "nan found! --why??"
                            #print "  kernel: ",kernel[i,j]
                            print "  cz_kern: ",cz_kern
                            print "  e_kern: ",e_kern
                            print "energy: %.2f, cz: %.2f"%(evals[i],czvals[j])
                            sys.exit(1)


                kernel_dict[flavour][int_type] = kernel
        kernel_dict['ebins'] = self.ebins
        kernel_dict['czbins'] = self.czbins

        return kernel_dict


    def _get_1D_kernel(self, flavour, int_type, energy, kvals, 
                       kde_dict=None, ktype=None):
        '''
        For the specified flavour, int_type, returns the estimated 1D kernel
        at 'energy' using linear histogram interpolation from KDEs defined
        at discrete energies.
        \params:
          * flavour - flavour of neutrino to get KDE for
          * int_type - interaction type ['cc' or 'nc']
          * energy - energy of bin center to get 1D kernel
          * kvals - values to apply reco kernel to. If coszen
            kernel, these should be (czvals - coszen) or if energy
            kernels, should be (evals - energy).
          * kde_dict - dict holding the kde parameters
          * ktype - kernel type to extract, either 'coszen' or 'energy'
        '''

        # For easy access:
        kde_dict = kde_dict[flavour][int_type]

        kde_key = ""
        if ktype == 'coszen': kde_key = "cz_kde"
        elif ktype == 'energy': kde_key = "egy_kde"
        else:
            raise NameError("ktype of %s is invalid. Please choose from strings [coszen, energy]"%ktype)

        # If energy lies outside the bounds of the defined KDEs,
        # simply return the pdf defined at the bound
        if energy <= kde_dict[0]["ecen"]:
            return (kde_dict[0][kde_key]).evaluate(kvals)
        if energy >= kde_dict[-1]["ecen"]:
            return (kde_dict[-1][kde_key]).evaluate(kvals)

        # Find nearest neighbors, then interpolate kde...
        i_low = 0
        for i in range(len((kde_dict[:]))):
            if (energy >= kde_dict[i]['ecen'] and energy <= kde_dict[i+1]['ecen']):
                i_low = i
                break

        kde_low = kde_dict[i_low][kde_key]
        e_low = kde_dict[i_low]['ecen']
        i_high = i_low+1
        kde_high = kde_dict[i_high][kde_key]
        e_high = kde_dict[i_high]['ecen']

        # Now interpolate these two pdfs:
        pdf = self._interpolate_pdf(kde_low,kde_high,e_low,e_high,energy,kvals)
        return pdf


    def _interpolate_pdf(self,kde_low,kde_high,val_low,val_high,value,xvals):
        '''
        Interpolate two pdfs defined on the xaxis=xvals for some
        parameter q defined at q=val_low and q=val_high.

        Using formula of "Linear Interpolation of Histograms" by
        A.L. Read, Nuc. Instr. and Methods in Physics Research A 425
        (1999) 357-360.

        \params:
          * kde_low - KDE describing distribution at val_low
          * kde_high - KDE describing distribution at val_high
          * val_low - value of parameter where kde_low is defined.
          * val_high - value of parameter where kde_high is defined.
          * value - value of parameter where we wish to define the pdf
            using interpolated data from kde_low and kde_high
          * xvals - values of x-axis on which to define the pdf.
        '''
        b = (value - val_low)/(val_high - val_low)
        a = 1.0 - b

        if (a+b > 1.0):
            raise ValueError('a + b can never be greater than 1!')

        f1 = kde_low
        f2 = kde_high
        pdf = []
        for x in xvals:
            x1 = x; x2 = x1
            yval = []
            if (f2.evaluate(x2) < 1.0e-10 and f1.evaluate(x1) < 1.0e-10):
                yval.append(0.0)
            else:
                yval = (f1.evaluate(x1)*f2.evaluate(x2)/
                        (a*f2.evaluate(x2) + b*f1.evaluate(x1)))
            pdf.append(yval[0])

        return np.array(pdf)

    def construct_KDEs(self,kdefile,remove_sim_downgoing=True):
        """
        Constructs the KDEs from the data files in 'kde_settings'. KDEs are then
        used as a parametrization to get the kernels in calculate_kernels()
        """

        try:
            kde_fh = h5py.File(find_resource(kdefile),'r')
        except IOError, e:
            logging.error("Unable to open KDE file %s"%kdefile)
            logging.error(e)
            sys.exit(1)

        logging.debug("Removing simulated downgoing events in KDE construction.")
        kde_dict = {}
        for flavour in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            flavour_dict = {}
            logging.debug("Working on %s kernels"%flavour)
            for int_type in ['cc','nc']:
                true_energy = np.array(kde_fh[flavour+'/'+int_type+'/true_energy'])
                true_coszen = np.array(kde_fh[flavour+'/'+int_type+'/true_coszen'])
                reco_energy = np.array(kde_fh[flavour+'/'+int_type+'/reco_energy'])
                reco_coszen = np.array(kde_fh[flavour+'/'+int_type+'/reco_coszen'])

                # Cut on simulated downgoing:
                if remove_sim_downgoing:
                    cuts = np.alltrue(np.array([true_coszen < 0.0]),axis=0)
                    true_energy = true_energy[cuts]
                    true_coszen = true_coszen[cuts]
                    reco_energy = reco_energy[cuts]
                    reco_coszen = reco_coszen[cuts]

                flavour_dict[int_type] = self.get_kde_list(true_energy,true_coszen,
                                                          reco_energy,reco_coszen)
            kde_dict[flavour] = flavour_dict

        return kde_dict

    def get_kde_list(self,e_true,cz_true,e_reco,cz_reco):
        '''
        For the set of true/reco data, form the kde vs. energy at discrete
        energy bins. Due to decreasing statistics at higher energies, the bin
        size with which to characterize the KDE will be varied, so that a min.
        number of counts will be present in each determination of the KDE of
        the resolution.

        returns - List of KDE where each element is (emin,emax,pdf_eres,pdf_czres)
        so that
        '''
        min_egy = self.ebins[0]
        max_egy = self.ebins[-1]
        ebin_width = 2 # GeV
        min_events = 800

        min_edge = min_egy
        max_edge = min_egy + ebin_width
        kde_list = []
        while (min_edge < max_egy):
            energy = (min_edge + max_edge)/2.0
            in_bin = np.alltrue(np.array([e_true >= min_edge,
                                          e_true < max_edge]), axis=0)
            e_res_data = e_reco[in_bin] - e_true[in_bin]
            cz_res_data = cz_reco[in_bin] - cz_true[in_bin]
            nevents = np.sum(in_bin)
            if (nevents < min_events):
                logging.trace("  Sliding bin width for-> energy: %.2f, nevents: %.2d"
                              %(energy,nevents))
                max_edge = self._get_max_edge(min_events,min_edge,max_egy,
                                              e_true,e_reco)
                # Break out of the loop, and stop getting the KDEs when
                # the bin edge can't be extended beyond the max energy
                # in the simulated data.
                if (max_edge >= max_egy): break
                in_bin = np.alltrue(np.array([e_true >= min_edge,
                                              e_true < max_edge]), axis=0)
                e_res_data = e_reco[in_bin] - e_true[in_bin]
                cz_res_data = cz_reco[in_bin] - cz_true[in_bin]


            egy_kde = gaussian_kde(e_res_data)
            cz_kde = gaussian_kde(cz_res_data)
            kde_list.append({"min_edge":min_edge,"max_edge":max_edge,
                             "ecen":(min_edge + max_edge)/2.0,
                             "egy_kde":egy_kde,"cz_kde":cz_kde})

            min_edge = max_edge
            max_edge = min_edge + ebin_width

        return kde_list

    def _get_max_edge(self,min_events,min_edge,max_egy,e_true,e_reco):
        '''
        If the number of events to define the KDE for this bin is too
        small, then redefine the bin size by extending the max edge of
        the bin until the number of counts is sufficient to get a good
        characterization of the KDE.

        \params:
          * min_edge - lower edge of the energy bin
          * max_egy  - maximum energy of the energy bins
          * e_true   - true energy list for simulated data
          * e_reco   - reco energy list for simulated data
        '''

        # First, make sure that there EXISTS a max edge betwen min_edge, max_egy:
        in_bin = np.alltrue(np.array([e_true >= min_edge,e_true < max_egy]),axis=0)
        e_res_data = e_reco[in_bin] - e_true[in_bin]
        if (np.sum(in_bin) < min_events): return max_egy

        # Give 10 as an error in number of counts:
        delta_N = 10

        # Find new ebin_width, get new res_data
        e_upper = max_egy
        e_lower = min_edge
        e_next = 0.0
        while(True):
            e_next = (e_upper + e_lower)/2.0
            in_bin = np.alltrue(np.array([e_true >=min_edge,
                                          e_true < e_next]),axis=0)
            e_res_data = e_reco[in_bin] - e_true[in_bin]
            counts = np.sum(in_bin)
            if(np.fabs(min_events - counts) < delta_N): break
            else:
                if(counts > min_events): e_upper = e_next
                else: e_lower = e_next

            #raw_input("PAUSED...")

        #print "new max edge found of: ",e_next
        return e_next
