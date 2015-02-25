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

from copy import deepcopy as copy

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
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


    def construct_1DKDE_dict(self,kdefile,remove_sim_downgoing=True):
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

                flavour_dict[int_type] = self._get_KDEs(true_energy,true_coszen,
                                                             reco_energy,reco_coszen)
            kde_dict[flavour] = copy(flavour_dict)
            if self.duplicate_nu_bar:
                flavour += '_bar'
                logging.debug("   >Copying into %s kernels"%flavour)
                kde_dict[flavour]  = copy(flavour_dict)

        return kde_dict


    def _get_KDEs(self,e_true,cz_true,e_reco,cz_reco):
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
