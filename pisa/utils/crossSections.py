#!/usr/bin/env python
#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
"""
Define CrossSections class for importing, working with, and storing neutrino
cross sections
"""

import os
import numpy as np
from copy import deepcopy

from scipy.interpolate import interp1d

import pisa.utils.fileio as fileio
import pisa.utils.flavInt as flavInt
import pisa.utils.utils as utils
from pisa.resources import resources as resources
from pisa.utils.log import logging, set_verbosity

# TODO: make class for groups of CX or just a function for finding eisting
# versions in a json file; eliminate the optional ver parameters in
# CrossSections class methods


class CrossSections(flavInt.FlavIntData):
    """Cross sections for each neutrino flavor & interaction type ("flavint").

    ver : str
        version of cross sections to load from file. e.g.: 'genie_2.6.4'

    xsec : str, dict, or another CrossSections object
        If str: provides PISA resource name from which to load cross sections
        If dict or CrossSections object: construct cross sections from a
            deepcopy of that object
    """
    def __init__(self, ver=None, xsec='cross_sections/cross_sections.json'):
        self.__ver = None
        if xsec is None:
            raise NotImplementedError(
                'Not able to instantiate an empty CrossSections object.'
            )
        elif isinstance(xsec, dict):
            xsec = deepcopy(xsec)
        elif isinstance(xsec, basestring):
            xsec = self.load(fpath=xsec, ver=ver)
        else:
            raise TypeError('Unhandled xsec type passed in arg: ' +
                            str(type(xsec)))

        if ver is not None:
            self.set_ver(ver)
            if xsec is not None:
                try:
                    xsec = xsec[ver]
                except KeyError:
                    pass

        if xsec is not None:
            self.validate(xsec)

        self.update(xsec)

    @staticmethod
    def load(fpath, ver=None, **kwargs):
        """Load cross sections from a file locatable and readable by the PISA
        from_file command. If `ver` is provided, it is used to index into the
        top level of the loaded dictionary"""
        xsec = fileio.from_file(fpath, **kwargs)
        if ver is not None:
            xsec = xsec[ver]
        return xsec

    @staticmethod
    def loadROOTFile(fpath, o_sfx='_o16', h_sfx='_h1'):
        """Load cross sections from root file, where graphs are first-level in
        hierarchy. This is yet crude and not very flexible, but at least it's
        recorded here for posterity.

        Requires ROOT and ROOT python module be installed

        Parameters
        ----------
        o_sfx : str (default = '_o16')
            Suffix for finding oxygen-16 cross sections in ROOT file
        h_sfx : str (default = '_h1')
            Suffix for finding hydrogen-1 cross sections in ROOT file

        Returns
        -------
        xsec : flavInt.NuFlavIntData
            Object containing the loaded cross sections
        """
        import ROOT

        def extractData(f, key):
            try:
                g = ROOT.gDirectory.Get(key)
                x = np.array([g.GetX()[i] for i in xrange(g.GetN())])
                y = np.array([g.GetY()[i] for i in xrange(g.GetN())])
            except AttributeError:
                logging.error('Possibly missing file `%s` or missing key `%s`'
                              ' within that file?' % (f, key))
                raise
            return x, y

        rfile = ROOT.TFile(fpath)
        try:
            xsec = flavInt.FlavIntData()
            for flavint in flavInt.ALL_NUFLAVINTS:
                O16_e, O16_xs = extractData(rfile, str(flavint) + o_sfx)
                H1_e, H1_xs = extractData(rfile, str(flavint) + h_sfx)
                if not xsec.has_key('energy'):
                    xsec['energy'] = O16_e
                assert np.alltrue(O16_e == xsec['energy'])
                assert np.alltrue(H1_e == xsec['energy'])
                # Note that units in ROOT files are [1e-38 cm^2] but PISA
                # requires units of [m^2], so the conversion is made here.
                this_flavint_xs = (H1_xs*2 + O16_xs) * 1e-38 * 1e-4
                xsec.set(flavint, this_flavint_xs)
        finally:
            rfile.Close()

        CrossSections.validate(xsec)

        return xsec

    @classmethod
    def newFromROOT(cls, fpath, ver=None):
        """Instantiate a new CrossSections object from ROOT file.

        Parameters
        ----------
        fpath : string
            PISA resource specification for location of ROOT file
        ver : string
            Specify the version name of the cross sections loaded

        Returns
        -------
        Instantiated CrossSections object
        """
        xsec = CrossSections.loadROOTFile(fpath)
        return cls(xsec=xsec, ver=ver)

    @staticmethod
    def validate(xsec):
        """Validate cross sections"""
        # TODO: different validation based on cross sections version string

        # Make sure the basics are present
        flavInt.FlavIntData(xsec)
        assert xsec.has_key('energy'), "missing 'energy'"

        energy = xsec['energy']
        # No NaN's
        assert np.sum(np.isnan(energy)) == 0
        # Energy spans at least 1-100 GeV
        assert np.min(energy) <= 1
        assert np.max(energy) >= 100
        # All event flavints need to be present
        for k in flavInt.ALL_NUFLAVINTS:
            # Uses "standard" PISA indexing scheme
            x = k.pidx(xsec)
            # Arrays are same lengths
            assert len(x) == len(energy)
            # No NaN's
            assert np.sum(np.isnan(x)) == 0
            # Max xsec/energy value is in range for units of [m^2/GeV]
            assert np.max(x/energy) < 40e-42

    def set_ver(self, ver):
        """Set the cross sections version to the string `ver`."""
        self.__ver = ver

    def ver(self):
        """Return the cross sections version string"""
        return self.__ver

    def get(self, *args):
        """Retrieve a cross section or the energies at which cross sections are
        defined"""
        if isinstance(args[0], basestring) and (args[0]).lower() in \
                ['e', 'energy', 'enu', 'e_nu']:
            return deepcopy(self['energy'])
        return flavInt.FlavIntData.get(self, *args)

    def set(self, *args):
        """Store a cross section for a given flavInt or store energy values.

        Parameters
        ----------
        *args
            If args[0] is one of 'e', 'energy', 'enu', or 'e_nu':
                Store args[1] as the energy values
            Otherwise:
                Use the first N-1 args to construct the flavint indexing into
                the FlavIntData object, and interpret the last arg (args[-1])
                as the cross section to be stored to that flavint
        """
        if isinstance(args[0], basestring) and (args[0]).lower() in \
                ['e', 'energy', 'enu', 'e_nu']:
            self['energy'] = deepcopy(args[1])
            return
        flavInt.FlavIntData.set(self, *args)

    def save(self, fpath, ver=None, **kwargs):
        """Save cross sections (and the energy specification) to a file at
        `fpath`."""
        if ver is None:
            if self.__ver is None:
                raise ValueError(
                    'Either a ver must be specified in call to `save` or it '
                    'must have been set prior to the invocation of `save`.'
                )
            ver = self.__ver
        try:
            fpath = resources.find_resource(fpath)
        except IOError:
            pass
        fpath = os.path.expandvars(os.path.expanduser(fpath))
        all_xs = {}
        if os.path.exists(fpath):
            old_data = fileio.from_file(fpath)
            all_xs.update({k:CrossSections(v) for k,v in old_data.iteritems()})
            if ver in all_xs:
                logging.warning('Overwriting existing version "' + ver +
                                '" in file ' + fpath)
        all_xs[ver] = self
        fileio.to_file(all_xs, fpath, **kwargs)

    def __combineXS(self, flavint_group):
        """Combine all cross sections specified by the flavints in
        flavint_group. All CC (NC) interactions are grouped together and
        averaged with one another and then the average of each interaction type
        is added to the other.

        If CC and NC interactions present, they *must* be from the same
        flavor(s). I.e., it doesn't make sense (and so causes an exception) if
        you combine numu CC with numubar NC. It does make sense if you combine
        numu and numubar CC with numu and numubar NC, though, and this is
        allowed.
        """
        flavint_group = flavInt.NuFlavIntGroup(flavint_group)
        cc_flavints = flavint_group.ccFlavInts()
        nc_flavints = flavint_group.ncFlavInts()
        if cc_flavints and nc_flavints:
            assert flavint_group.ccFlavs() == flavint_group.ncFlavs(), \
                    'Combining CC and NC but CC flavors do not match NC flavors'
        cc_avg_xs = 0
        if cc_flavints:
            logging.trace('cc_flavints = %s' % (cc_flavints,))
            cc_avg_xs = np.sum([self.get(k) for k in cc_flavints], axis=0) \
                    / len(cc_flavints)
        nc_avg_xs = 0
        if nc_flavints:
            logging.trace('nc_flavints = %s' % (nc_flavints,))
            nc_avg_xs = np.sum([self.get(k) for k in nc_flavints], axis=0) \
                    / len(nc_flavints)
        tot_xs = cc_avg_xs + nc_avg_xs
        logging.trace('mean(tot_xs) = %s' % (np.mean(tot_xs),))
        return tot_xs

    def integrate(self, flavint_group, e_range, gamma=0):
        """Numerical integral using trapezoidal rule of combined cross sections
        of flavints specfied in `flavint_group`; if `gamma` specified, weight
        integral by simulated spectrum with that power-spectral index
        (E^{-gamma})"""
        e_min = min(e_range)
        e_max = max(e_range)
        n_steps = int(max(1e5, np.ceil(1e5*(e_max-e_min))))
        energy = np.linspace(e_min, e_max, n_steps)

        # Get combined cross section for all flavints
        xs_data = self.__combineXS(flavint_group)

        logging.trace('mean(xs_data) = %e' % (np.mean(xs_data)))

        # Create interpolant (for energy range's endpoints)
        xs_interp = interp1d(x=self['energy'], y=xs_data,
                             kind='linear', copy=True, bounds_error=True,
                             fill_value=0, assume_sorted=False)

        # Get indices of data points within the specified energy range
        idx = (self['energy'] > e_min) & (self['energy'] < e_max)

        # Get xsec at endpoints
        xs_emin = xs_interp(e_min)
        xs_emax = xs_interp(e_max)

        logging.trace('xs_emin = %e, xs_emax = %e' %(xs_emin, xs_emax))

        # Attach endpoints and scale xsec by simulated energy spectrum (power
        # law exponent = -gamma).
        energy = np.concatenate([[e_min], self['energy'][idx], [e_max]])
        xs = np.concatenate([[xs_emin], xs_data[idx], [xs_emax]]) \
                * energy**(-gamma)
        logging.trace('mean(xs) = %e' % (np.mean(xs)))

        # Integral via trapezoidal rule
        xs_integral = np.trapz(y=xs, x=energy)

        logging.trace('xs_integral = %e' % (xs_integral))

        # If weighting was applied, normalize by dividing by integral of weight
        # function in the same energy range
        if gamma != 0:
            xs_integral = xs_integral / \
                    np.trapz(y=energy**(-gamma), x=energy) * (e_max-e_min)

        logging.trace('xs_integral = %e' %(xs_integral))

        return xs_integral

    def ratio(self, flavint_group0, flavint_group1, e_range, gamma):
        """Ratio of numerical integrals of combined cross sections for
        flavint_group0 to combined cross sections for flavint_group1, using the
        integrate method defined elsewhere in this class. Integral is from
        min(e_range) to max(e_range) and is weighted by E**(-gamma), the
        simlation spectrum."""
        int0 = self.integrate(flavint_group=flavint_group0,
                              e_range=e_range,
                              gamma=gamma)
        int1 = self.integrate(flavint_group=flavint_group1,
                              e_range=e_range,
                              gamma=gamma)
        ratio = int0 / int1
        logging.trace('int0/int1 = %e' % (ratio))
        return ratio

    def mean(self, flavint_group, e_range, gamma):
        """Mean of combined cross sections for flavints in `flavint_group`,
        using the integrate method defined elsewhere in this class. Mean is
        from min(`e_range`) to max(`e_range`) and is weighted by E**(-`gamma`),
        the simlation spectrum."""
        int0 = self.integrate(flavint_group=flavint_group,
                              e_range=e_range,
                              gamma=gamma)
        mean = int0 / (max(e_range) - min(e_range))
        logging.trace('mean(sigma) = %e' % (mean))
        return mean

    def plot(self, save=None):
        """Plot cross sections per GeV; optionally, save plot to file. Requires
        matplotlib and optionally uses seaborn to set "pretty" defaults.

        save : None, str, or dict
            If None, figure is not saved (default)
            If a string, saves figure as that name
            If a dict, calls pyplot.savefig(**save) (i.e., save contains
            keyword args to savefig)
        """
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            pass

        if self.__ver is None:
            leg_ver = ''
        else:
            leg_ver = self.__ver + '\n'
        leg_fs = 11
        figsize = (9, 6)
        alpha = 1.0
        f = plt.figure(figsize=figsize)
        f.clf()
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        ls = [dict(lw=5, ls='-'),
              dict(lw=2, ls='-'),
              dict(lw=1, ls='--'),
              dict(lw=5, ls='-'),
              dict(lw=3, ls='-.'),
              dict(lw=1, ls='-')]

        energy = self['energy']
        nc_n = cc_n = 0
        for flavint in list(flavInt.ALL_NUFLAVINTS.particles()) + \
                list(flavInt.ALL_NUFLAVINTS.antiParticles()):
            # Convert from [m^2] to [1e-38 cm^2]
            xs = flavint.pidx(self) * 1e38 * 1e4
            if flavint.isCC():
                ax1.plot(energy, xs/energy,
                         alpha=alpha,
                         label=flavInt.tex(flavint.flav(), d=1),
                         **ls[cc_n%len(ls)])
                cc_n += 1
            else:
                ax2.plot(energy, xs/energy,
                         alpha=alpha,
                         label=flavInt.tex(flavint.flav(), d=1),
                         **ls[nc_n%len(ls)])
                nc_n += 1

        l1 = ax1.legend(title=leg_ver +
                        r'$\nu+{\rm H_2O}$, total CC',
                        fontsize=leg_fs, frameon=False, ncol=2)
        l2 = ax2.legend(title=leg_ver +
                        r'$\nu+{\rm H_2O}$, total NC',
                        fontsize=leg_fs, frameon=False, ncol=2)

        for (ax, leg) in [(ax1, l1), (ax2, l2)]:
            ax.set_xlim(0.1, 100)
            ax.set_xscale('log')
            ax.set_xlabel(r'$E_\nu\,{\rm [GeV]}$')
            ax.set_ylabel(
                r'$\sigma(E_\nu)/E_\nu\quad ' +
                r'\left[10^{-38} \mathrm{cm^2} \mathrm{GeV^{-1}}\right]$'
            )
            leg.get_title().set_fontsize(leg_fs)
        plt.tight_layout()

        if isinstance(save, basestring):
            logging.info('Saving cross sections plots to file ' + save)
            f.savefig(save)
        elif isinstance(save, dict):
            logging.info('Saving cross sections plots using figure.save'
                         ' params %s' % (save,))
            f.savefig(**save)


def test_CrossSections():
    set_verbosity(2)

    # "Standard" location of cross sections file in PISA; retrieve 2.6.4 for
    # testing purposes
    pisa_xs_file = 'cross_sections/cross_sections.json'
    xs = CrossSections(ver='genie_2.6.4', xsec=pisa_xs_file)

    # Location of the root file to use (not included in PISA at the moment)
    test_dir = utils.expandPath('$PISA/test/cross_sections')
    ROOT_xs_file = os.path.join(test_dir, 'genie_2.6.4_simplified.root')

    # Make sure that the XS newly-imported from ROOT match those stored in PISA
    if os.path.isfile(ROOT_xs_file):
        xs_from_root = CrossSections.newFromROOT(ROOT_xs_file,
                                                 ver='genie_2.6.4')
        logging.info('Found and loaded ROOT source cross sections file %s'
                     % ROOT_xs_file)
        assert xs_from_root.allclose(xs, rtol=1e-7)

    # Check XS ratio for numu_cc to numu_cc + numu_nc (user must inspect)
    kg0 = flavInt.NuFlavIntGroup('numu_cc')
    kg1 = flavInt.NuFlavIntGroup('numu_nc')
    logging.info('\\int_1^80 xs(numu_cc) E^{-1} dE = %e' %
                 xs.integrate(kg0, e_range=[1, 80], gamma=-1))
    logging.info('int(sigma_numu_cc)/int(sigma_(numu_cc+numu_nc)) = %e' %
                 xs.ratio(kg0, kg0+kg1, e_range=[1, 80], gamma=-1))
    # Check that XS ratio for numu_cc+numu_nc to the same is 1.0
    assert xs.ratio(kg0+kg1, kg0+kg1, e_range=[1, 80], gamma=-1) == 1.0

    # Check via plot that the

    # Plot all cross sections stored in PISA xs file
    try:
        alldata = fileio.from_file(pisa_xs_file)
        xs_versions = alldata.keys()
        for ver in xs_versions:
            xs = CrossSections(ver=ver, xsec=pisa_xs_file)
            xs.plot(save=os.path.join(
                test_dir,
                'pisa_' + ver + '_nuxCCNC_H2O_cross_sections.pdf'
            ))
    except ImportError as exc:
        logging.debug('Could not plot; possible that matplotlib not'
                      'installed. ImportError: %s' % exc)


if __name__ == "__main__":
    test_CrossSections()
