from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.log import set_verbosity

set_verbosity(1)

# livetimes = [2, 3, 4, 5, 6, 7, 8] * ureg.common_year
livetimes = [1, 4, 16, 64] * ureg.common_year
# livetimes = [1, 2] * ureg.common_year

template_maker = DistributionMaker('settings/pipeline/cfx.cfg')

template_maker.params.fix(template_maker.params.free)
template_maker.params.unfix('livetime')

re_param = template_maker.params['regularisation']
sf_param = template_maker.params['stat_fluctuations']
lt_param = template_maker.params['livetime']

frac_err, frac_err_err = [], []
for lt in livetimes:
    print '==========='
    print 'livetime = {0}'.format(lt)
    print '==========='

    lt_param.value = lt
    template_maker.update_params(lt_param)

    re_param.value = 0 * ureg.dimensionless
    template_maker.update_params(re_param)
    nom_out = template_maker.get_outputs()[0].pop()

    re_param.value = 2 * ureg.dimensionless
    sf_param.value = 1234 * ureg.dimensionless
    template_maker.update_params(re_param)
    template_maker.update_params(sf_param)
    fe = []
    for x in xrange(200):
        temp_out = template_maker.get_outputs()[0].pop()
        with np.errstate(divide='ignore', invalid='ignore'):
            div = unp.nominal_values(temp_out.hist) / \
                    unp.nominal_values(nom_out.hist)
        num_invalid = ~np.isfinite(div)
        div[num_invalid] = 0
        fe.append(div)
    frac_err.append(np.mean(fe))
    frac_err_err.append(np.std(fe))

frac_err, frac_err_err = map(np.array, (frac_err, frac_err_err))
fe = unp.uarray(frac_err, frac_err_err)
print fe

import matplotlib as mpl
# headless mode
mpl.use('Agg')
# fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

binning = livetimes.m
fig = plt.figure(figsize=(9, 5))
fig.suptitle('Ratio to nominal with errors representing the spread \n'
             'of an ensemble of unfoldings with psuedodata', fontsize=18,
             y=1.007)
ax = fig.add_subplot(111)
ax.set_xlim(np.min(binning)-1, np.max(binning)+1)
ax.set_ylim(0.5, 1.5)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=12)

ax.set_xlabel('livetime (years)', fontsize=18)
for ymaj in ax.yaxis.get_majorticklocs():
    ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=1)
for xmaj in ax.xaxis.get_majorticklocs():
    ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=1)

def get_edges_from_cen(bincen):
    hwidth = 0.5*(bincen[1] - bincen[0])
    return np.append([bincen[0]-hwidth], bincen[:]+hwidth)

fe_0 = np.concatenate([[fe[0]], fe])
ax.errorbar(
    binning, unp.nominal_values(fe), color='blue', xerr=0,
    yerr=unp.std_devs(fe), capsize=3, alpha=1, linestyle='--',
    markersize=2, linewidth=1
)
fig.savefig('./images/livetime.png', bbox_inches='tight', dpi=150)
