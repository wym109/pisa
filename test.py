from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.log import set_verbosity

set_verbosity(1)

livetimes = [1, 4, 16, 64] * ureg.year

template_maker = DistributionMaker('settings/pipeline/cfx.cfg')

template_maker.params.fix(template_maker.params.free)
template_maker.params.unfix('livetime')

sf_param = template_maker.params['stat_fluctuations']
lt_param = template_maker.params['livetime']

frac_err = []
for lt in livetimes:
    print '==========='
    print 'livetime = {0}'.format(lt)
    print '==========='

    lt_param.value = lt
    template_maker.update_params(lt_param)

    sf_param.value = True
    template_maker.update_params(sf_param)
    fe = []
    for x in xrange(20):
        temp_out = template_maker.get_outputs()[0].pop()
        fe.append(np.mean(unp.std_devs(temp_out.hist)) / \
                      np.sum(unp.nominal_values(temp_out.hist)))
    frac_err.append(np.mean(fe))

print frac_err
