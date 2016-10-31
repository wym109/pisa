from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.log import set_verbosity

set_verbosity(1)

livetimes = [1, 50, 100] * ureg.year

template_maker = DistributionMaker('settings/pipeline/cfx.cfg')

template_maker.params.fix(template_maker.params.free)
template_maker.params.unfix('livetime')

sf_param = template_maker.params['stat_fluctuations']
lt_param = template_maker.params['livetime']

chisqaure = []
for lt in livetimes:
    print '==========='
    print 'livetime = {0}'.format(lt)
    print '==========='

    lt_param.value = lt
    template_maker.update_params(lt_param)

    sf_param.value = False
    template_maker.update_params(sf_param)
    nom_out = template_maker.get_outputs()[0].pop()

    sf_param.value = True
    template_maker.update_params(sf_param)
    t_chi2 = []
    for x in xrange(20):
        temp_out = template_maker.get_outputs()[0].pop()
        try:
            t_chi2.append(
                nom_out.chi2(expected_values=temp_out.hist)
            )
        except ValueError:
            pass
    chisqaure.append(np.mean(t_chi2))

print chisqaure
