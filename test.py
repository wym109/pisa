from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.stats import chi2
from pisa.utils.log import set_verbosity

set_verbosity(1)

livetimes = [1, 2, 3, 4, 5, 6] * ureg.year

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
    temp_out = template_maker.get_outputs()[0].pop()

    chisqaure.append(
        chi2(actual_values=nom_out.hist, expected_values=temp_out.hist)
    )

print chisqaure
