# pylint: disable=not-callable

"""
Stage for resolution improvement studies

"""

from __future__ import absolute_import, print_function, division

from pisa.core.stage import Stage
from pisa.utils.log import logging


class resolutions(Stage):
    """
    stage to change the reconstructed information by a given amount
    This can be used to esimate the impact of improved recosntruction
    resolutions for instance

    Parameters
    ----------
    params
        Expected params .. ::

            energy_improvement : quantity (dimensionless)
               scale the reco error down by this fraction
            coszen_improvement : quantity (dimensionless)
                scale the reco error down by this fraction
            pid_improvement : quantity (dimensionless)
                applies a shift to the classification parameter

    """
    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        calc_mode=None,
        apply_mode=None,
    ):
        expected_params = (
            'energy_improvement',
            'coszen_improvement',
            'pid_improvement',
        )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply

        # what are keys added or altered in the calculation used during apply

        # what keys are added or altered for the outputs during apply

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            calc_mode=calc_mode,
            apply_mode=apply_mode,
        )

        assert self.calc_mode is None
        assert self.input_mode == self.output_mode

    def setup_function(self):

        self.data.representation = self.apply_mode

        for container in self.data:
            logging.info('Changing energy resolutions')
            tmp = container['reco_energy']
            tmp += (container['true_energy'] - container['reco_energy']) * self.params.energy_improvement.m_as('dimensionless')
            container.mark_changed('reco_energy')

            logging.info('Changing coszen resolutions')
            tmp = container['reco_coszen']
            tmp += (container['true_coszen'] - container['reco_coszen']) * self.params.coszen_improvement.m_as('dimensionless')
            container.mark_changed('reco_coszen')
            # make sure coszen is within -1/1 ?

            logging.info('Changing PID resolutions')
            tmp = container['pid']
            if container.name in ['numu_cc', 'numubar_cc']:
                tmp += self.params.pid_improvement.m_as('dimensionless')
            else:
                tmp -= self.params.pid_improvement.m_as('dimensionless')
            container.mark_changed('pid')
