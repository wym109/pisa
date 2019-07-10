"""
Stage to apply pre-calculated Genie uncertainties
"""

from __future__ import absolute_import, print_function, division

__all__ = ["pi_genie_sys", "SIGNATURE", "apply_genie_sys"]

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE
import math

class pi_genie_sys(PiStage): # pylint: disable=invalid-name
    """
    Stage to apply pre-calculated Genie systematics.

    Paramaters
    ----------
    data
    params
        Must contain ::

            Genie_Ma_QE : quantity (dimensionless)
            Genie_Ma_RES : quantity (dimensionless)

    input_names
    output_names
    debug_mode
    input_specs
    calc_specs
    output_specs

    Notes
    -----
    Requires the events have the following keys :

        GENIE_linear_fit_MaCCQE
            Genie CC quasi elastic linear coefficient
        GENIE_quad_fit_MaCCQE
            Genie CC quasi elastic quadratic coefficient
        GENIE_linear_fit_MaCCRES
            Genie CC resonance linear coefficient
        GENIE_quad_fit_MaCCRES
            Genie CC resonance quadratic coefficient

    """
    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        input_specs=None,
        calc_specs=None,
        output_specs=None,
    ):
        expected_params = (
            'Genie_Ma_QE',
            'Genie_Ma_RES',
            'Genie_AhtBY',
            'Genie_BhtBY',
            'Genie_CV1uBY',
            'Genie_CV2uBY',
            'nu_diff_DIS',
            'nubar_diff_DIS',
            'hadron_DIS',
            'A_scale_DIS',
        )
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_apply_keys = (
            'GENIE_linear_fit_MaCCQE',
            'GENIE_quad_fit_MaCCQE',
            'GENIE_linear_fit_MaCCRES',
            'GENIE_quad_fit_MaCCRES',
            'GENIE_linear_fit_AhtBY',
            'GENIE_quad_fit_AhtBY',
            'GENIE_linear_fit_BhtBY',
            'GENIE_quad_fit_BhtBY',
            'GENIE_linear_fit_CV1uBY',
            'GENIE_quad_fit_CV1uBY',
            'GENIE_linear_fit_CV2uBY',
            'GENIE_quad_fit_CV2uBY',            
            'GENIE_x',            
            'GENIE_y',            
            'GENIE_A',            
        )
        # what keys are added or altered for the outputs during apply
        output_apply_keys = (
            'weights',
        )

        # init base class
        super(pi_genie_sys, self).__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        assert self.input_mode is not None
        assert self.calc_mode is None
        assert self.output_mode is not None

    @profile
    def apply_function(self):
        genie_ma_qe = self.params.Genie_Ma_QE.m_as('dimensionless')
        genie_ma_res = self.params.Genie_Ma_RES.m_as('dimensionless')
        genie_ahtby = self.params.Genie_AhtBY.m_as('dimensionless')
        genie_bhyby = self.params.Genie_BhtBY.m_as('dimensionless')
        genie_cv1uby = self.params.Genie_CV1uBY.m_as('dimensionless')
        genie_cv2uby = self.params.Genie_CV2uBY.m_as('dimensionless')
        nu_diff_dis = self.params.nu_diff_DIS.m_as('dimensionless')
        nubar_diff_dis = self.params.nubar_diff_DIS.m_as('dimensionless')
        hadron_dis = self.params.hadron_DIS.m_as('dimensionless')
        a_scale_dis = self.params.A_scale_DIS.m_as('dimensionless')

        for container in self.data:
            apply_genie_sys(
                genie_ma_qe,
                container['GENIE_linear_fit_MaCCQE'].get(WHERE),
                container['GENIE_quad_fit_MaCCQE'].get(WHERE),
                genie_ma_res,
                container['GENIE_linear_fit_MaCCRES'].get(WHERE),
                container['GENIE_quad_fit_MaCCRES'].get(WHERE),
                genie_ahtby,
                container['GENIE_linear_fit_AhtBY'].get(WHERE),
                container['GENIE_quad_fit_AhtBY'].get(WHERE),
                genie_bhyby,
                container['GENIE_linear_fit_BhtBY'].get(WHERE),
                container['GENIE_quad_fit_BhtBY'].get(WHERE),
                genie_cv1uby,
                container['GENIE_linear_fit_CV1uBY'].get(WHERE),
                container['GENIE_quad_fit_CV1uBY'].get(WHERE),
                genie_cv2uby,
                container['GENIE_linear_fit_CV2uBY'].get(WHERE),
                container['GENIE_quad_fit_CV2uBY'].get(WHERE),

                out=container['weights'].get(WHERE),
            )

            # Differential xsec systematic
            diff_dis = nu_diff_dis
            diff_norm = 1.- (1.6525*diff_dis)

            if 'bar' in container.name:
                diff_dis = nubar_diff_dis
                diff_norm = 1.- (1.8073*diff_dis)

            apply_diffxsec_sys(diff_dis,
                               diff_norm,
                               container['GENIE_x'].get(WHERE),
                               out=container['weights'].get(WHERE),
            )                

            # High W hadronization systematic
            if hadron_dis != 0.:
                apply_highW_sys(hadron_dis,
                                container['GENIE_y'].get(WHERE),
                                out=container['weights'].get(WHERE),
                )                

            # DIS A-dependent systematic
            if a_scale_dis:
                apply_Ascale_sys(container['GENIE_A'].get(WHERE),
                                container['GENIE_x'].get(WHERE),
                                out=container['weights'].get(WHERE),
                )                
            
            container['weights'].mark_changed(WHERE)            


if FTYPE == np.float64:
    SIGNATURE = '(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8[:])'
    SIGNATURE2 = '(f8, f8, f8, f8[:])'
    SIGNATURE3 = '(f8, f8, f8[:])'
else:
    SIGNATURE = '(f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4[:])'
    SIGNATURE2 = '(f4, f4, f4, f4[:])'
    SIGNATURE3 = '(f4, f4, f4[:])'

@guvectorize([SIGNATURE], '(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()->()', target=TARGET)
def apply_genie_sys(genie_ma_qe, linear_fit_maccqe, quad_fit_maccqe, genie_ma_res, linear_fit_maccres, quad_fit_maccres, 
                    genie_ahtby, linear_fit_ahtby, quad_fit_ahtby, genie_bhtby, linear_fit_bhtby, quad_fit_bhtby, 
                    genie_cv1uby, linear_fit_cv1uby, quad_fit_cv1uby, genie_cv2uby, linear_fit_cv2uby, quad_fit_cv2uby, 
                    out):
    out[0] *= (
            (1. + (linear_fit_maccqe + quad_fit_maccqe * genie_ma_qe) * genie_ma_qe)
            * (1. + (linear_fit_maccres + quad_fit_maccres * genie_ma_res) * genie_ma_res)
            * (1. + (linear_fit_ahtby + quad_fit_ahtby * genie_ahtby) * genie_ahtby)
            * (1. + (linear_fit_bhtby + quad_fit_bhtby * genie_bhtby) * genie_bhtby)
            * (1. + (linear_fit_cv1uby + quad_fit_cv1uby * genie_cv1uby) * genie_cv1uby)
            * (1. + (linear_fit_cv2uby + quad_fit_cv2uby * genie_cv2uby) * genie_cv2uby)
            )

@guvectorize([SIGNATURE2], '(),(),()->()', target=TARGET)
def apply_diffxsec_sys(diff_dis, diff_norm, genie_x, out):
    out[0] *= (
            diff_norm * math.pow(math.fabs(genie_x), -diff_dis)
            )

@guvectorize([SIGNATURE3], '(),()->()', target=TARGET)
def apply_highW_sys(hadron_dis, genie_y, out):
    out[0] *= (
            1./(1.+ (2.* hadron_dis * math.exp(-(genie_y/hadron_dis))))
            )
@guvectorize([SIGNATURE3], '(),()->()', target=TARGET)
def apply_Ascale_sys(genie_a, genie_x, out):
    p0 = (10.*genie_a)/(16.+(9.9-0.0084*genie_a)*genie_a)
    p1 = (0.95*(15.-genie_a))/genie_a
    p2 = (0.95*(genie_a-13.25))/(genie_a-10.)

    out[0] *= (
            p0+((p1+p2*genie_x)*genie_x)
            )
