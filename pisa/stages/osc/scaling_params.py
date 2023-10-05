"""
Author : Sharmistha Chattopadhyay
Date : August 10, 2023
"""

from __future__ import absolute_import,division

import numpy as np
import os
from pisa import FTYPE
import numba
# FTYPE = np.float32

if numba is None:
    class jit(object):
        """Decorator class to mimic Numba's `jit` when Numba is missing"""
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args):
            return args[0]
else:
    jit = numba.jit
    ftype = numba.typeof(FTYPE(1))


__all__ = ['mass_scaling','core_scaling_constrained','Layers_scale']

class mass_scaling():
    """
    Uses a single scaling factor for all the layers
    """
    def __init__(self):
        self._density_scale = 0.

    @property
    def density_scale(self):
        
        return self._density_scale
    
    @density_scale.setter
    def density_scale(self, value):
        self._density_scale = value
    

class core_scaling_constrained(object):
    """
    Returns scaling factors for inner mantle and middle mantle by taking scaling factor of inner core and outer core as input.
    Scaling factor of inner and outer core = core_density_scale (alpha)
    Scaling factor of inner mantle = beta
    Scaling factor of middle mantle = gamma
    Outer mantle not scaled
    This function solves the equations for two constraints: mass of earth and moment of inertia, by taking core_density_scale as an independent 
    parameter, and returns scaling factor factors for inner and outer mantle.
    
    """
    def __init__(self):
        self._core_density_scale = 0.

    @property
    def core_density_scale(self):
        
        return self._core_density_scale
    
    @core_density_scale.setter
    def core_density_scale(self, value):
        self._core_density_scale = value

    def is_positive(self,lst):
        for i in range(len(lst)):
            if lst[i] < 0:
                return False
        return True    
    
    def is_descending(self,lst):
        for i in range(len(lst) - 1):
            if lst[i] < lst[i + 1]:
                return False
        return True

    @property
    def scaling_array(self):

        radius = [0.0,1221.50,3480.00, 5701.00, 6151.0, 6371.00]
        R = [r*10**5 for r in radius]

        rho = [13.0, 13.0, 10.96, 5.03, 3.7, 2.5]

        a1 = (4*np.pi/3)*(rho[1]* R[1]**3)
        a2 = (8*np.pi/15)*(rho[1]* R[1]**5)
        b1 = (4*np.pi/3)*(rho[2]* (R[2]**3 - R[1]**3))
        b2 = (8*np.pi/15)*(rho[2]* (R[2]**5 - R[1]**5))
        c1 = (4*np.pi/3)*(rho[3]* (R[3]**3 - R[2]**3))
        c2 = (8*np.pi/15)*(rho[3]* (R[3]**5 - R[2]**5))
        d1 = (4*np.pi/3)*(rho[4]* (R[4]**3 - R[3]**3))
        d2 = (8*np.pi/15)*(rho[4]* (R[4]**5 - R[3]**5))
        e1 = (4*np.pi/3)*(rho[5]* (R[5]**3 - R[4]**3))
        e2 = (8*np.pi/15)*(rho[5]* (R[5]**5 - R[4]**5))

        I = a2 + b2 +c2 + d2 + e2
        M = a1 + b1 +c1 + d1 + e1

        alpha = self.core_density_scale
        # alpha = 0.9

        new_rho = np.zeros(6)
        gamma = ((I*c1-M*c2)-alpha*(c1*a2 - c2*a1)- alpha*(c1*b2-b1*c2)-(c1*e2 - e1*c2))/(c1*d2-d1*c2)
        beta = (I - alpha * a2 - alpha * b2 - gamma*d2 - e2)/(c2)


        new_rho[0] = alpha * rho[0]
        new_rho[1] = alpha * rho[1]
        new_rho[2] = alpha * rho[2]
        new_rho[3] = beta * rho[3]
        new_rho[4] = gamma * rho[4]
        new_rho[5] = rho[5]

        tmp_array = np.zeros((len(radius), 2))
        if self.is_positive(new_rho):   # and self.is_descending(new_rho): ##turn this on if you want to put hydrostatic equilibrium condition
            tmp_array[:, 0] = radius
            
            tmp_array[0,1] = alpha
            tmp_array[1,1] = alpha
            tmp_array[2,1] = alpha
            tmp_array[3,1] = round(beta,5)
            tmp_array[4,1] = round(gamma,5)
            tmp_array[5,1] = 1
            
        return tmp_array

@jit(nopython=True, nogil=True, cache=True)
def extCalcLayers_scale(cz,
        r_detector,
        prop_height,
        detector_depth,
        rhos_scale,
        coszen_limit,
        radii,
        max_layers):
    """Layer density/distance calculator for each coszen specified.

    Accelerated with Numba if present.

    Parameters
    ----------
    cz             : coszen values (array of float)
    r_detector     : radial position of the detector (float)
    prop_height    : height at which neutrinos are assumed to be produced (float)
    detector_depth : depth at which the detector is buried (float)
    rhos_scale     : scaling factor for different layers (ndarray)
    radii          : radii defining the Earth's layer (ndarray)
    coszen         : coszen values corresponding to the radii above (ndarray)
    max_layers     : maximum number of layers it is possible to cross (int)

    Returns
    -------
    
    density_scale : array of scaling factors for diff layers, flattened from (cz, max_layers)
    
    
    """

    # The densities, distances and number of layers are appended to one long list
    # which is later reshaped into containers of size (# of cz values, max_layers)
    # in the pi_prob3 module

    densities_scale = np.ones((len(cz), max_layers), dtype=FTYPE)
    # distances = np.zeros((len(cz), max_layers), dtype=FTYPE)
    # number_of_layers = np.zeros(len(cz))

    # Loop over all CZ values
    for i, coszen in enumerate(cz):

        r_prop = r_detector+detector_depth+prop_height
        # Compute the full path length
        # path_len = -r_detector * coszen + np.sqrt(r_detector**2. * coszen**2 - (r_detector**2. - r_prop**2.))

        # Determine if there will be a crossing of layer
        # idx is the index of the first inner layer
        idx = np.where(radii<r_detector)[0][0]

        # Deal with paths that do not have tangeants
        if coszen >= coszen_limit[idx]: 
            cumulative_distances = -r_detector * coszen + np.sqrt(r_detector**2. * coszen**2. - r_detector**2. + radii[:idx]**2.)
            # a bit of flippy business is done here to order terms
            # such that numpy diff can work
            segments_lengths = np.diff(np.concatenate((np.array([0.]), cumulative_distances[::-1])))
            segments_lengths = segments_lengths[::-1]
            segments_lengths = np.concatenate((segments_lengths, np.zeros(radii.shape[0] - idx)))

            density_scale = rhos_scale*(segments_lengths > 0.)

        else:
            #
            # Figure out how many layers are crossed twice
            # (meaning we calculate the negative and positive roots for these layers)
            #
            
            calculate_small_root = (coszen < coszen_limit) * (coszen_limit <= coszen_limit[idx])
            calculate_large_root = (coszen_limit>coszen)

            small_roots = - r_detector * coszen * calculate_small_root - np.sqrt(r_detector**2 * coszen**2 - r_detector**2 + radii**2) #, where=calculate_small_root, out=np.zeros_like(radii))
            large_roots = - r_detector * coszen * calculate_large_root + np.sqrt(r_detector**2 * coszen**2 - r_detector**2 + radii**2) #, where=calculate_large_root, out=np.zeros_like(radii))

            # Remove the negative root numbers, and the initial zeros distances
            small_roots = small_roots[small_roots>0]
            small_roots = np.concatenate((np.array([0.]), small_roots))

            # Reverse the order of the large roots
            # That should give the segment distances from the furthest layer to
            # the middle layer
            large_roots = large_roots[::-1]

            # concatenate large and small roots together
            #  this gives the cumulative distance from the detector outward
            full_distances = np.concatenate((np.zeros(1),small_roots[small_roots>0], large_roots[large_roots>0]))

            # Diff the distances and reverse the order 
            # such that the path starts away from the detector
            segments_lengths = np.diff(full_distances)
            segments_lengths = segments_lengths[::-1]


            # The last problem is to match back the densities
            # to the proper array elements. 
            # Densities coming out of the earth core are inverted w.r.t the
            # densities of the ones coming in, with the following exception:
            #
            # - the middle layer and the atmosphere must be counted only once
            #
            # - densities corresponding to layers that are not crossed must be removed
            #
            # NOTE: this assumes that the detector is not positioned in the atmosphere
            #
            # start by removing the layers not crossed from rhos
            inner_layer_mask = coszen_limit>coszen
            density_scale = np.concatenate((rhos_scale[inner_layer_mask],rhos_scale[inner_layer_mask][1:-1][::-1]))

            # As an extra precaution, set all densities that are not crossed to zero
            density_scale*=(segments_lengths>0)

        # number_of_layers[i] = np.sum(segments_lengths > 0.)
        # append to the large list
        for j in range(len(density_scale)):
            # index j may not run all the way to max_layers, unreached indices stay zero
            densities_scale[i, j] = density_scale[j]
            # distances[i, j] = segments_lengths[j]

    return densities_scale


class Layers_scale(object):
    """
    Parameters
    ----------
    scale : 2d array (takes in tmp_array returned by scaling array function)
        
    detector_depth : float
        depth of detector underground in km

    prop_height : float
        the production height of the neutrinos in the atmosphere in km (?)


    Attributes
    ----------
    max_layers : int
            maximum number of layers (this is important for the shape of the
            output! if less than maximumm number of layers are crossed, it's
            filled up with 0s

    density : 1d float array of length (max_layers * len(cz))
            containing density values and filled up with 0s otherwise

    distance : 1d float array of length (max_layers * len(cz))
            containing distance values and filled up with 0s otherwise

    

    """
    def __init__(self, scale, detector_depth=1., prop_height=2.):
        # Load earth model
        

            # The following radii and scaling factors are extracted in reverse order
            # w.r.t the file. The first elements of the arrays below corresponds
            # the Earth's surface, and the following numbers go deeper toward the 
            # planet's core
        self.rhos_scale = scale[..., 1][::-1].astype(FTYPE)
        self.radii = scale[..., 0][::-1].astype(FTYPE)
        r_earth = 6371.0
            

            # Add an external layer corresponding to the atmosphere / production boundary
        self.radii = np.concatenate((np.array([r_earth+prop_height]), self.radii))
        self.rhos_scale  = np.concatenate((np.ones(1, dtype=FTYPE), self.rhos_scale))
        # print(self.rhos_scale)
        self.max_layers = 2 * (len(self.radii))


        


        #
        # Make some checks about the input production height and detector depth
        #
        assert detector_depth > 0, 'ERROR: detector depth must be a positive value'
        assert detector_depth <= r_earth, 'ERROR: detector depth is deeper than one Earth radius!'
        assert prop_height >= 0, 'ERROR: neutrino production height must be positive'

        # Set some other
        self.r_detector = r_earth - detector_depth
        self.prop_height = prop_height
        self.detector_depth = detector_depth

        self.computeMinLengthToLayers_scale()
            


    def computeMinLengthToLayers_scale(self):
        '''
        Deterine the coszen values for which a track will 
        be tangeant to a given layer.

        Given the detector radius and the layer radii:

        - A layer will be tangeant if radii<r_detector

        - Given r_detector and r_i, the limit angle 
          will be:

                sin(theta) = r_i / r_detector

        that angle can then be expressed back into a cosine using
        trigonometric identities

        '''
        coszen_limit = []
        # First element of self.radii is largest radius!
        for i, rad in enumerate(self.radii):
            # Using a cosine threshold instead!
            if rad >= self.r_detector:
                x = 1.
            else:
                x = - np.sqrt(1 - (rad**2 / self.r_detector**2))
            coszen_limit.append(x)
        self.coszen_limit = np.array(coszen_limit, dtype=FTYPE)



    def calcLayers_scale(self, cz):
        """

        Parameters
        ----------
        cz : 1d float array
            Array of coszen values

        """
        print(self.rhos_scale)
        # run external function
        self._density_scale = extCalcLayers_scale(
            cz=cz,
            r_detector=self.r_detector,
            prop_height=self.prop_height,
            detector_depth=self.detector_depth,
            rhos_scale=self.rhos_scale,
            coszen_limit=self.coszen_limit,
            radii=self.radii,
            max_layers=self.max_layers,
        )

    @property
    def density_scale(self):
        
        return self._density_scale

def test_scaling_params():
    pass

if __name__=='__main__':
    from pisa import TARGET
    from pisa.utils.log import set_verbosity, logging
    assert TARGET == 'cpu', "Cannot test functions on GPU, set PISA_TARGET to 'cpu'"
    set_verbosity(1)
    test_scaling_params()





