from pisa.utils.fileio import from_file
import numpy as np
import numba

# external calculation function to cope with numba
@numba.jit('''Tuple((int32[:],float32[:],float32[:]))(
               float32[:],
               float32,
               float32,
               float32,
               int32,
               float32,
               float32[:],
               float32[:],
               float32[:],
               float32,
               float32[:],
               float32[:])''',
              nopython=True, nogil=True, cache=True)
def int_calc_layers(cz, 
                    rDetector,
                    prop_height,
                    DetectorDepth,
                    max_layers,
                    MinDetectorDepth,
                    Rhos,
                    YeFrac,
                    YeOuterRadius,
                    default_elec_frac,
                    coszen_limit,
                    Radii):

    # something to store the final results in
    shape = (numba.int64(len(cz)),numba.int64(max_layers))
    n_layers = np.zeros(shape[0], dtype=np.int32)
    distance = np.zeros(shape=shape, dtype=np.float32)
    density = np.zeros(shape=shape, dtype=np.float32)

    # loop over all CZ values
    for k,coszen in enumerate(cz):

        TotalEarthLength = -2.0 * coszen * rDetector
        
        # to store results
        TraverseRhos = np.zeros(max_layers,dtype=np.float32)
        TraverseDistance = np.zeros(max_layers,dtype=np.float32)
        TraverseElectronFrac = np.zeros(max_layers,dtype=np.float32)
       
        # above horizon 
        if coszen >= 0:
            kappa = (DetectorDepth + prop_height)/rDetector
            PathLength = rDetector * np.sqrt(np.square(coszen) - 1 + np.square(1 + kappa)) - rDetector * coszen
            #Path through the air:
            kappa = DetectorDepth/rDetector
            lam = coszen + np.sqrt(np.square(coszen) - 1 + (1 + kappa) * (1 + kappa))
            lam *= rDetector
            pathThroughAtm = (prop_height * (prop_height + 2. * DetectorDepth + 2.0*rDetector))/(PathLength + lam)
            pathThroughOuterLayer = PathLength - pathThroughAtm
            TraverseRhos[0] = 0.0
            TraverseDistance[0] = pathThroughAtm
            TraverseElectronFrac[0] = default_elec_frac
            Layers = 1
            
            # in that case the neutrino passes through some earth (?)
            if DetectorDepth > MinDetectorDepth:
                TraverseRhos[1] = Rhos[0]
                TraverseDistance[1] = pathThroughOuterLayer
                TraverseElectronFrac[1] = YeFrac[-1]
                Layers+=1

        # below horizon
        else:
            PathLength =    np.sqrt(np.square((rDetector + prop_height + DetectorDepth)) \
                            - np.square(rDetector) * (1 - np.square(coszen))) \
                            - rDetector * coszen
            # path through air (that's down from production height in the atmosphere?)
            TraverseRhos[0] = 0.
            TraverseDistance[0] = prop_height * (prop_height + DetectorDepth + 2. * rDetector) / PathLength
            # why default here?
            TraverseElectronFrac[0] = default_elec_frac
            iTrav = 1

            # path through the final layer above the detector (if necessary)
            # Note: outer top layer is assumed to be the same as the next layer inward.
            if (DetectorDepth > MinDetectorDepth):
                TraverseRhos[1] = Rhos[0]
                TraverseDistance[1] = PathLength - TotalEarthLength - TraverseDistance[0]
                TraverseElectronFrac[1] = YeFrac[-1]
                iTrav += 1

            Layers = 0
            # see how many layers we will pass
            for val in coszen_limit:
                if coszen < val:
                    Layers += 1
            
            # the zeroth layer is the air!
            # and the first layer is the top layer (if detector is not on surface)
            for i in range(Layers):
                # this is the density
                TraverseRhos[i+iTrav] = Rhos[i]
                # ToDo why default? is this air with density 0 and electron fraction just doesn't matter?
                TraverseElectronFrac[i+iTrav] = default_elec_frac
                for iRad in range(len(YeOuterRadius)):
                    # why 1.001?
                    if Radii[i] < (YeOuterRadius[iRad] * 1.001):
                        TraverseElectronFrac[i+iTrav] = YeFrac[iRad]
                        break
                
                # now calculate the distance travele in layer 
                c2 = np.square(coszen)
                R2 = np.square(rDetector)
                s1 = np.square(Radii[i]) - R2*(1 -c2)
                s2 = np.square(Radii[i+1]) - R2*(1 -c2)
                CrossThis = 2. * np.sqrt(s1)
                if i < Layers - 1:
                    CrossNext = 2. * np.sqrt(s2)
                    TraverseDistance[i+iTrav]  =  0.5 * (CrossThis - CrossNext)
                else:
                    TraverseDistance[i+iTrav]  =  CrossThis

                #assumes azimuthal symmetry
                if 0 < i and i < Layers:
                    index = 2 * Layers - i + iTrav - 1
                    TraverseRhos[index] = TraverseRhos[i+iTrav-1]
                    TraverseDistance[index] = TraverseDistance[i+iTrav-1]
                    TraverseElectronFrac[index] = TraverseElectronFrac[i+iTrav-1]

            # that is now the total
            Layers = 2 * Layers + iTrav - 1

        n_layers[k] = np.int32(Layers)
        density[k] = TraverseRhos * TraverseElectronFrac
        distance[k] = TraverseDistance
    return n_layers, density.ravel(), distance.ravel()

class Layers(object):
    ''' class used to calculate the path through earth for a given layer model with
        densities (PREM), the electron fractions (Ye) and an array of coszen values

        Params:
        ------

        prem_file : str
            path to PREM file containing layer radii and densities as white space separated txt
        DetectorDepth : float 
            depth of detector underground in km
        prop_height : float
            the production height of the neutrinos in the atmosphere in km (?)


        Methods:
        -------

        SetElecFrac : float, float, float
            set the three electron fractions YeI, YeO, YeM (where I = inner, O = middle, M = outeri ?!)
        calc_layers : 1d float array
            run the calculation for an array of CZ values

        Attributes:
        ----------
        
        max_layers : int
                maximum number of layers (this is important for the shape of the output!
                if less than maximumm number of layers are crossed, it's filled up with 0s
        n_layers : 1d int array of length len(cz)
                number of layers crossed for every CZ value
        density : 1d float array of length (max_layers * len(cz))
                containing density values and filled up with 0s otherwise
        distance : 1d float array of length (max_layers * len(cz))
                containing distance values and filled up with 0s otherwise
    '''

    def __init__(self, prem_file, DetectorDepth=1., prop_height=2.):
        # load earth model
        prem = from_file(prem_file, as_array=True)
        self.Rhos = prem[...,1][::-1].astype(np.float32)
        self.Radii = prem[...,0][::-1].astype(np.float32)
        rEarth = prem[-1][0]
        self.rDetector = rEarth - DetectorDepth
        self.default_elec_frac = 0.5
        self.prop_height = prop_height
        self.DetectorDepth = DetectorDepth
        self.MinDetectorDepth = 1.0e-3 # <-- Why? // [km] so min is ~ 1 m
        N_prem = len(self.Radii) - 1
        self.max_layers = 2 * N_prem + 1
        # change outermost radius to a bit underground, where the detector
        if (self.DetectorDepth >= self.MinDetectorDepth):
            self.Radii[0] -= DetectorDepth
            self.max_layers += 1
        self.ComputeMinLengthToLayers()

    def SetElecFrac(self, YeI, YeO, YeM):
        self.YeFrac = np.array([YeI, YeO, YeM],dtype=np.float32)
        # and these numbers are just hard coded for some reason
        self.YeOuterRadius = np.array([1121.5, 3480.0, self.rDetector],dtype=np.float32)

    def ComputeMinLengthToLayers(self):
        # compute which layer is tangeted at which angle
	coszen_limit = []
    	#first element of self.Radii is largest radius!
        for i,rad in enumerate(self.Radii):
	    # Using a cosine threshold instead!
	    if i == 0:
		x = 0
            else:
                x = - np.sqrt(1 - (np.square(rad) / np.square(self.rDetector)))
	    coszen_limit.append(x)
        self.coszen_limit = np.array(coszen_limit,dtype=np.float32)

    def calc_layers(self,cz):
        # run external function
        out = int_calc_layers(cz,
                              self.rDetector,
                              self.prop_height,
                              self.DetectorDepth,
                              self.max_layers,
                              self.MinDetectorDepth,
                              self.Rhos,
                              self.YeFrac,
                              self.YeOuterRadius,
                              self.default_elec_frac,
                              self.coszen_limit,
                              self.Radii)
        self._n_layers = out[0]
        self._density = out[1]
        self._distance = out[2]

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def density(self):
        return self._density

    @property
    def distance(self):
        return self._distance

if __name__ == '__main__':
    layer = Layers('osc/PREM_4layer.dat')
    layer.SetElecFrac( 0.4656, 0.4656, 0.4957)
    cz = np.linspace(-1,1,1e5,dtype=np.float32)
    layer.calc_layers(cz);
    print layer.n_layers
    print layer.density
    print layer.distance
