/*

  author: Timothy C. Arlen
          tca3@psu.edu

  date:   24 Feb 2015

  Class to perform all the setup operations necessary to feed the
  grid_propagator kernel on the GPU.

*/

#include "GridPropagator.h"

// This contains setmix_sin & setmass funcs called from SetMNS()
#include "OscUtils.h"


// Simple accessors:
void GridPropagator::Get_dm_mat(fType dm_mat[3][3])
{
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      dm_mat[i][j] = m_dm[i][j];
}


void GridPropagator::Get_mix_mat(fType mix_mat[3][3][2])
{
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      mix_mat[i][j][0] = m_mix[i][j][0];
      mix_mat[i][j][1] = m_mix[i][j][1];
    }
  }
}


void GridPropagator::GetDensityInLayer( fType* densityInLayer, int len)
{
  // assert that len == m_nczbinss*m_maxLayers?? Not sure why below won't work...
  //if (len != m_nczbins*m_maxLayers) {
  //  fprintf(stderr, "ERROR: INVALID binning in %s",__PRETTY_FUNCTION__);
  //  exit(EXIT_FAILURE);
  // }
  for(int i=0; i<m_nczbins; i++)
    for(int j=0; j<m_maxLayers; j++)
      *(densityInLayer + i*m_maxLayers + j) = *(m_densityInLayer + i*m_maxLayers + j);
}


void GridPropagator::GetDistanceInLayer( fType* distanceInLayer, int len)
{
  // assert that len == m_numberOfLayers*m_maxLayers??
  for(int i=0; i<m_nczbins; i++)
    for(int j=0; j<m_maxLayers; j++)
      *(distanceInLayer + i*m_maxLayers + j) = *(m_distanceInLayer + i*m_maxLayers + j);
}


void GridPropagator::GetNumberOfLayers( int* numLayers, int len)
{
  // assert that len == maxLayers??
  for (int i=0; i<len; i++) *(numLayers +i) = *(m_numberOfLayers+i);
}


GridPropagator::GridPropagator(char* earthModelFile,fType* czcen, int nczbins,
                               fType detector_depth)
/*
  \params:
    * earthModelFile - earth density model file to initialize earth_density
    * czcen - bin centers of coszen fine grid we wish to calculate
    * nczbins - number of coszen bins

  Extremely annoyingly, had to split up "Constructor" into this
  constructor that takes the array/numpy array, and the SetMNS()
  function if one wishes to do anything later on.
*/
{

  m_czcen = czcen;
  m_nczbins = nczbins;

  // These are all defined below:
  m_earthModel = NULL;
  m_densityInLayer = NULL;
  m_distanceInLayer = NULL;
  m_numberOfLayers = NULL;

  //SetEarthDensityParams(earthModelFile,detector_depth);
  //LoadEarthModel(earthModelFile,detector_depth);
  m_earthModel = new EarthDensity(earthModelFile, detector_depth);

}


void GridPropagator::SetMNS(fType dm_solar,fType dm_atm,fType x12,fType x13,fType x23,
                            fType deltacp)
/*
  NOTE: This gets called in the constructor but can also be called
  later to generate a new set of transition matrices for the same
  earth model.

  Expects dm_solar and dm_atm to be in [eV^2], and x_{ij} to be
  sin^2(theta_{ij})

  \params:
    * sinSqT<ij> - sin^2(theta_{ij}) values to use in oscillation calc.
    * dmSolar - delta M_{21}^2 value [eV^2]
    * dmAtm - delta M_{32}^2 value [eV^2] if Normal hierarchy, or
              delta M_{31}^2 value if Inverted Hierarchy (following
              BargerPropagator class).
    * deltacp - \delta_{cp} value to use.
*/
{
  // NOTE: does not support values of x_ij given in sin^2(2*theta_ij)
  fType sin12 = sqrt(x12);
  fType sin13 = sqrt(x13);
  fType sin23 = sqrt(x23);

  // Comment BargerPropagator.cc:
  // "For the inverted Hierarchy, adjust the input
  // by the solar mixing (should be positive)
  // to feed the core libraries the correct value of m32."
  if( dm_atm < 0.0 ) dm_atm -= dm_solar;

  // these functions are all described in mosc.h/mosc.cu
  //setMatterFlavor(nue_type);
  setmix_sin(sin12,sin23,sin13,deltacp,m_mix);
  setmass(dm_solar,dm_atm,m_dm);

  if (VERBOSE) {
    printf("dm21,dm32   : %f %f \n",dm_solar,dm_atm);
    printf("s12,s23,s31 : %f %f %f \n",sin12,sin23,sin13);
    printf("m_dm  : %f %f %f \n",m_dm[0][0],m_dm[0][1],m_dm[0][2]);
    printf("m_dm  : %f %f %f \n",m_dm[1][0],m_dm[1][1],m_dm[1][2]);
    printf("m_dm  : %f %f %f \n",m_dm[2][0],m_dm[2][1],m_dm[2][2]);
    //***********
    //**********
    printf("m_mix : %f %f %f \n",m_mix[0][0][0],m_mix[0][1][0],m_mix[0][2][0]);
    printf("m_mix : %f %f %f \n",m_mix[1][0][0],m_mix[1][1][0],m_mix[1][2][0]);
    printf("m_mix : %f %f %f \n",m_mix[2][0][0],m_mix[2][1][0],m_mix[2][2][0]);
    printf("m_mix : %f %f %f \n",m_mix[0][0][1],m_mix[0][1][1],m_mix[0][2][1]);
    printf("m_mix : %f %f %f \n",m_mix[1][0][1],m_mix[1][1][1],m_mix[1][2][1]);
    printf("m_mix : %f %f %f \n",m_mix[2][0][1],m_mix[2][1][1],m_mix[2][2][1]);
    //***********
  }

}


void GridPropagator::SetEarthDensityParams(fType prop_height,
                                           fType YeI, fType YeO, fType YeM)
/*
  prop_height - atmospheric propagation height in km
*/
{

  prop_height *= kmTOcm;

  fType rDetector = m_earthModel->get_RDetector()*kmTOcm;
  m_earthModel->SetElecFrac(YeI, YeO, YeM);

  // Set member variable pointers/arrays:
  m_maxLayers = m_earthModel->get_MaxLayers();

  size_t layer_size = m_nczbins*m_maxLayers*sizeof(fType);
  m_densityInLayer = (fType*)malloc(layer_size);
  memset(m_densityInLayer,0.0,layer_size);
  m_distanceInLayer = (fType*)malloc(layer_size);
  memset(m_distanceInLayer,0.0,layer_size);

  m_numberOfLayers = (int*)malloc(m_nczbins*sizeof(int));

  for (int i=0; i<m_nczbins; i++) {
    fType coszen = m_czcen[i];
    fType pathLength = DefinePath(coszen, prop_height, rDetector);
    m_earthModel->SetDensityProfile( coszen, pathLength, prop_height );

    //printf("here?\n");
    *(m_numberOfLayers+i) = m_earthModel->get_LayersTraversed();
    if (VERBOSE) {
      printf("coszen: %f, layers traversed: %d, path length: %f\n",coszen,
             *(m_numberOfLayers+i), pathLength/kmTOcm);
    }
    for (int j=0; j < *(m_numberOfLayers+i); j++) {
      fType density = m_earthModel->get_DensityInLayer(j)*
        m_earthModel->get_ElectronFractionInLayer(j);
      *(m_densityInLayer + i*m_maxLayers + j) = density;

      fType distance = m_earthModel->get_DistanceAcrossLayer(j)/kmTOcm;
      *(m_distanceInLayer + i*m_maxLayers + j) = distance;

      if (VERBOSE) {
        printf("  >> Layer: %d, density: %f, distance: %f\n",j,
               *(m_densityInLayer + i*m_maxLayers + j),
               *(m_distanceInLayer + i*m_maxLayers + j));
      }
    }
  }  // end loop over number of cz bins


}


fType GridPropagator::DefinePath(fType cz, fType prod_height, fType rDetector)
/*
  prod_height - Atmospheric production height [cm]
  rDetector - Detector radius [cm]

 */
{

  fType path_length = 0.0;
  fType depth = (fType)m_earthModel->get_DetectorDepth()*kmTOcm;

  if(cz < 0) {
    path_length = sqrt((rDetector + prod_height +depth)*(rDetector + prod_height +depth)
                       - (rDetector*rDetector)*( 1 - cz*cz)) - rDetector*cz;
  } else {
    fType kappa = (depth + prod_height)/rDetector;
    path_length = rDetector*sqrt(cz*cz - 1 + (1 + kappa)*(1 + kappa)) - rDetector*cz;
  }

  return path_length;

}
