/*

  \author: Timothy C. Arlen
           tca3@psu.edu

  \date:   24 Feb 2015

  This helper class performs the setup required to do oscillation
  probability calculations on a grid. Works in conjunction with the
  grid_propagator kernel, by defining all needed

  Namely, it must define and initialize (on the host)
    -- "dm" matrix
    -- "mix" matrix
    -- Earth Model setup, and inputs to kernel:
      * MaxLayers
      * numberOfLayers
      * densityInLayer
      * distanceInLayer
*/

#ifndef __GRIDPROPAGATOR_H__
#define __GRIDPROPAGATOR_H__

#include "EarthDensity.h"
#include "constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


class GridPropagator
{
 public:

  GridPropagator(char* earthModelFile, fType* czcen, int nczbins, fType detector_depth);

  //void Propagate(fType sinSqT12, fType sinSqT13, fType sinSqT23,
  //fType dmSolar, fType dmAtm, fType deltacp);

  // SetMNS() - re-initialize mixing matrices...
  void SetEarthDensityParams(fType prop_height, fType YeI, fType YeO, fType YeM);
  void  SetMNS(fType dm_solar,fType dm_atm,fType x12,fType x13,fType x23,
               fType deltacp);
  fType DefinePath(fType cz, fType prod_height, fType rDetector);

  // Public member variables

  ////////////////////////////////////
  // Simple Accessors - returns the pointer to host memory location.
  // Recall that the first dimension (row) is cz_cen, & the 2nd dimension
  // is the layer in m_numberOfLayers.
  void Get_dm_mat(fType dm_mat[3][3]);
  void Get_mix_mat(fType mix_mat[3][3][2]);
  void GetDensityInLayer( fType* densityInLayer, int len);
  void GetDistanceInLayer( fType* distanceInLayer, int len);
  void GetNumberOfLayers( int* numLayers, int len);
  inline int GetMaxLayers( void ) { return m_maxLayers; }

  // Write the proper accessors for m_dm and m_mix and keep private?

  // Destructor:
  ~GridPropagator() {
    delete m_earthModel;

    //free(m_mix);
    free(m_densityInLayer);
    free(m_distanceInLayer);
    free(m_numberOfLayers);
  }

 private:

  //  void  SetEarthDensityParams(char* earthModelFile);
  //void  setmix_sin(double s12,double s23,double s13,double dcp);
  //void  setmass(double dms21, double dms23);

  EarthDensity* m_earthModel;

  //fType  m_dm[3][3];
  //fType (*m_mix)[3][2];
  fType m_dm[3][3];
  fType m_mix[3][3][2];

  fType* m_densityInLayer;
  fType* m_distanceInLayer;
  int*   m_numberOfLayers;
  int    m_maxLayers;

  fType* m_czcen;
  int    m_nczbins;

};

#endif
