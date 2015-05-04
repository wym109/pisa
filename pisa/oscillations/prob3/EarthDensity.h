#ifndef _EarthDensity_
#define _EarthDensity_

#include <map>
#include <vector>
#include <fstream>
#include <math.h>
#include <string>


//
//  EarthDensity is an object designed to represent the density profile
//  of the earth, or other planet. It can read in a radial density profile
//  via a user-specified text file. Once the density
//  profile is loaded for given neutrino trajectory, the distance
//  across each layer it will traverse is computed. These are later
//  available for code wishing to propagate the neutrino. In essense
//  a density profile specific to a neutrino's path is created
//
// 20050816 rvw
// 20081001 rvw (update)

//  User-specified density profiles must contain two columns of floating point
//  numbers, the first is the radial distance [km] from the sphere center, the second
//  is the density [g/cm^3] for
//  0.      x_0
//  r_1     x_1
//  ..      ..
//  r_n     x_n
//  the last entry should contain the radius of the sphere.
//  each x_i represents the density up to and including r_i
//  the entry for zero radial density must be included.

/*
  20140512 tca - MAJOR UPDATE.

  1) In order to more accurately model the matter potential through the
  earth, we've allowed for the electron fraction to take on different
  values from 0.5 (_YeFrac) in each earth layer (_YeOuterRadius).

  2) Furthermore, we changed the code so that a detector can now be
  placed at some distance below the surface of the earth, as long as
  it's still within the first layer of Earth, defined in the earth
  model file. Note that now REarth is changed to RDetector in the
  calculations, and the corresponding get functions are also changed.

*/


using namespace std;

class EarthDensity
{
 public:
  // default contstructor for the Earth, and a radial density profile
  // as specified by the SK 3f paper: PRD.74.032002 (2006)
  EarthDensity( );

  // constructor for a user-specified density profile, see PREM.dat
  EarthDensity( const char *, double detectorDepth = 0.0);
  virtual ~EarthDensity( );

  void init();

  // Load the Density profile for a given neutrino path:
  //
  //    ProdHt - amount of vacuum to travese before matter
  //		corresponds to height in atmosphere for atmospheric neutrinos
  //	  	or distance from neutrion source for extraplanetary nu's
  virtual void SetDensityProfile( double CosineZ, double PathLength , double ProdHt);

  // Set electron fraction: YeI - Ye (Inner Core), YeO - Ye (Outer Core)
  //   YeM - Ye (Mantle)
  void SetElecFrac(double YeI, double YeO, double YeM);

  // Read in radii and densities
  void Load();
  double GetEarthRadiuskm( ) { return  REarth; };

  // The next accessor functions are only available after a call to
  // SetDensityProfile...

  // number of layers the current neutrino sees
  int get_LayersTraversed( ) {return Layers;};
  double get_DetectorDepth() {return DetectorDepth;}
  double get_RDetector()     {return RDetector;}  // km

  // self-explanatory
  double get_DistanceAcrossLayer( int i) { return _TraverseDistance[i];};
  double get_DensityInLayer( int i) { return _TraverseRhos[i];};
  double get_ElectronFractionInLayer(int i) {return _TraverseElectronFrac[i];}

  double get_MaxLayers() { return 2.0*_Radii.size(); }

  // return total path length through the sphere, including vacuum layers
  double get_Pathlength(){
    double Sum = 0.0;
    for( int i=0 ; i < Layers ; i++ )
      Sum += _TraverseDistance[i];
    return Sum;

  }

  virtual void LoadDensityProfile( const char * );

 protected:
  // Computes the minimum pathlenth ( zenith angle ) a track needs to cross
  // each of the radial layers of the earth
  void ComputeMinLengthToLayers( );

  string DensityFileName;

  map<double, double>	_CosLimit;
  map<double, double>	_density;

  vector< double >	_Radii;
  vector< double >	_Rhos;
  vector< double >      _YeOuterRadius;
  vector< double >      _YeFrac;

  double * _TraverseDistance;
  double * _TraverseRhos;
  double * _TraverseElectronFrac;

  double _YeI;
  double _YeO;
  double _YeM;

  double REarth;
  double RDetector;
  double DetectorDepth;
  double MinDetectorDepth;
  int    Layers;

};


#endif



