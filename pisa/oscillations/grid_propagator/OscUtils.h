/*
  \author: Timothy C. Arlen
           tca3@psu.edu

  \date:   24 Feb 2015

  Broke off host functions from mosc.cu (originally from mosc.c) and
  included them here.

*/

#ifndef __OSCUTILS_H__
#define __OSCUTILS_H__

// WARNING: PUT IN SEPARATE INCLUDE FILE
//#define fType double

#define re (0)
#define im (1)


inline void setmix_sin(double s12,double s23,double s13,double dcp,
                double Mix[3][3][2])
{
  double c12,c23,c13,sd,cd;

  if ( s12>1.0 ) s12=1.0;
  if ( s23>1.0 ) s23=1.0;
  if ( s13>1.0 ) s13=1.0;
  //if ( cd >1.0 ) cd =1.0;

  sd = sin( dcp );
  cd = cos( dcp );

  c12 = sqrt(1.0-s12*s12);
  c23 = sqrt(1.0-s23*s23);
  c13 = sqrt(1.0-s13*s13);

  //if (matrix_type == standard_type) {
  Mix[0][0][re] =  c12*c13;
  Mix[0][0][im] =  0.0;
  Mix[0][1][re] =  s12*c13;
  Mix[0][1][im] =  0.0;
  Mix[0][2][re] =  s13*cd;
  Mix[0][2][im] = -s13*sd;
  Mix[1][0][re] = -s12*c23-c12*s23*s13*cd;
  Mix[1][0][im] =         -c12*s23*s13*sd;
  Mix[1][1][re] =  c12*c23-s12*s23*s13*cd;
  Mix[1][1][im] =         -s12*s23*s13*sd;
  Mix[1][2][re] =  s23*c13;
  Mix[1][2][im] =  0.0;
  Mix[2][0][re] =  s12*s23-c12*c23*s13*cd;
  Mix[2][0][im] =         -c12*c23*s13*sd;
  Mix[2][1][re] = -c12*s23-s12*c23*s13*cd;
  Mix[2][1][im] =         -s12*c23*s13*sd;
  Mix[2][2][re] =  c23*c13;
  Mix[2][2][im] =  0.0;

}

inline void setmass(double dms21, double dms23, double dmVacVac[][3])
{

  double delta=5.0e-9;
  double mVac[3];

  mVac[0] = 0.0;
  mVac[1] = dms21;
  mVac[2] = dms21+dms23;

  /* Break any degeneracies */
  if (dms21==0.0) mVac[0] -= delta;
  if (dms23==0.0) mVac[2] += delta;

  dmVacVac[0][0] = dmVacVac[1][1] = dmVacVac[2][2] = 0.0;
  dmVacVac[0][1] = mVac[0]-mVac[1]; dmVacVac[1][0] = -dmVacVac[0][1];
  dmVacVac[0][2] = mVac[0]-mVac[2]; dmVacVac[2][0] = -dmVacVac[0][2];
  dmVacVac[1][2] = mVac[1]-mVac[2]; dmVacVac[2][1] = -dmVacVac[1][2];

}

#endif
