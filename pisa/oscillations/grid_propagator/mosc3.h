#ifndef __MOSC3_H__
#define __MOSC3_H__

// Sets all entries to zero
__device__ void clear_complex_matrix( double A[][3][2] );

__device__ void copy_complex_matrix( double A[][3][2], double B[][3][2] );

__device__ void multiply_complex_matrix( double A[][3][2], double B[][3][2], double C[][3][2] );

__device__ void clear_probabilities( double A[3][3] );

__device__ void multiply_complex_matvec( double A[][3][2], double V[][2], double W[][2] );

__device__ void convert_from_mass_eigenstate( int state, int flavor, double pure[][2],
                                              double mix[][3][2] );

__device__ void get_transition_matrix( int nutype, double Enu, double rho, double Len,
                                       double Aout[][3][2], double phase_offset,
                                       double mix[][3][2], double dm[][3]);


#endif
