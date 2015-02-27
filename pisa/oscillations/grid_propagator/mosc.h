/*
 * author: Timothy C. Arlen
 *
 * date: 31 Jan 2014
 *
 * Broke off code from probGpu.cu to put in it's own file, to make
 * analagous to original prob3++.
 *
 */

#ifndef MOSCHINCLUDED
#define MOSCHINCLUDED

#define elec (0)
#define muon (1)
#define tau  (2)
#define re (0)
#define im (1)

typedef enum nu_type {
  data_type,
  nue_type,
  numu_type,
  nutau_type,
  sterile_type,
  unknown_type} NuType;


typedef enum matrix_type {
  standard_type,
  barger_type} MatrixType;


__device__ void getM(double Enu, double rho,
                     double Mix[][3][2], double dmVacVac[][3], int antitype,
                     double dmMatMat[][3], double dmMatVac[][3]);
__device__ void getA(double L, double E, double rho,
                     double Mix[][3][2], double dmMatVac[][3],
                     double dmMatMat[][3], int antitype, double A[3][3][2],
                     double phase_offset);
__device__ void get_product(double L, double E, double rho,double Mix[][3][2],
                            double dmMatVac[][3], double dmMatMat[][3],
                            int antitype,
                            double product[][3][3][2]);


#endif /* MOSCHINCLUDED */
