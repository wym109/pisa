/***************************************************
 * Swig module description file for wrapping the
 * the GridPropagator class.
 ***************************************************/

%module GridPropagator

 //#define fType float
%include "constants.h"

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_TYPECHECK_DOUBLE_ARRAY
#include "GridPropagator.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (fType* IN_ARRAY1, int DIM1) {(fType* czcen, int nczbins)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* numLayers, int len)};
%apply (fType* INPLACE_ARRAY1, int DIM1 ) {(fType* densityInLayer, int len)};
%apply (fType* INPLACE_ARRAY1, int DIM1 ) {(fType* distanceInLayer, int len)};
%apply (fType INPLACE_ARRAY2[ANY][ANY]) {(fType dm_mat[3][3])};
%apply (fType INPLACE_ARRAY3[ANY][ANY][ANY]) {(fType mix_mat[3][3][2])};

%include "GridPropagator.h"
