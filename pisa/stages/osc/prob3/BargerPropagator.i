/***************************************************
 * Swig module description file for wrapping the
 * the BargerPropagator class.
 ***************************************************/

%module BargerPropagator

%{
#include "BargerPropagator.h"
#include "EarthDensity.h"
#include "NeutrinoPropagator.h"

#include "mosc3.h"
#include "mosc.h"
%}

%include "BargerPropagator.h"
