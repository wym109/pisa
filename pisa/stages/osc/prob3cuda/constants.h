/*
# NOTE: 2015-05-21 (TCA) attempted to use single precision, and at
# least on my system, I got all junk in the output of my osc prob
# maps. Unfortunately, I don't want to spend the time right now to
# figure out WHY this is the case, but until someone figures this out,
# keep fType to double and np.float64.
*/


#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define fType float
const fType kmTOcm = 1.0e5;

// debugging purposes:
#define VERBOSE false

#endif
