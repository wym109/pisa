#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__


/*
 * Swap out single/double precision by changing this #define statement, then
 * fType is defined appropriately.
 *
 * See Also
 * --------
 * pisa/utils/const.py for the Python-side definition of data size (these must
 * be set to the same thing, or undefined/untested behavior will result!)
 *
 */
#define SINGLE_PRECISION
//#define DOUBLE_PRECISION


#ifdef SINGLE_PRECISION
#define fType float
#endif

#ifdef DOUBLE_PRECISION
#define fType double
#endif

const fType kmTOcm = 1.0e5;

/*
 * Debugging
 */
#define VERBOSE false

#endif
