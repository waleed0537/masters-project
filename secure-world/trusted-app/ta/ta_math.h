#ifndef TA_MATH_H
#define TA_MATH_H

#include <stdlib.h>   /* TA dev kit provides malloc/free/size_t */

/* Minimal libm subset for the generated model, since the TA
   dev kit does not ship math.h / float.h. */

float expf(float x);
float sqrtf(float x);
float fabsf(float x);
float fmaxf(float a, float b);
float logf(float x);

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38F
#endif
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-07F
#endif
#ifndef INFINITY
#define INFINITY (1.0e30f * 1.0e30f)
#endif

#endif /* TA_MATH_H */
