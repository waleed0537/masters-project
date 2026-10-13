#include "ta_math.h"

/* Minimal libm subset for the generated RawTFNet model.
   Accuracy is sufficient for float32 inference; verified against
   the native/ONNX reference in the thesis PoC. */

float fabsf(float x)
{
	return x < 0.0f ? -x : x;
}

float fmaxf(float a, float b)
{
	if (a != a) return b;   /* NaN handling */
	if (b != b) return a;
	return a > b ? a : b;
}

float sqrtf(float x)
{
	if (x < 0.0f) return 0.0f / 0.0f;  /* NaN */
	if (x == 0.0f) return 0.0f;
	/* Newton-Raphson, seeded well; converges fast for f32 range */
	float guess = x;
	for (int i = 0; i < 30; i++)
		guess = 0.5f * (guess + x / guess);
	return guess;
}

/* expf via range reduction: exp(x) = 2^k * exp(r),
   x = k*ln2 + r, |r| <= ln2/2, then a Taylor series on r. */
float expf(float x)
{
	const float LN2 = 0.6931471805599453f;
	if (x > 88.0f)  return FLT_MAX;      /* overflow guard */
	if (x < -88.0f) return 0.0f;         /* underflow guard */

	int k = (int)(x / LN2 + (x >= 0 ? 0.5f : -0.5f));
	float r = x - (float)k * LN2;

	/* Taylor series for exp(r) */
	float term = 1.0f, sum = 1.0f;
	for (int n = 1; n < 12; n++) {
		term *= r / (float)n;
		sum += term;
	}

	/* multiply by 2^k */
	float result = sum;
	if (k > 0) {
		for (int i = 0; i < k; i++) result *= 2.0f;
	} else {
		for (int i = 0; i < -k; i++) result *= 0.5f;
	}
	return result;
}

/* logf via mantissa/exponent decomposition:
   x = m * 2^e, m in [1,2); log(x) = e*ln2 + log(m). */
float logf(float x)
{
	const float LN2 = 0.6931471805599453f;
	if (x <= 0.0f) return -FLT_MAX;   /* -inf-ish guard */

	int e = 0;
	while (x >= 2.0f) { x *= 0.5f; e++; }
	while (x < 1.0f)  { x *= 2.0f; e--; }

	/* now x in [1,2); use atanh series on (x-1)/(x+1) */
	float t = (x - 1.0f) / (x + 1.0f);
	float t2 = t * t;
	float sum = 0.0f, tp = t;
	for (int n = 1; n <= 15; n += 2) {
		sum += tp / (float)n;
		tp *= t2;
	}
	return 2.0f * sum + (float)e * LN2;
}

