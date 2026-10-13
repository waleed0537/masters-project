#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <string.h>
#include <rawtfnet_ta.h>

#define FEAT_N (32 * 23 * 16)

void model(const float features[1][32][23][16], float logits[2]);

TEE_Result TA_CreateEntryPoint(void) { return TEE_SUCCESS; }
void TA_DestroyEntryPoint(void) { }

TEE_Result TA_OpenSessionEntryPoint(uint32_t pt,
			TEE_Param params[4] __unused, void **sess __unused)
{
	uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
				       TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	if (pt != exp)
		return TEE_ERROR_BAD_PARAMETERS;
	return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *sess __unused) { }

static TEE_Result run_inference(uint32_t pt, TEE_Param params[4])
{
	uint32_t exp = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
				       TEE_PARAM_TYPE_MEMREF_INPUT,
				       TEE_PARAM_TYPE_MEMREF_OUTPUT,
				       TEE_PARAM_TYPE_NONE);
	if (pt != exp)
		return TEE_ERROR_BAD_PARAMETERS;

	if (params[0].memref.size != FEAT_N * sizeof(float))
		return TEE_ERROR_BAD_PARAMETERS;
	if (params[2].memref.size < 2 * sizeof(float))
		return TEE_ERROR_SHORT_BUFFER;

	/* Copy the filename label into a safe local buffer for logging */
	char label[64] = {0};
	size_t lsz = params[1].memref.size;
	if (lsz > sizeof(label) - 1) lsz = sizeof(label) - 1;
	memcpy(label, params[1].memref.buffer, lsz);

	const float (*features)[32][23][16] =
		(const float (*)[32][23][16])params[0].memref.buffer;
	float logits[2] = { 0.0f, 0.0f };

	/* PROOF: this line is printed FROM INSIDE THE SECURE WORLD,
	   visible in the Secure World console, tied to this exact file. */
	IMSG("=== SECURE WORLD: processing audio file: %s ===", label);

	model(features, logits);

	IMSG("=== SECURE WORLD: %s -> logits=[%d, %d] (x1000) ===",
	     label, (int)(logits[0]*1000), (int)(logits[1]*1000));

	memcpy(params[2].memref.buffer, logits, sizeof(logits));
	params[2].memref.size = sizeof(logits);

	return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess __unused, uint32_t cmd,
			uint32_t pt, TEE_Param params[4])
{
	switch (cmd) {
	case TA_RAWTFNET_CMD_INFERENCE:
		return run_inference(pt, params);
	default:
		return TEE_ERROR_NOT_SUPPORTED;
	}
}
