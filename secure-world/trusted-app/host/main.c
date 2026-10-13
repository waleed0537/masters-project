#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tee_client_api.h>
#include <rawtfnet_ta.h>

#define FEAT_N (32 * 23 * 16)
#define FEATURES_PATH "/mnt/host/current.bin"
#define LABEL_PATH    "/mnt/host/current_label.txt"

int main(void)
{
	static float input[FEAT_N];
	char label[64] = {0};
	float logits[2] = { 0.f, 0.f };

	FILE *f = fopen(FEATURES_PATH, "rb");
	if (!f) errx(1, "cannot open %s", FEATURES_PATH);
	size_t n = fread(input, sizeof(float), FEAT_N, f);
	fclose(f);
	if (n != FEAT_N) errx(1, "expected %d floats, got %zu", FEAT_N, n);

	FILE *lf = fopen(LABEL_PATH, "r");
	if (lf) {
		if (fgets(label, sizeof(label), lf)) {
			label[strcspn(label, "\r\n")] = 0;
		}
		fclose(lf);
	} else {
		strcpy(label, "unknown");
	}

	printf("================================================\n");
	printf(" Audio file being checked: %s\n", label);
	printf("================================================\n");

	TEEC_Result res;
	TEEC_Context ctx;
	TEEC_Session sess;
	TEEC_Operation op;
	TEEC_UUID uuid = TA_RAWTFNET_UUID;
	uint32_t origin;

	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS) errx(1, "TEEC_InitializeContext failed 0x%x", res);

	res = TEEC_OpenSession(&ctx, &sess, &uuid, TEEC_LOGIN_PUBLIC,
			       NULL, NULL, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_OpenSession failed 0x%x origin 0x%x", res, origin);

	memset(&op, 0, sizeof(op));
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
					 TEEC_MEMREF_TEMP_INPUT,
					 TEEC_MEMREF_TEMP_OUTPUT,
					 TEEC_NONE);
	op.params[0].tmpref.buffer = input;
	op.params[0].tmpref.size = sizeof(input);
	op.params[1].tmpref.buffer = label;
	op.params[1].tmpref.size = strlen(label);
	op.params[2].tmpref.buffer = logits;
	op.params[2].tmpref.size = sizeof(logits);

	printf("Sending to secure world (TrustZone)...\n");
	res = TEEC_InvokeCommand(&sess, TA_RAWTFNET_CMD_INFERENCE, &op, &origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand failed 0x%x origin 0x%x", res, origin);

	printf("------------------------------------------------\n");
	printf(" RESULT for %s\n", label);
	/* Output order is [spoof, bonafide]: the model was trained with bonafide = 1,
	 * spoof = 0, so logits[1] is the bonafide logit and the decision score. */
	printf(" Spoof logit:    %f\n", logits[0]);
	printf(" Bonafide logit: %f\n", logits[1]);
	printf(" Verdict: %s\n", logits[1] > 0.f ? "BONAFIDE (real)" : "SPOOF (fake)");
	printf("------------------------------------------------\n");

	TEEC_CloseSession(&sess);
	TEEC_FinalizeContext(&ctx);
	return 0;
}
