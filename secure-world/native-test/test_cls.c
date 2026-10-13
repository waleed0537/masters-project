#include <stdio.h>

/* adjust if the signature from step 4 differs */
void model(const float features[restrict 1][32][23][16], float logits[restrict 2]);

static float in[1][32][23][16];
static float out[2];

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open input\n"); return 1; }
    size_t n = fread(in, sizeof(float), 32*23*16, f);
    fclose(f);
    if (n != 32*23*16) { printf("expected %d floats, got %zu\n", 32*23*16, n); return 1; }
    model(in, out);
    printf("logits: %f %f\n", out[0], out[1]);
    return 0;
}
